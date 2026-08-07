"""
Flask app: text -> G2P -> expected phonemes, audio -> Wav2Vec2 -> spoken
phonemes, aligned and scored into provisional pronunciation feedback and an
adaptive per-phoneme practice loop.

Scoring/tokenization/alignment/mastery policy all live in dedicated modules
(one authoritative implementation each); this file is HTTP wiring plus the
Wav2Vec2 acoustic step:

    tokenization.py                 canonical phoneme tokenization
    phoneme_vectors_professional.py distance / inventory / substitution policy
    scoring.py                      DP alignment + metrics
    mastery.py                      soft, per-recording Beta mastery
    audio_quality.py                the scorability gate
    assessment.py / services.py     evidence-aware level + exercise selection
"""

from __future__ import annotations

import base64
import os
import sys
import uuid
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import noisereduce as nr
except Exception:
    nr = None

try:
    import soundfile as sf
except Exception:
    sf = None

# Load .env (API keys, LLM config) before importing content.
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

import librosa
import numpy as np
import torch
from flask import Flask, jsonify, render_template, request, send_from_directory
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

import assessment as assessment_mod
from arpabet import metrics_to_internal_ipa, to_public_arpabet
import content
import db
import mastery
import services
from app_datetime import format_db_datetime, parse_db_datetime
from audio_quality import AudioQualityDecision, analyze_audio_quality, should_update_mastery
from cleanvoice_service import (
    CleanvoiceProcessingError,
    cleanvoice_configured,
    cleanvoice_enabled,
    cleanvoice_sdk_available,
    cleanvoice_strict,
    enhance_recording,
)
from g2p_service import (
    G2P_DIR,
    HETERONYMS_PATH,
    IPA_DICT_PATH,
    g2p_convert_with_metadata,
    get_g2p_mode,
    heteronym_resolution_active,
    load_g2p_engine,
    validate_heteronym_lexicon,
)
from phoneme_vectors_professional import (
    canonical_inventory,
    canonicalize_phoneme,
    panphon_available,
    scoring_engine,
    scoring_trusted,
    validate_g2p_inventory,
    validate_panphon_inventory,
)
from scoring import align_phonemes, calculate_metrics, to_api_alignment
from tokenization import (
    PHONEME_GUIDE,
    ipa_reading_guide,
    normalize_ipa,
    tokenize_ctc_prediction,
    tokenize_reference_ipa,
)

app = Flask(__name__)


def jsonify_arpabet(payload: Any):
    """Serialize an internal IPA payload for the ARPAbet browser contract."""
    return jsonify(to_public_arpabet(payload))


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
MODEL_PATH = BASE_DIR / "model" / "my_wav2vec2_phoneme_model"
VOICE_FILTERING_DIR = BASE_DIR / "voice-filtering"

# Recordings are PRIVATE and TEMPORARY by default: the original upload, the
# converted WAV, and the noise-reduced WAV are all deleted after processing.
# Set RETAIN_AUDIO=1 (env) only if you explicitly want to keep them (e.g. to
# offer noise-reduced playback). Never enable this for shared/production use.
RETAIN_AUDIO = os.environ.get("RETAIN_AUDIO", "0") == "1"
MAX_INLINE_PLAYBACK_BYTES = 12 * 1024 * 1024

UPLOAD_FOLDER.mkdir(exist_ok=True)
db.init_db()

if str(VOICE_FILTERING_DIR) not in sys.path:
    sys.path.insert(0, str(VOICE_FILTERING_DIR))

try:
    from audio_filter_safe import process_audio_file as safe_process_audio_file
except Exception as exc:
    safe_process_audio_file = None
    print("audio_filter_safe is unavailable. Falling back to legacy preprocessing.")
    print("Reason:", repr(exc))

processor: Optional[Wav2Vec2Processor] = None
model: Optional[Wav2Vec2ForCTC] = None
device = "cuda" if torch.cuda.is_available() else "cpu"


class AudioDecodeError(RuntimeError):
    """Raised when an uploaded browser recording cannot be decoded."""


# -----------------------------
# Wav2Vec2 model loading
# -----------------------------
def load_wav2vec_model() -> None:
    """Load the trained wav2vec2 phoneme model from the local model folder."""
    global processor, model
    if processor is not None and model is not None:
        return
    if not (MODEL_PATH / "config.json").exists():
        raise FileNotFoundError(
            f"Model files were not found in: {MODEL_PATH}. "
            "Put your downloaded trained model files inside this folder first."
        )
    processor = Wav2Vec2Processor.from_pretrained(str(MODEL_PATH))
    model = Wav2Vec2ForCTC.from_pretrained(str(MODEL_PATH))
    model.to(device)
    model.eval()


# -----------------------------
# Audio conversion + quality diagnostics + transcription
# -----------------------------
def _find_ffmpeg_executable() -> Optional[str]:
    """Locate FFmpeg from an explicit env var, PATH, or imageio-ffmpeg."""
    configured = os.environ.get("FFMPEG_BINARY", "").strip()
    if configured:
        configured_path = Path(configured)
        if configured_path.exists():
            return str(configured_path)
        found = shutil.which(configured)
        if found:
            return found

    found = shutil.which("ffmpeg")
    if found:
        return found

    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def convert_audio_to_wav(input_path: Path) -> Path:
    """Convert browser audio to 16kHz mono WAV when FFmpeg is available."""
    input_path = Path(input_path)
    output_path = input_path.with_name(input_path.stem + "_converted.wav")
    if output_path.exists():
        return output_path
    if input_path.suffix.lower() in {".wav", ".wave"}:
        return input_path
    ffmpeg = _find_ffmpeg_executable()
    if ffmpeg is None:
        raise AudioDecodeError(
            "This browser sent compressed audio that needs FFmpeg to decode. "
            "Install FFmpeg or run `pip install imageio-ffmpeg`, restart the app, "
            "then try recording again."
        )
    command = [
        ffmpeg, "-y", "-i", str(input_path),
        "-ac", "1", "-ar", "16000", str(output_path),
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or "").strip().splitlines()
        tail = detail[-1] if detail else "FFmpeg could not decode the uploaded recording."
        raise AudioDecodeError(f"Could not decode the uploaded recording: {tail}") from exc
    if completed.returncode != 0 or not output_path.exists() or output_path.stat().st_size == 0:
        raise AudioDecodeError("Could not decode the uploaded recording into WAV.")
    return output_path


def _extension_for_mimetype(mimetype: str) -> str:
    """Use a plausible extension so FFmpeg can probe mobile browser uploads."""
    mimetype = (mimetype or "").lower()
    if "wav" in mimetype or "wave" in mimetype:
        return ".wav"
    if "ogg" in mimetype:
        return ".ogg"
    if "mp4" in mimetype or "mpeg" in mimetype or "aac" in mimetype:
        return ".m4a"
    if "webm" in mimetype or "opus" in mimetype:
        return ".webm"
    return ".webm"


def _apply_noise_reduction(wav_path: Path, reduced_path: Path):
    """Best-effort mild noise reduction. Returns
    (model_input_path, noise_reduction_applied, preprocessing_pipeline)."""
    if safe_process_audio_file is not None:
        try:
            safe_process_audio_file(input_path=wav_path, output_path=reduced_path, use_noise_reduce=True)
            if reduced_path.exists():
                applied = nr is not None
                pipeline = "audio_filter_safe" if applied else "audio_filter_safe_without_noisereduce"
                return reduced_path, applied, pipeline
        except Exception as exc:
            print("audio_filter_safe failed. Falling back to legacy noisereduce.")
            print("Reason:", repr(exc))

    audio_array, sr = librosa.load(str(wav_path), sr=16000, mono=True)
    if nr is not None:
        reduced = nr.reduce_noise(y=audio_array, sr=sr, stationary=False, prop_decrease=0.35)
        pipeline, applied = "legacy_noisereduce", True
    else:
        reduced, pipeline, applied = audio_array, "raw_audio_fallback", False
    if sf is not None:
        sf.write(str(reduced_path), reduced, sr)
        if reduced_path.exists():
            return reduced_path, applied, pipeline
    return wav_path, applied, pipeline


def _cleanup_audio_files(paths: List[Optional[Path]]) -> None:
    """Delete temporary recording files (original, converted, reduced) unless
    retention is explicitly enabled. Recordings are private by default."""
    if RETAIN_AUDIO:
        return
    seen = set()
    for path in paths:
        try:
            if path is None:
                continue
            resolved = Path(path)
            if resolved in seen:
                continue
            seen.add(resolved)
            if resolved.exists():
                resolved.unlink()
        except Exception as exc:
            print("Could not delete temporary audio:", repr(exc))


def _audio_data_url(path: Optional[Path]) -> Optional[str]:
    """Encode processed audio for playback before deleting its temporary file."""
    if path is None:
        return None
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return None
    if path.stat().st_size > MAX_INLINE_PLAYBACK_BYTES:
        return None
    mime_type = {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".ogg": "audio/ogg",
        ".m4a": "audio/mp4",
        ".webm": "audio/webm",
        ".flac": "audio/flac",
    }.get(path.suffix.lower(), "application/octet-stream")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _candidate_cleanup_paths(audio_path: Path) -> List[Path]:
    """Every filename the post-save recording path can create."""
    audio_path = Path(audio_path)
    return [
        audio_path,
        audio_path.with_name(audio_path.stem + "_converted.wav"),
        audio_path.with_name(audio_path.stem + "_cleanvoice.wav"),
        audio_path.with_name(audio_path.stem + "_reduced.wav"),
    ]


def process_recording(audio_path: Path) -> Dict[str, Any]:
    """Convert -> inspect quality -> enhance -> transcribe.

    Every decodable recording continues through enhancement and transcription.
    The quality decision is advisory for UI warnings and mastery protection; it
    never blocks the user from seeing a score.
    ``cleanup_paths`` lists every on-disk artifact this produced so the caller
    can delete them after processing (private/temporary by default).
    """
    wav_path = convert_audio_to_wav(audio_path)
    raw_audio, sr = librosa.load(str(wav_path), sr=16000, mono=True)
    decision = analyze_audio_quality(raw_audio, sr)

    result: Dict[str, Any] = {
        "quality_decision": decision,
        "predicted_ipa": None,
        "reduced_audio_path": None,
        "noise_reduction_applied": False,
        "preprocessing_pipeline": "not_processed",
        "cleanvoice_applied": False,
        "cleanvoice_error": None,
        "cleanup_paths": [audio_path, wav_path],
    }
    cleanvoice_path = audio_path.with_name(audio_path.stem + "_cleanvoice.wav")
    result["cleanup_paths"].append(cleanvoice_path)
    model_input_path: Path
    applied = False
    pipeline = "raw_audio_fallback"

    if cleanvoice_configured():
        try:
            print("Enhancing with CleanVoice...")
            model_input_path = enhance_recording(wav_path, cleanvoice_path)
            applied = True
            pipeline = "cleanvoice_noise_reduction_normalization"
            result["cleanvoice_applied"] = True
        except CleanvoiceProcessingError as exc:
            print("Cleanvoice preprocessing failed:", str(exc))
            result["cleanvoice_error"] = str(exc)
            if cleanvoice_strict():
                raise
            reduced_path = audio_path.with_name(audio_path.stem + "_reduced.wav")
            model_input_path, applied, local_pipeline = _apply_noise_reduction(wav_path, reduced_path)
            pipeline = f"cleanvoice_failed_fallback:{local_pipeline}"
            result["cleanup_paths"].append(reduced_path)
    else:
        reduced_path = audio_path.with_name(audio_path.stem + "_reduced.wav")
        model_input_path, applied, pipeline = _apply_noise_reduction(wav_path, reduced_path)
        result["cleanup_paths"].append(reduced_path)

    result["noise_reduction_applied"] = applied
    result["preprocessing_pipeline"] = pipeline
    result["reduced_audio_path"] = model_input_path if model_input_path.exists() else None

    load_wav2vec_model()
    model_audio, _ = librosa.load(str(model_input_path), sr=16000, mono=True)
    inputs = processor(model_audio, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(inputs.input_values.to(device)).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    result["predicted_ipa"] = normalize_ipa(processor.batch_decode(predicted_ids)[0])
    return result


# -----------------------------
# Profile <-> mastery-model persistence
# -----------------------------
def _persist_phoneme_stats(user_id: int, stats: Dict[str, "mastery.PhonemeStat"]) -> None:
    for phoneme, stat in stats.items():
        db.upsert_phoneme_state(
            user_id=user_id,
            phoneme=phoneme,
            alpha=stat.alpha,
            beta=stat.beta,
            attempts_count=stat.independent_attempts,
            occurrence_count=stat.occurrence_count,
            last_practiced_at=format_db_datetime(stat.last_practiced_at or datetime.now(timezone.utc)),
        )


def _stat_to_row(stat: "mastery.PhonemeStat") -> Dict[str, Any]:
    return {
        "alpha": stat.alpha,
        "beta": stat.beta,
        "attempts_count": stat.independent_attempts,
        "occurrence_count": stat.occurrence_count,
        "last_practiced_at": format_db_datetime(stat.last_practiced_at or datetime.now(timezone.utc)),
    }


def _record_recording(
    user_id: int,
    user_text: str,
    reference_ipa: str,
    predicted_ipa: str,
    metrics: Dict[str, Any],
    alignment: List[Dict[str, Any]],
    quality_decision: AudioQualityDecision,
    exercise_id: Optional[int],
    engine_trusted: bool,
    engine_name: str,
    reference_g2p_trusted: bool,
    g2p_mode: str,
    reference_g2p_reason: Optional[str] = None,
) -> Dict[str, Any]:
    """Persist one processed recording as a SINGLE atomic transaction (attempt +
    events + mastery + assignment completion). Every decoded recording may be
    displayed and saved, but mastery is updated only when its quality, scoring
    engine, and reference pronunciation are trusted. Returns
    {attempt_id, mastery_updated}."""
    mastery_updated = should_update_mastery(
        scorable=quality_decision.scorable,
        scoring_trusted=engine_trusted and reference_g2p_trusted,
    )

    phoneme_states: Optional[Dict[str, Dict[str, Any]]] = None
    # The user completed the recording even if its quality is too weak for
    # long-term mastery evidence; do not force the same assignment again.
    complete_exercise_id: Optional[int] = exercise_id
    if mastery_updated:
        now = datetime.now(timezone.utc)
        stats = services.load_profile_stats(user_id)
        updated = mastery.update_mastery_for_recording(
            stats, alignment, now, quality_weight=quality_decision.quality_weight
        )
        # Persist only the phonemes this recording actually touched.
        touched = {
            canonicalize_phoneme(r["expected"])
            for r in alignment if r.get("expected") not in (None, "-")
        }
        phoneme_states = {ph: _stat_to_row(updated[ph]) for ph in touched if ph in updated}

    attempt_kwargs = {
        "text": user_text,
        "reference_ipa": reference_ipa,
        "predicted_ipa": predicted_ipa,
        "phoneme_error_rate": metrics["phoneme_error_rate"],
        "weighted_error": metrics["weighted_error"],
        "exercise_id": exercise_id,
        "raw_weighted_per": metrics["raw_weighted_per"],
        "quality_weight": quality_decision.quality_weight,
        "scorable": quality_decision.scorable,
        "scoring_engine": engine_name,
        "scoring_trusted": engine_trusted,
        "mastery_updated": mastery_updated,
        "insertion_count": metrics.get("insertion_count", metrics.get("insertions", 0)),
        "reference_unit_count": metrics.get("reference_unit_count", 0),
        "g2p_mode": g2p_mode,
        "reference_g2p_trusted": reference_g2p_trusted,
        "reference_g2p_reason": reference_g2p_reason,
    }
    attempt_id = db.record_recording_atomic(
        user_id=user_id,
        attempt_kwargs=attempt_kwargs,
        alignment=alignment,
        scoring_engine=engine_name,
        phoneme_states=phoneme_states,
        complete_exercise_id=complete_exercise_id,
    )
    return {"attempt_id": attempt_id, "mastery_updated": mastery_updated}


def validate_startup_inventory() -> Dict[str, Any]:
    """Startup validation: (1) every tagged bank phoneme maps into the canonical
    inventory, and (2) PanPhon can vectorize every assessable phoneme. Both are
    logged; neither is allowed to silently degrade into untrusted scoring."""
    bank_phonemes = db.get_all_bank_phonemes()
    report = validate_g2p_inventory(bank_phonemes)
    if not report["ok"]:
        print("WARNING: exercise-bank phonemes outside canonical inventory:", report["unsupported"])

    panphon_report = validate_panphon_inventory()
    heteronym_report = validate_heteronym_lexicon()
    if not panphon_available():
        print("NOTE: PanPhon not installed -- scoring runs in the untrusted "
              "fallback_features mode; mastery will NOT be updated.")
    elif not panphon_report["ok"]:
        print("WARNING: PanPhon is installed but cannot vectorize:",
              panphon_report["failures"], "-- scoring is UNTRUSTED (fallback_features).")
    else:
        print("PanPhon validated: all assessable phonemes vectorize -- scoring engine is TRUSTED.")
    if not heteronym_report["fully_supported"]:
        print(
            "Heteronym lexicon validated with explicitly unsupported contrasts:",
            heteronym_report["unsupported_contrasts"],
        )
    return {"inventory": report, "panphon": panphon_report, "heteronyms": heteronym_report}


# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    """Readiness, not just config presence. ``?deep=1`` additionally verifies a
    full processor+model load (heavy)."""
    deep = request.args.get("deep") == "1"
    model_config_present = (MODEL_PATH / "config.json").exists()
    weight_present = (MODEL_PATH / "model.safetensors").exists() or (MODEL_PATH / "pytorch_model.bin").exists()

    processor_loadable: Any = "unchecked"
    model_loadable: Any = "loaded" if model is not None else "unchecked"
    if deep:
        try:
            Wav2Vec2Processor.from_pretrained(str(MODEL_PATH))
            processor_loadable = True
        except Exception as exc:
            processor_loadable = f"error: {exc!r}"
        try:
            load_wav2vec_model()
            model_loadable = True
        except Exception as exc:
            model_loadable = f"error: {exc!r}"
    elif processor is not None:
        processor_loadable = True

    inventory_report = validate_g2p_inventory(db.get_all_bank_phonemes())
    bank_count = db.count_exercise_bank()
    panphon_report = validate_panphon_inventory()
    engine_trusted = scoring_trusted()
    heteronym_report = validate_heteronym_lexicon()

    ready = bool(
        model_config_present
        and weight_present
        and inventory_report["ok"]
        and heteronym_report["schema_and_inventory_ok"]
        and bank_count > 0
    )

    return jsonify_arpabet({
        "status": "running",
        "ready": ready,
        "alphabet": "arpabet",
        "internal_alphabet": "ipa",
        "device": device,
        "model_config_present": model_config_present,
        "model_weight_present": weight_present,
        "processor_loadable": processor_loadable,
        "model_loadable": model_loadable,
        "panphon_available": panphon_available(),
        "panphon_inventory_ok": panphon_report["ok"],
        "panphon_inventory_failures": panphon_report["failures"],
        "scoring_engine": scoring_engine(),
        "scoring_trusted": engine_trusted,
        "g2p_available": HETERONYMS_PATH.exists() and IPA_DICT_PATH.exists(),
        "g2p_mode": get_g2p_mode(),
        "heteronym_resolution_active": heteronym_resolution_active(),
        "heteronym_entries_checked": heteronym_report["checked"],
        "heteronym_unsupported_contrasts": heteronym_report["unsupported_contrasts"],
        "exercise_bank_count": bank_count,
        "exercise_bank_populated": bank_count > 0,
        "canonical_inventory_size": len(canonical_inventory()),
        "canonical_inventory_ok": inventory_report["ok"],
        "canonical_inventory_unsupported": inventory_report["unsupported"],
        "audio_filter_safe_available": safe_process_audio_file is not None,
        "audio_retention_enabled": RETAIN_AUDIO,
        "cleanvoice_enabled": cleanvoice_enabled(),
        "cleanvoice_configured": cleanvoice_configured(),
        "cleanvoice_sdk_available": cleanvoice_sdk_available(),
        "cleanvoice_strict": cleanvoice_strict(),
    })


@app.route("/uploads/<path:filename>")
def uploaded_audio(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route("/g2p", methods=["POST"])
def g2p_route():
    try:
        data = request.get_json(silent=True) or {}
        text = str(data.get("text", "")).strip()
        if not text:
            return jsonify({"error": "Please send text."}), 400
        resolution = g2p_convert_with_metadata(text)
        return jsonify_arpabet({
            "text": text,
            "ipa": resolution.ipa,
            "guide": ipa_reading_guide(resolution.ipa),
            **resolution.to_dict(),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/analyze", methods=["POST"])
def analyze():
    cleanup_paths: List[Optional[Path]] = []
    try:
        user_text = request.form.get("text", "").strip()
        profile_name = request.form.get("user", "").strip()
        sentence_id_raw = request.form.get("sentence_id", "").strip()
        sentence_id = int(sentence_id_raw) if sentence_id_raw.isdigit() else None

        if not user_text:
            return jsonify({"error": "Please enter text first."}), 400
        if "audio" not in request.files:
            return jsonify({"error": "No audio file received."}), 400

        audio_file = request.files["audio"]
        mimetype = (audio_file.mimetype or "").lower()
        extension = _extension_for_mimetype(mimetype)
        filename = f"{uuid.uuid4()}{extension}"
        audio_path = UPLOAD_FOLDER / filename
        # Register all deterministic artifacts before the first post-save
        # operation. This covers G2P, FFmpeg, preprocessing/model, scoring, and
        # database failures even when process_recording never returns.
        cleanup_paths.extend(_candidate_cleanup_paths(audio_path))
        audio_file.save(str(audio_path))

        reference = g2p_convert_with_metadata(user_text)
        reference_ipa = reference.ipa
        recording = process_recording(audio_path)
        cleanup_paths.extend(recording.get("cleanup_paths", []))
        decision: AudioQualityDecision = recording["quality_decision"]

        engine_name = scoring_engine()
        engine_trusted = scoring_trusted()
        reference_reason_parts = [
            *(f"unresolved:{word}" for word in reference.unresolved_heteronyms),
            *(f"unsupported:{word}" for word in reference.unsupported_heteronyms),
            *(f"oov:{word}" for word in reference.oov_words),
            *(f"predicted:{word}" for word in reference.predicted_words),
        ]
        reference_reason = ",".join(reference_reason_parts) or None
        user_row = db.get_or_create_user(profile_name) if profile_name else None

        predicted_ipa = recording["predicted_ipa"]
        reduced_audio_path = recording["reduced_audio_path"]
        processed_audio_data_url = (
            _audio_data_url(reduced_audio_path)
            if request.form.get("include_processed_audio") == "1"
            else None
        )
        # Reduced-audio playback is only offered when retention is enabled;
        # otherwise the file is deleted below and there is nothing to serve.
        reduced_audio_url = (
            f"/uploads/{reduced_audio_path.name}"
            if (reduced_audio_path and RETAIN_AUDIO) else None
        )

        ref_seq = tokenize_reference_ipa(reference_ipa)
        hyp_seq = tokenize_ctc_prediction(predicted_ipa)
        rows = align_phonemes(ref_seq, hyp_seq)
        metrics = calculate_metrics(rows)
        api_alignment = to_api_alignment(rows)

        profile = None
        mastery_updated = False
        if user_row is not None:
            valid_sentence_id = (
                sentence_id if sentence_id is not None and db.get_sentence_by_id(sentence_id) is not None else None
            )
            record = _record_recording(
                user_id=user_row["id"], user_text=user_text, reference_ipa=reference_ipa,
                predicted_ipa=predicted_ipa, metrics=metrics, alignment=rows,
                quality_decision=decision, exercise_id=valid_sentence_id,
                engine_trusted=engine_trusted, engine_name=engine_name,
                reference_g2p_trusted=reference.reference_g2p_trusted,
                g2p_mode=reference.g2p_mode,
                reference_g2p_reason=reference_reason,
            )
            mastery_updated = record["mastery_updated"]
            profile = {"id": user_row["id"], "name": user_row["name"]}

        if not engine_trusted:
            mastery_note = (
                "PanPhon unavailable or incomplete: showing provisional fallback scores only; "
                "mastery was NOT updated."
            )
        elif not reference.reference_g2p_trusted:
            mastery_note = (
                "The reference pronunciation is unresolved or unsupported; mastery was NOT updated."
            )
        elif not decision.scorable:
            mastery_note = (
                "Audio quality warnings were detected, but the recording was still processed. "
                "The score is shown without updating progress."
            )
        else:
            mastery_note = None

        return jsonify_arpabet({
            "scorable": True,
            "quality_warning": not decision.scorable,
            "text": user_text,
            "reference_ipa": reference_ipa,
            "predicted_ipa": predicted_ipa,
            "reference_guide": ipa_reading_guide(reference_ipa),
            "predicted_guide": ipa_reading_guide(predicted_ipa),
            **reference.to_dict(),
            "scoring_engine": engine_name,
            "scoring_trusted": engine_trusted,
            "mastery_updated": mastery_updated,
            "mastery_note": mastery_note,
            "noise_reduction_applied": recording["noise_reduction_applied"],
            "preprocessing_pipeline": recording["preprocessing_pipeline"],
            "cleanvoice_applied": recording.get("cleanvoice_applied", False),
            "cleanvoice_error": recording.get("cleanvoice_error"),
            "audio_quality": decision.to_dict(),
            "reduced_audio_url": reduced_audio_url,
            "processed_audio_data_url": processed_audio_data_url,
            "alignment": api_alignment,
            "metrics": metrics,
            "profile": profile,
        })

    except AudioDecodeError as e:
        return jsonify({"error": str(e), "code": "audio_decode_unavailable"}), 400
    except CleanvoiceProcessingError as e:
        return jsonify({"error": str(e), "code": "cleanvoice_unavailable"}), 502
    except Exception as e:
        import traceback
        print("\n========== REAL ERROR ==========")
        traceback.print_exc()
        print("================================\n")
        return jsonify({"error": str(e)}), 500
    finally:
        # One outer cleanup covers every operation after the upload is named,
        # including partial FFmpeg/preprocessing output and database failures.
        _cleanup_audio_files(cleanup_paths)


@app.route("/users", methods=["GET", "POST"])
def users_route():
    if request.method == "POST":
        data = request.get_json(silent=True) or request.form
        name = str(data.get("name", "")).strip()
        if not name:
            return jsonify({"error": "Please provide a name."}), 400
        user_row = db.get_or_create_user(name)
        return jsonify({"id": user_row["id"], "name": user_row["name"]})
    return jsonify([{"id": row["id"], "name": row["name"]} for row in db.list_users()])


@app.route("/practice/next")
def practice_next():
    """Adaptive next exercise, now integrating the evidence-aware assessment
    and a continuing cold-start diagnostic phase."""
    try:
        profile_name = request.args.get("user", "").strip()
        if not profile_name:
            return jsonify({"error": "Please provide a user name."}), 400

        user_row = db.get_or_create_user(profile_name)
        user_id = user_row["id"]
        now = datetime.now(timezone.utc)

        stats = services.load_profile_stats(user_id)
        context_stats = db.get_phoneme_context_stats(user_id)
        assessment = services.assess_profile(user_id, now=now)
        diag = assessment_mod.diagnostic_status(context_stats)
        recently_served = db.get_recently_served_sentence_ids(user_id)
        confusion_pairs = db.get_confusion_pairs(user_id)

        overmastered = sorted(mastery.get_overmastered_phonemes(stats, now=now))
        under_observed = diag["uncovered"]

        confusion_hint = None
        confusion_phoneme = None
        exercise_type = "diagnostic"

        if diag["in_diagnostic"] or not stats:
            mode = "diagnostic"
            targets: List[str] = []
            selection = services.choose_exercise(
                targets=[], overmastered=overmastered, under_observed=under_observed,
                overall_level=assessment.get("exercise_level", assessment["overall_level"]),
                recently_served_ids=recently_served, g2p_convert=g2p_convert_with_metadata,
                ipa_to_tokens=tokenize_reference_ipa, exercise_type="diagnostic", diagnostic=True,
            )
        else:
            targets = mastery.rank_weak_phonemes(stats, now=now)
            top_target = targets[0] if targets else None
            confusion = assessment_mod.main_confusion_for(top_target, confusion_pairs) if top_target else None
            if confusion is not None:
                confusion_phoneme = confusion["spoken"]
                confusion_hint = f"{confusion['expected']} vs {confusion['spoken']}"
            if top_target is not None:
                exercise_type = assessment_mod.exercise_type_for_mastery(
                    mastery.posterior_mean(stats[top_target], now=now) if top_target in stats else None
                )
            selection = services.choose_exercise(
                targets=targets, overmastered=overmastered, under_observed=under_observed,
                overall_level=assessment.get("exercise_level", assessment["overall_level"]),
                recently_served_ids=recently_served, g2p_convert=g2p_convert_with_metadata,
                ipa_to_tokens=tokenize_reference_ipa, exercise_type=exercise_type,
                top_target=top_target, confusion_phoneme=confusion_phoneme, diagnostic=False,
            )
            mode = selection["source_mode"]

        chosen = selection["exercise"]
        if selection["generated"] is not None:
            gen = selection["generated"]
            new_id = db.insert_sentence(
                text=gen["text"], reference_ipa=gen["reference_ipa"], word_count=gen["word_count"],
                level_proxy=gen["level_proxy"], phoneme_counts=gen["phoneme_counts"],
                source=gen.get("source", "llm_generated"),
            )
            if new_id is not None:
                gen["id"] = new_id
                chosen = gen
                mode = selection["source_mode"]
            else:
                existing = db.get_sentence_by_text(gen["text"])
                if existing is not None:
                    gen["id"] = existing["id"]
                    chosen = gen
                    mode = selection["source_mode"]

        if chosen is None:
            return jsonify({
                "error": "No practice sentences are available yet. Run scripts/build_exercise_bank.py first."
            }), 503

        db.record_practice_assignment(user_id, chosen["id"], targets)

        return jsonify_arpabet({
            "sentence_id": chosen["id"],
            "text": chosen["text"],
            "reference_ipa": chosen["reference_ipa"],
            "reference_guide": ipa_reading_guide(chosen["reference_ipa"]),
            "target_phonemes": targets,
            "mode": mode,
            "exercise_type": exercise_type,
            "confusion_hint": confusion_hint,
            "assessment": assessment,
            "diagnostic": {
                "in_diagnostic": diag["in_diagnostic"],
                "covered_count": diag["covered_count"],
                "coverage_target": diag["coverage_target"],
            },
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/exercise", methods=["GET", "POST"])
def exercise_route():
    """Focused service endpoint. Profile path uses the evidence-aware
    assessment; the stateless metrics path reports a provisional level."""
    try:
        user_name = None
        recently_served: set = set()
        metrics = None

        if request.method == "POST":
            data = request.get_json(silent=True) or {}
            if "metrics" in data:
                try:
                    metrics = metrics_to_internal_ipa(data.get("metrics"))
                except (TypeError, ValueError) as exc:
                    return jsonify({"error": f"Invalid ARPAbet metrics: {exc}"}), 400
            else:
                user_name = str(data.get("user", "")).strip()
        else:
            user_name = request.args.get("user", "").strip()
            if not user_name:
                return jsonify({"error": "Provide ?user=NAME, or POST a metrics object."}), 400

        if user_name:
            user_row = db.get_or_create_user(user_name)
            recently_served = db.get_recently_served_sentence_ids(user_row["id"])
            result = services.generate_exercise_for_profile(
                user_row["id"],
                g2p_convert_with_metadata,
                tokenize_reference_ipa,
                recently_served,
            )
        else:
            result = services.generate_exercise(
                metrics or {}, g2p_convert_with_metadata, tokenize_reference_ipa, recently_served
            )

        exercise = result.get("exercise")
        if exercise is None:
            return jsonify_arpabet(result), 503

        if user_name:
            db.record_practice_assignment(user_row["id"], exercise["sentence_id"], result["target_phonemes"])

        exercise["reference_guide"] = ipa_reading_guide(exercise["reference_ipa"])
        return jsonify_arpabet(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/practice/assessment")
def practice_assessment():
    """Evidence-aware level assessment for a profile (for the UI)."""
    profile_name = request.args.get("user", "").strip()
    if not profile_name:
        return jsonify({"error": "Please provide a user name."}), 400
    user_row = db.get_user_by_name(profile_name)
    if user_row is None:
        return jsonify({"assessment": None})
    return jsonify_arpabet({"assessment": services.assess_profile(user_row["id"])})


@app.route("/practice/gaps")
def practice_gaps():
    profile_name = request.args.get("user", "").strip()
    if not profile_name:
        return jsonify({"error": "Please provide a user name."}), 400

    user_row = db.get_user_by_name(profile_name)
    if user_row is None:
        return jsonify({"phonemes": []})

    now = datetime.now(timezone.utc)
    stats = services.load_profile_stats(user_row["id"])
    ranked = sorted(stats.items(), key=lambda pair: mastery.lower_confidence_bound(pair[1], now=now))

    phonemes = []
    for phoneme, stat in ranked:
        guide = PHONEME_GUIDE.get(phoneme, {})
        phonemes.append({
            "phoneme": phoneme,
            "mastery": round(mastery.posterior_mean(stat, now=now), 3),
            "lower_confidence_bound": round(mastery.lower_confidence_bound(stat, now=now), 3),
            "independent_attempts": stat.independent_attempts,
            "occurrence_count": stat.occurrence_count,
            "last_practiced_at": stat.last_practiced_at.isoformat() if stat.last_practiced_at else None,
            "example": guide.get("example", ""),
        })
    return jsonify_arpabet({"phonemes": phonemes})


@app.route("/practice/history")
def practice_history():
    profile_name = request.args.get("user", "").strip()
    if not profile_name:
        return jsonify({"error": "Please provide a user name."}), 400
    user_row = db.get_user_by_name(profile_name)
    if user_row is None:
        return jsonify({"attempts": []})
    rows = db.get_user_attempts(user_row["id"], limit=20)
    return jsonify({
        "attempts": [
            {
                "id": row["id"],
                "text": row["text"],
                "phoneme_error_rate": row["phoneme_error_rate"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]
    })


if __name__ == "__main__":
    load_g2p_engine()
    validate_startup_inventory()
    app.run(host="127.0.0.1", port=5000, debug=True)
