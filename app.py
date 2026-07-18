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
import content
import db
import mastery
import services
from app_datetime import format_db_datetime, parse_db_datetime
from audio_quality import AudioQualityDecision, analyze_audio_quality, should_update_mastery
from g2p_service import (
    G2P_DIR,
    HETERONYMS_PATH,
    IPA_DICT_PATH,
    g2p_convert,
    get_g2p_mode,
    load_g2p_engine,
)
from phoneme_vectors_professional import (
    canonical_inventory,
    panphon_available,
    scoring_engine,
    validate_g2p_inventory,
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

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
MODEL_PATH = BASE_DIR / "model" / "my_wav2vec2_phoneme_model"
VOICE_FILTERING_DIR = BASE_DIR / "voice-filtering"

# When True, PROVISIONAL articulatory scores from the fallback (non-PanPhon)
# engine are never folded into trusted mastery -- they're clearly a degraded
# state, not "professional" substitution scores.
MASTERY_REQUIRES_PANPHON = True

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
# Audio conversion + quality gate + transcription
# -----------------------------
def convert_audio_to_wav(input_path: Path) -> Path:
    """Convert browser audio to 16kHz mono WAV when FFmpeg is available."""
    input_path = Path(input_path)
    output_path = input_path.with_name(input_path.stem + "_converted.wav")
    if output_path.exists():
        return output_path
    if shutil.which("ffmpeg") is None:
        return input_path
    command = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-ac", "1", "-ar", "16000", str(output_path),
    ]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_path


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


def process_recording(audio_path: Path) -> Dict[str, Any]:
    """Convert -> quality-gate -> (if scorable) noise-reduce -> transcribe.

    If the audio-quality gate rejects the recording, transcription is skipped
    entirely and ``predicted_ipa`` is None -- the caller must not score it.
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
    }
    if not decision.scorable:
        return result  # gate: do not transcribe or score an unscorable recording

    reduced_path = audio_path.with_name(audio_path.stem + "_reduced.wav")
    model_input_path, applied, pipeline = _apply_noise_reduction(wav_path, reduced_path)
    result["noise_reduction_applied"] = applied
    result["preprocessing_pipeline"] = pipeline
    result["reduced_audio_path"] = reduced_path if reduced_path.exists() else None

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


def _record_recording(
    user_id: int,
    user_text: str,
    reference_ipa: str,
    predicted_ipa: str,
    metrics: Dict[str, Any],
    alignment: List[Dict[str, Any]],
    quality_decision: AudioQualityDecision,
    exercise_id: Optional[int],
    scoring_trusted: bool,
) -> Dict[str, Any]:
    """Persist one scorable recording, and update mastery ONLY when scoring is
    trusted (PanPhon-backed). Returns {attempt_id, mastery_updated}."""
    attempt_id = db.record_attempt(
        user_id=user_id,
        text=user_text,
        reference_ipa=reference_ipa,
        predicted_ipa=predicted_ipa,
        phoneme_error_rate=metrics["phoneme_error_rate"],
        weighted_error=metrics["weighted_error"],
        exercise_id=exercise_id,
        raw_weighted_per=metrics["raw_weighted_per"],
        quality_weight=quality_decision.quality_weight,
        scorable=True,
    )
    db.record_phoneme_events(attempt_id, alignment)

    mastery_updated = False
    if should_update_mastery(scorable=True, scoring_trusted=scoring_trusted):
        now = datetime.now(timezone.utc)
        stats = services.load_profile_stats(user_id)
        updated = mastery.update_mastery_for_recording(stats, alignment, now)
        _persist_phoneme_stats(user_id, updated)
        mastery_updated = True
        if exercise_id is not None:
            db.complete_latest_assignment(user_id, exercise_id, attempt_id)

    return {"attempt_id": attempt_id, "mastery_updated": mastery_updated}


def validate_startup_inventory() -> Dict[str, Any]:
    """Check that the phonemes tagged into the exercise bank all map into the
    canonical scoring inventory; log any that don't. Unsupported phonemes must
    never silently enter scoring."""
    bank_phonemes = db.get_all_bank_phonemes()
    report = validate_g2p_inventory(bank_phonemes)
    if not report["ok"]:
        print("WARNING: exercise-bank phonemes outside canonical inventory:", report["unsupported"])
    return report


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
    panphon_ok = panphon_available()

    ready = bool(model_config_present and weight_present and inventory_report["ok"] and bank_count > 0)

    return jsonify({
        "status": "running",
        "ready": ready,
        "device": device,
        "model_config_present": model_config_present,
        "model_weight_present": weight_present,
        "processor_loadable": processor_loadable,
        "model_loadable": model_loadable,
        "panphon_available": panphon_ok,
        "scoring_engine": scoring_engine(),
        "scoring_trusted": panphon_ok,
        "g2p_available": HETERONYMS_PATH.exists() and IPA_DICT_PATH.exists(),
        "g2p_mode": get_g2p_mode(),
        "exercise_bank_count": bank_count,
        "exercise_bank_populated": bank_count > 0,
        "canonical_inventory_size": len(canonical_inventory()),
        "canonical_inventory_ok": inventory_report["ok"],
        "canonical_inventory_unsupported": inventory_report["unsupported"],
        "audio_filter_safe_available": safe_process_audio_file is not None,
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
        ipa = g2p_convert(text)
        return jsonify({"text": text, "ipa": ipa, "guide": ipa_reading_guide(ipa), "g2p_mode": get_g2p_mode()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/analyze", methods=["POST"])
def analyze():
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
        extension = ".ogg" if "ogg" in mimetype else ".webm"
        filename = f"{uuid.uuid4()}{extension}"
        audio_path = UPLOAD_FOLDER / filename
        audio_file.save(str(audio_path))

        reference_ipa = g2p_convert(user_text)
        recording = process_recording(audio_path)
        decision: AudioQualityDecision = recording["quality_decision"]

        user_row = db.get_or_create_user(profile_name) if profile_name else None

        # ---- Audio-quality gate: reject unscorable recordings ----
        if not decision.scorable:
            if user_row is not None:
                # Optionally store the rejected attempt (no events, no mastery).
                db.record_attempt(
                    user_id=user_row["id"], text=user_text, reference_ipa=reference_ipa,
                    predicted_ipa="", phoneme_error_rate=0.0, weighted_error=0.0,
                    quality_weight=decision.quality_weight, scorable=False,
                    rejected_reason=",".join(decision.reasons),
                )
            return jsonify({
                "scorable": False,
                "audio_quality": decision.to_dict(),
                "message": "That recording could not be scored. Please record again "
                           "(check your mic, avoid clipping, and speak the full sentence).",
                "reference_ipa": reference_ipa,
                "reference_guide": ipa_reading_guide(reference_ipa),
                "profile": ({"id": user_row["id"], "name": user_row["name"]} if user_row else None),
            })

        predicted_ipa = recording["predicted_ipa"]
        reduced_audio_path = recording["reduced_audio_path"]
        reduced_audio_url = f"/uploads/{reduced_audio_path.name}" if reduced_audio_path else None

        ref_seq = tokenize_reference_ipa(reference_ipa)
        hyp_seq = tokenize_ctc_prediction(predicted_ipa)
        rows = align_phonemes(ref_seq, hyp_seq)
        metrics = calculate_metrics(rows)
        api_alignment = to_api_alignment(rows)

        scoring_trusted = panphon_available() or not MASTERY_REQUIRES_PANPHON

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
                scoring_trusted=scoring_trusted,
            )
            mastery_updated = record["mastery_updated"]
            profile = {"id": user_row["id"], "name": user_row["name"]}

        return jsonify({
            "scorable": True,
            "text": user_text,
            "reference_ipa": reference_ipa,
            "predicted_ipa": predicted_ipa,
            "reference_guide": ipa_reading_guide(reference_ipa),
            "predicted_guide": ipa_reading_guide(predicted_ipa),
            "g2p_mode": get_g2p_mode(),
            "scoring_engine": scoring_engine(),
            "scoring_trusted": scoring_trusted,
            "mastery_updated": mastery_updated,
            "mastery_note": (
                None if scoring_trusted else
                "PanPhon unavailable: showing provisional fallback scores only; mastery was NOT updated."
            ),
            "noise_reduction_applied": recording["noise_reduction_applied"],
            "preprocessing_pipeline": recording["preprocessing_pipeline"],
            "audio_quality": decision.to_dict(),
            "reduced_audio_url": reduced_audio_url,
            "alignment": api_alignment,
            "metrics": metrics,
            "profile": profile,
        })

    except Exception as e:
        import traceback
        print("\n========== REAL ERROR ==========")
        traceback.print_exc()
        print("================================\n")
        return jsonify({"error": str(e)}), 500


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
        exercise_type = "diagnostic"

        if diag["in_diagnostic"] or not stats:
            mode = "diagnostic"
            targets: List[str] = []
            selection = services.choose_exercise(
                targets=[], overmastered=overmastered, under_observed=under_observed,
                overall_level=assessment["overall_level"], confusion_hint=None,
                recently_served_ids=recently_served, g2p_convert=g2p_convert,
                ipa_to_tokens=tokenize_reference_ipa, diagnostic=True,
            )
        else:
            targets = mastery.rank_weak_phonemes(stats, now=now)
            top_target = targets[0] if targets else None
            confusion_hint = services._confusion_hint_for(top_target, confusion_pairs)
            if top_target is not None:
                exercise_type = assessment_mod.exercise_type_for_mastery(
                    mastery.posterior_mean(stats[top_target], now=now)
                    if top_target in stats else None
                )
            selection = services.choose_exercise(
                targets=targets, overmastered=overmastered, under_observed=under_observed,
                overall_level=assessment["overall_level"], confusion_hint=confusion_hint,
                recently_served_ids=recently_served, g2p_convert=g2p_convert,
                ipa_to_tokens=tokenize_reference_ipa, diagnostic=False,
            )
            mode = selection["source_mode"]

        chosen = selection["exercise"]
        if selection["generated"] is not None:
            gen = selection["generated"]
            new_id = db.insert_sentence(
                text=gen["text"], reference_ipa=gen["reference_ipa"], word_count=gen["word_count"],
                level_proxy=gen["level_proxy"], phoneme_counts=gen["phoneme_counts"], source="llm_generated",
            )
            if new_id is not None:
                gen["id"] = new_id
                chosen = gen
                mode = "generated"

        if chosen is None:
            return jsonify({
                "error": "No practice sentences are available yet. Run scripts/build_exercise_bank.py first."
            }), 503

        db.record_practice_assignment(user_id, chosen["id"], targets)

        return jsonify({
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

        if request.method == "POST":
            data = request.get_json(silent=True) or {}
            if "metrics" in data:
                metrics = {str(k): float(v) for k, v in (data.get("metrics") or {}).items()}
            else:
                user_name = str(data.get("user", "")).strip()
                metrics = services.metrics_for_user(user_name) if user_name else {}
        else:
            user_name = request.args.get("user", "").strip()
            if not user_name:
                return jsonify({"error": "Provide ?user=NAME, or POST a metrics object."}), 400
            metrics = services.metrics_for_user(user_name)

        if user_name:
            user_row = db.get_user_by_name(user_name)
            if user_row is not None:
                recently_served = db.get_recently_served_sentence_ids(user_row["id"])

        result = services.generate_exercise(metrics, g2p_convert, tokenize_reference_ipa, recently_served)

        exercise = result.get("exercise")
        if exercise is None:
            return jsonify(result), 503

        if user_name:
            user_row = db.get_or_create_user(user_name)
            db.record_practice_assignment(user_row["id"], exercise["sentence_id"], result["target_phonemes"])

        exercise["reference_guide"] = ipa_reading_guide(exercise["reference_ipa"])
        return jsonify(result)
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
    return jsonify({"assessment": services.assess_profile(user_row["id"])})


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
    return jsonify({"phonemes": phonemes})


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
