import os
import re
import sys
import uuid
import unicodedata
import subprocess
import shutil
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

from phoneme_vectors import canonicalize_phoneme, phoneme_distance, panphon_available
import librosa
import torch
from flask import Flask, jsonify, render_template, request, send_from_directory
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
MODEL_PATH = BASE_DIR / "model" / "my_wav2vec2_phoneme_model"
G2P_DIR = BASE_DIR / "g2p_pipeline_split_v2"
HETERONYMS_PATH = G2P_DIR / "heteronyms.json"
IPA_DICT_PATH = G2P_DIR / "cmudict-0.7b-ipa.txt"
VOICE_FILTERING_DIR = BASE_DIR / "voice-filtering"

UPLOAD_FOLDER.mkdir(exist_ok=True)

if str(VOICE_FILTERING_DIR) not in sys.path:
    sys.path.insert(0, str(VOICE_FILTERING_DIR))

try:
    from audio_filter_safe import process_audio_file as safe_process_audio_file
except Exception as exc:
    safe_process_audio_file = None
    print("audio_filter_safe is unavailable. Falling back to legacy preprocessing.")
    print("Reason:", repr(exc))

try:
    from audio_quality_check import analyze_audio as run_audio_quality_check
except Exception as exc:
    run_audio_quality_check = None
    print("audio_quality_check is unavailable. Skipping dropout checks.")
    print("Reason:", repr(exc))

processor = None
model = None
g2p_engine = None
g2p_mode = "not_loaded"
device = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------
# Wav2Vec2 model loading
# -----------------------------
def load_wav2vec_model():
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
# G2P loading
# -----------------------------
def load_ipa_dictionary() -> Dict[str, str]:
    """Simple local IPA dictionary fallback using cmudict-0.7b-ipa.txt."""
    if not IPA_DICT_PATH.exists():
        raise FileNotFoundError(f"IPA dictionary not found: {IPA_DICT_PATH}")

    dictionary: Dict[str, str] = {}
    with IPA_DICT_PATH.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(";;;"):
                continue

            # The file is usually WORD<TAB>IPA. Some lines may have spaces.
            if "\t" in line:
                word, ipa = line.split("\t", 1)
            else:
                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    continue
                word, ipa = parts

            word = re.sub(r"\(\d+\)$", "", word).lower()
            ipa = ipa.split(",")[0].strip()  # choose first pronunciation
            dictionary.setdefault(word, ipa)

    return dictionary


class DictionaryIpaG2p:
    """Fallback G2P if NeMo or spaCy is not installed."""

    def __init__(self):
        self.dictionary = load_ipa_dictionary()

    def __call__(self, text: str) -> List[str]:
        words = re.findall(r"[A-Za-z']+", text.lower())
        out = []
        for word in words:
            lookup = word.strip("'")
            out.append(self.dictionary.get(lookup, lookup))
        return out


def load_g2p_engine():
    """
    Prefer your ContextAwareIpaG2p engine.
    If NeMo/spaCy is unavailable, fall back to the bundled IPA dictionary.
    """
    global g2p_engine, g2p_mode

    if g2p_engine is not None:
        return

    if str(G2P_DIR) not in sys.path:
        sys.path.insert(0, str(G2P_DIR))

    try:
        from contextual_g2p import ContextAwareIpaG2p

        g2p_engine = ContextAwareIpaG2p(
            heteronyms_json_path=str(HETERONYMS_PATH),
            ipa_dict_path=str(IPA_DICT_PATH),
        )
        g2p_mode = "context_aware_ipa_g2p"
    except Exception as exc:
        print("ContextAwareIpaG2p could not be loaded. Using dictionary fallback.")
        print("Reason:", repr(exc))
        g2p_engine = DictionaryIpaG2p()
        g2p_mode = "dictionary_ipa_fallback"


# -----------------------------
# IPA normalization and tokenization
# -----------------------------
MULTI_CHAR_PHONEMES = [
    "tʃ", "dʒ",
    "aɪ", "aʊ", "eɪ", "oʊ", "ɔɪ",
    "iː", "uː", "ɑː", "ɔː", "ɜː", "ɝ", "ɚ",
    "ər", "ɪr", "ɛr", "ʊr", "ɔr", "ɑr",
]

COMBINING_MARKS = {"ː", "̃", "̩", "̯", "ʰ"}


def normalize_ipa(text: str) -> str:
    text = unicodedata.normalize("NFC", str(text))
    text = text.replace("ˈ", "")
    text = text.replace("ˌ", "")
    text = text.replace(":", "ː")
    text = text.replace("/", " ")
    text = text.replace("[", " ").replace("]", " ")
    text = text.replace("|", " | ")
    text = " ".join(text.split())
    return text


def split_ipa_word(ipa_word: str) -> List[str]:
    """
    Convert one IPA word string into phoneme-like tokens.
    Example: skuːl -> s k uː l
    """
    ipa_word = normalize_ipa(ipa_word).replace(" ", "")
    if not ipa_word:
        return []

    tokens: List[str] = []
    i = 0
    while i < len(ipa_word):
        matched = None
        for ph in sorted(MULTI_CHAR_PHONEMES, key=len, reverse=True):
            if ipa_word.startswith(ph, i):
                matched = ph
                break

        if matched is not None:
            tokens.append(matched)
            i += len(matched)
            continue

        char = ipa_word[i]
        if char in COMBINING_MARKS and tokens:
            tokens[-1] += char
        elif char not in {" ", "_"}:
            tokens.append(char)
        i += 1

    return tokens


def words_to_spaced_ipa(ipa_words: List[str]) -> str:
    """Join G2P word outputs as phoneme-spaced words separated by |."""
    spaced_words = []
    for word_ipa in ipa_words:
        tokens = split_ipa_word(word_ipa)
        if tokens:
            spaced_words.append(" ".join(tokens))
    return " | ".join(spaced_words)


def g2p_convert(text: str) -> str:
    """
    Text -> IPA phonemes using the bundled G2P.
    Returns phonemes separated by spaces and words separated by |.
    """
    load_g2p_engine()
    ipa_words = g2p_engine(text)
    return normalize_ipa(words_to_spaced_ipa(ipa_words))


def ipa_to_tokens(ipa: str) -> List[str]:
    ipa = normalize_ipa(ipa)
    if " " in ipa:
        tokens = [tok for tok in ipa.split() if tok != "|"]
    else:
        # Fallback for model output without spaces.
        tokens = [tok for tok in split_ipa_word(ipa) if tok != "|"]

    return [canon for canon in (canonicalize_phoneme(tok) for tok in tokens) if canon]



# -----------------------------
# IPA reading guide
# -----------------------------
PHONEME_GUIDE = {
    "i": {"example": "see", "description": "long ee sound, like 'see'"},
    "iː": {"example": "see", "description": "long ee sound, like 'see'"},
    "ɪ": {"example": "sit", "description": "short i sound, like 'sit'"},
    "eɪ": {"example": "say", "description": "like 'ay' in 'say'"},
    "ɛ": {"example": "bed", "description": "short e sound, like 'bed'"},
    "æ": {"example": "cat", "description": "a sound like 'cat'"},
    "ɑ": {"example": "father", "description": "open ah sound, like 'father'"},
    "ɔ": {"example": "thought", "description": "aw sound, like 'thought'"},
    "ʊ": {"example": "foot", "description": "short oo sound, like 'foot'"},
    "u": {"example": "food", "description": "oo sound, like 'food'"},
    "uː": {"example": "food", "description": "long oo sound, like 'food'"},
    "ʌ": {"example": "cup", "description": "uh sound, like 'cup'"},
    "ə": {"example": "about", "description": "schwa: weak 'uh' sound"},
    "ɝ": {"example": "bird", "description": "r-colored vowel, like 'bird'"},
    "aɪ": {"example": "my", "description": "like 'eye', as in 'my'"},
    "aʊ": {"example": "now", "description": "like 'ow', as in 'now'"},
    "oʊ": {"example": "go", "description": "like 'oh', as in 'go'"},
    "ɔɪ": {"example": "boy", "description": "like 'oy', as in 'boy'"},
    "p": {"example": "pen", "description": "voiceless p sound"},
    "b": {"example": "boy", "description": "voiced b sound"},
    "t": {"example": "top", "description": "voiceless t sound"},
    "d": {"example": "dog", "description": "voiced d sound"},
    "k": {"example": "cat", "description": "voiceless k sound"},
    "g": {"example": "go", "description": "voiced g sound"},
    "ɡ": {"example": "go", "description": "voiced g sound"},
    "f": {"example": "fish", "description": "voiceless f sound"},
    "v": {"example": "van", "description": "voiced v sound"},
    "θ": {"example": "think", "description": "voiceless th sound, like 'think'"},
    "ð": {"example": "this", "description": "voiced th sound, like 'this'"},
    "s": {"example": "see", "description": "s sound"},
    "z": {"example": "zoo", "description": "z sound"},
    "ʃ": {"example": "she", "description": "sh sound"},
    "ʒ": {"example": "measure", "description": "zh sound, like 'measure'"},
    "h": {"example": "hat", "description": "h sound"},
    "tʃ": {"example": "chair", "description": "ch sound"},
    "dʒ": {"example": "jump", "description": "j sound"},
    "m": {"example": "man", "description": "m sound"},
    "n": {"example": "no", "description": "n sound"},
    "ŋ": {"example": "sing", "description": "ng sound, like the end of 'sing'"},
    "l": {"example": "love", "description": "l sound"},
    "ɹ": {"example": "red", "description": "English r sound"},
    "r": {"example": "red", "description": "English r sound"},
    "w": {"example": "we", "description": "w sound"},
    "j": {"example": "yes", "description": "y sound, like 'yes'"},
}


def ipa_reading_guide(ipa: str):
    """Return a word-by-word reading guide for an IPA sequence."""
    ipa = normalize_ipa(ipa)
    words = [w.strip() for w in ipa.split("|") if w.strip()]
    guide = []

    for word_index, word in enumerate(words, start=1):
        tokens = word.split() if " " in word else split_ipa_word(word)
        phonemes = []
        for token in tokens:
            token = canonicalize_phoneme(token)
            info = PHONEME_GUIDE.get(token, {
                "example": "unknown",
                "description": "No guide available for this phoneme yet.",
            })
            phonemes.append({
                "symbol": token,
                "example": info["example"],
                "description": info["description"],
            })
        guide.append({"word_index": word_index, "phonemes": phonemes})

    return guide


def convert_audio_to_wav(input_path: Path) -> Path:
    """Convert browser audio to 16kHz mono WAV when FFmpeg is available."""
    input_path = Path(input_path)
    output_path = input_path.with_name(input_path.stem + "_converted.wav")

    if output_path.exists():
        return output_path

    if shutil.which("ffmpeg") is None:
        return input_path

    command = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-ac", "1",
        "-ar", "16000",
        str(output_path),
    ]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_path


def summarize_quality_report(report: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if report is None:
        return None

    if "error" in report:
        return {
            "error": str(report["error"]),
        }

    near_silent = report.get("near_silent_regions_over_30ms", []) or []
    exact_zero = report.get("exact_zero_regions_over_10ms", []) or []

    return {
        "duration_seconds": report.get("duration_seconds"),
        "overall_rms": report.get("overall_rms"),
        "peak_amplitude": report.get("peak_amplitude"),
        "near_silent_region_count": len(near_silent),
        "exact_zero_region_count": len(exact_zero),
        "possible_dropout": bool(near_silent or exact_zero),
        "near_silent_regions_preview": near_silent[:5],
        "exact_zero_regions_preview": exact_zero[:5],
    }


# -----------------------------
# Audio to Phonemes using Wav2Vec2
# -----------------------------
# def transcribe_audio_to_phonemes(audio_path: Path) -> str:
#     load_wav2vec_model()

#     audio_array, _ = librosa.load(str(audio_path), sr=16000, mono=True)

#     inputs = processor(
#         audio_array,
#         sampling_rate=16000,
#         return_tensors="pt",
#         padding=True,
#     )

#     input_values = inputs.input_values.to(device)

#     with torch.no_grad():
#         logits = model(input_values).logits

#     predicted_ids = torch.argmax(logits, dim=-1)
#     predicted_ipa = processor.batch_decode(predicted_ids)[0]

#     return normalize_ipa(predicted_ipa)

def transcribe_audio_to_phonemes(audio_path: Path):
    load_wav2vec_model()

    wav_or_original_path = convert_audio_to_wav(audio_path)
    reduced_path = audio_path.with_name(audio_path.stem + "_reduced.wav")

    quality_report = None
    if run_audio_quality_check is not None:
        try:
            quality_report = run_audio_quality_check(str(wav_or_original_path), save_plots=False)
        except Exception as exc:
            quality_report = {"error": str(exc)}

    model_input_path: Optional[Path] = wav_or_original_path
    noise_reduction_applied = False
    preprocessing_pipeline = "raw_audio_fallback"

    if safe_process_audio_file is not None:
        try:
            safe_process_audio_file(
                input_path=wav_or_original_path,
                output_path=reduced_path,
                use_noise_reduce=True,
            )
            if reduced_path.exists():
                model_input_path = reduced_path
                noise_reduction_applied = nr is not None
                preprocessing_pipeline = "audio_filter_safe" if nr is not None else "audio_filter_safe_without_noisereduce"
        except Exception as exc:
            print("audio_filter_safe failed. Falling back to legacy noisereduce.")
            print("Reason:", repr(exc))

    if model_input_path == wav_or_original_path:
        audio_array, sr = librosa.load(
            str(wav_or_original_path),
            sr=16000,
            mono=True
        )

        if nr is not None:
            model_audio = nr.reduce_noise(
                y=audio_array,
                sr=sr
            )
            noise_reduction_applied = True
            preprocessing_pipeline = "legacy_noisereduce"
        else:
            model_audio = audio_array
            preprocessing_pipeline = "raw_audio_fallback"

        if sf is not None:
            sf.write(str(reduced_path), model_audio, sr)
            if reduced_path.exists():
                model_input_path = reduced_path
        else:
            model_input_path = None

    if model_input_path is not None:
        model_audio, _ = librosa.load(
            str(model_input_path),
            sr=16000,
            mono=True
        )

    inputs = processor(
        model_audio,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
    )

    input_values = inputs.input_values.to(device)

    with torch.no_grad():
        logits = model(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    predicted_ipa = processor.batch_decode(predicted_ids)[0]

    return (
        normalize_ipa(predicted_ipa),
        (reduced_path if reduced_path.exists() else None),
        noise_reduction_applied,
        preprocessing_pipeline,
        summarize_quality_report(quality_report),
    )

# -----------------------------
# Dynamic Programming Alignment
# -----------------------------
# def align_phonemes(ref_seq: List[str], hyp_seq: List[str]):
#     n = len(ref_seq)
#     m = len(hyp_seq)

#     dp = [[0] * (m + 1) for _ in range(n + 1)]
#     backtrack = [[None] * (m + 1) for _ in range(n + 1)]

#     gap_penalty = -1
#     match_score = 2
#     mismatch_penalty = -1

#     for i in range(1, n + 1):
#         dp[i][0] = i * gap_penalty
#         backtrack[i][0] = "UP"

#     for j in range(1, m + 1):
#         dp[0][j] = j * gap_penalty
#         backtrack[0][j] = "LEFT"

#     for i in range(1, n + 1):
#         for j in range(1, m + 1):
#             diag = dp[i - 1][j - 1] + (match_score if ref_seq[i - 1] == hyp_seq[j - 1] else mismatch_penalty)
#             up = dp[i - 1][j] + gap_penalty
#             left = dp[i][j - 1] + gap_penalty

#             best = max(diag, up, left)
#             dp[i][j] = best

#             if best == diag:
#                 backtrack[i][j] = "DIAG"
#             elif best == up:
#                 backtrack[i][j] = "UP"
#             else:
#                 backtrack[i][j] = "LEFT"

#     aligned_ref = []
#     aligned_hyp = []
#     operations = []

#     i, j = n, m
#     while i > 0 or j > 0:
#         move = backtrack[i][j]

#         if move == "DIAG":
#             ref_ph = ref_seq[i - 1]
#             hyp_ph = hyp_seq[j - 1]
#             aligned_ref.append(ref_ph)
#             aligned_hyp.append(hyp_ph)
#             operations.append("correct" if ref_ph == hyp_ph else "substitution")
#             i -= 1
#             j -= 1
#         elif move == "UP":
#             aligned_ref.append(ref_seq[i - 1])
#             aligned_hyp.append("-")
#             operations.append("deletion")
#             i -= 1
#         elif move == "LEFT":
#             aligned_ref.append("-")
#             aligned_hyp.append(hyp_seq[j - 1])
#             operations.append("insertion")
#             j -= 1
#         else:
#             break

#     aligned_ref.reverse()
#     aligned_hyp.reverse()
#     operations.reverse()
#     return aligned_ref, aligned_hyp, operations

# -----------------------------
# Alignment cost model
# -----------------------------
MATCH_COST = 0.0

DELETION_COST = 0.85
INSERTION_COST = 0.85

VERY_CLOSE_SUB_COST = 0.25
CLOSE_SUB_COST = 0.45
MEDIUM_SUB_COST = 0.75
MAJOR_SUB_COST = 1.05

VOWEL_CONSONANT_SUB_COST = 1.35
UNKNOWN_SUB_COST = 1.20

VOWEL_PHONEMES = {
    "i", "iː", "ɪ", "e", "ɛ", "æ", "ɑ", "ɔ", "ʊ", "u", "uː", "ʌ", "ə", "ɝ", "ɚ",
    "aɪ", "aʊ", "eɪ", "oʊ", "ɔɪ",
}

CONSONANT_PHONEMES = {
    "p", "b", "t", "d", "k", "g", "ɡ", "f", "v", "θ", "ð", "s", "z", "ʃ", "ʒ", "h",
    "tʃ", "dʒ", "m", "n", "ŋ", "l", "ɹ", "r", "w", "j",
}

KNOWN_PHONEMES = {canonicalize_phoneme(ph) for ph in (VOWEL_PHONEMES | CONSONANT_PHONEMES)}


def _is_vowel(phoneme: str) -> bool:
    return canonicalize_phoneme(phoneme) in VOWEL_PHONEMES


def _is_known_phoneme(phoneme: str) -> bool:
    return canonicalize_phoneme(phoneme) in KNOWN_PHONEMES


def substitution_cost_and_label(ref_ph: str, hyp_ph: str):
    ref_ph = canonicalize_phoneme(ref_ph)
    hyp_ph = canonicalize_phoneme(hyp_ph)

    if ref_ph == hyp_ph:
        return MATCH_COST, "correct"

    if not _is_known_phoneme(ref_ph) or not _is_known_phoneme(hyp_ph):
        return UNKNOWN_SUB_COST, "unknown_substitution"

    if _is_vowel(ref_ph) != _is_vowel(hyp_ph):
        return VOWEL_CONSONANT_SUB_COST, "vowel_consonant_substitution"

    distance_value = phoneme_distance(ref_ph, hyp_ph)

    if distance_value <= 0.15:
        return VERY_CLOSE_SUB_COST, "very_close_substitution"
    if distance_value <= 0.35:
        return CLOSE_SUB_COST, "close_substitution"
    if distance_value <= 0.65:
        return MEDIUM_SUB_COST, "medium_substitution"
    return MAJOR_SUB_COST, "major_substitution"


def align_phonemes(ref_seq, hyp_seq):
    n = len(ref_seq)
    m = len(hyp_seq)

    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    backtrack = [[None] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = i * DELETION_COST
        backtrack[i][0] = "UP"

    for j in range(1, m + 1):
        dp[0][j] = j * INSERTION_COST
        backtrack[0][j] = "LEFT"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            ref_ph = ref_seq[i - 1]
            hyp_ph = hyp_seq[j - 1]

            sub_cost, _ = substitution_cost_and_label(ref_ph, hyp_ph)

            diag = dp[i - 1][j - 1] + sub_cost
            up = dp[i - 1][j] + DELETION_COST
            left = dp[i][j - 1] + INSERTION_COST

            best = min(diag, up, left)
            dp[i][j] = best

            # Tie-breaking:
            # Prefer exact match, then deletion/insertion, then substitution.
            if sub_cost == 0.0 and diag == best:
                backtrack[i][j] = "DIAG"
            elif up == best:
                backtrack[i][j] = "UP"
            elif left == best:
                backtrack[i][j] = "LEFT"
            else:
                backtrack[i][j] = "DIAG"

    aligned_ref = []
    aligned_hyp = []
    operations = []
    distances = []

    i, j = n, m

    while i > 0 or j > 0:
        move = backtrack[i][j]

        if move == "DIAG":
            ref_ph = ref_seq[i - 1]
            hyp_ph = hyp_seq[j - 1]
            dist, label = substitution_cost_and_label(ref_ph, hyp_ph)

            aligned_ref.append(ref_ph)
            aligned_hyp.append(hyp_ph)
            distances.append(round(dist, 3))

            operations.append(label)

            i -= 1
            j -= 1

        elif move == "UP":
            aligned_ref.append(ref_seq[i - 1])
            aligned_hyp.append("-")
            operations.append("deletion")
            distances.append(DELETION_COST)
            i -= 1

        elif move == "LEFT":
            aligned_ref.append("-")
            aligned_hyp.append(hyp_seq[j - 1])
            operations.append("insertion")
            distances.append(INSERTION_COST)
            j -= 1

    aligned_ref.reverse()
    aligned_hyp.reverse()
    operations.reverse()
    distances.reverse()

    return aligned_ref, aligned_hyp, operations, distances

# def calculate_metrics(operations: List[str]):
#     total_reference_units = len([op for op in operations if op != "insertion"])

#     substitutions = operations.count("substitution")
#     deletions = operations.count("deletion")
#     insertions = operations.count("insertion")
#     correct = operations.count("correct")

#     if total_reference_units > 0:
#         per =min(100, ((substitutions + deletions + insertions) / total_reference_units) * 100)
#     else:
#         per = 0

#     return {
#         "correct": correct,
#         "substitutions": substitutions,
#         "deletions": deletions,
#         "insertions": insertions,
#         "phoneme_error_rate": round(per, 2),
#     }


def calculate_metrics(operations: List[str], distances: List[float] | None = None):
    total_reference_units = len([op for op in operations if op != "insertion"])

    very_close = operations.count("very_close_substitution")
    close = operations.count("close_substitution")
    medium = operations.count("medium_substitution")
    major = operations.count("major_substitution")
    vowel_consonant = operations.count("vowel_consonant_substitution")
    unknown = operations.count("unknown_substitution")

    substitutions = sum(1 for op in operations if op.endswith("_substitution"))
    deletions = operations.count("deletion")
    insertions = operations.count("insertion")
    correct = operations.count("correct")

    minor = very_close + close
    major_total = major + vowel_consonant + unknown

    if distances and len(distances) == len(operations):
        # Weighted PER from the configured alignment costs.
        weighted_error = sum(
            dist for op, dist in zip(operations, distances)
            if op != "correct"
        )
    else:
        # Fallback if distances are unavailable.
        weighted_error = (
            very_close * VERY_CLOSE_SUB_COST +
            close * CLOSE_SUB_COST +
            medium * MEDIUM_SUB_COST +
            major * MAJOR_SUB_COST +
            vowel_consonant * VOWEL_CONSONANT_SUB_COST +
            unknown * UNKNOWN_SUB_COST +
            deletions * DELETION_COST +
            insertions * INSERTION_COST
        )

    if total_reference_units > 0:
        per = min(100, (weighted_error / total_reference_units) * 100)
    else:
        per = 0

    return {
        "correct": correct,
        "substitutions": substitutions,
        "minor_substitutions": minor,
        "medium_substitutions": medium,
        "major_substitutions": major_total,
        "very_close_substitutions": very_close,
        "close_substitutions": close,
        "vowel_consonant_substitutions": vowel_consonant,
        "unknown_substitutions": unknown,
        "deletions": deletions,
        "insertions": insertions,
        "weighted_error": round(weighted_error, 3),
        "phoneme_error_rate": round(per, 2),
    }

# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    model_ready = (MODEL_PATH / "config.json").exists()
    g2p_ready = HETERONYMS_PATH.exists() and IPA_DICT_PATH.exists()
    return jsonify({
        "status": "running",
        "device": device,
        "model_ready": model_ready,
        "model_path": str(MODEL_PATH),
        "g2p_ready": g2p_ready,
        "g2p_mode": g2p_mode,
        "g2p_path": str(G2P_DIR),
        "panphon_available": panphon_available(),
        "audio_filter_safe_available": safe_process_audio_file is not None,
        "audio_quality_check_available": run_audio_quality_check is not None,
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
        return jsonify({"text": text, "ipa": ipa, "guide": ipa_reading_guide(ipa), "g2p_mode": g2p_mode})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        user_text = request.form.get("text", "").strip()

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
        (
            predicted_ipa,
            reduced_audio_path,
            noise_reduction_applied,
            preprocessing_pipeline,
            quality_report,
        ) = transcribe_audio_to_phonemes(audio_path)
        reduced_audio_url = f"/uploads/{reduced_audio_path.name}" if reduced_audio_path else None

        ref_seq = ipa_to_tokens(reference_ipa)
        hyp_seq = ipa_to_tokens(predicted_ipa)

        aligned_ref, aligned_hyp, operations, distances = align_phonemes(ref_seq, hyp_seq)
        metrics = calculate_metrics(operations, distances)

        alignment = []
        for ref, hyp, op, dist in zip(aligned_ref, aligned_hyp, operations, distances):
            alignment.append({
            "expected": ref,
            "spoken": hyp,
            "result": op,
            "distance": dist,
            })

        return jsonify({
            "text": user_text,
            "reference_ipa": reference_ipa,
            "predicted_ipa": predicted_ipa,
            "reference_guide": ipa_reading_guide(reference_ipa),
            "predicted_guide": ipa_reading_guide(predicted_ipa),
            "g2p_mode": g2p_mode,
            "vectorizer": "panphon" if panphon_available() else "fallback_features",
            "noise_reduction_applied": noise_reduction_applied,
            "preprocessing_pipeline": preprocessing_pipeline,
            "audio_quality_check": quality_report,
            "reduced_audio_url": reduced_audio_url,
            "alignment": alignment,
            "metrics": metrics,
        })

    except Exception as e:
        import traceback
        print("\n========== REAL ERROR ==========")
        traceback.print_exc()
        print("================================\n")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    load_g2p_engine()
    app.run(host="127.0.0.1", port=5000, debug=True)
