import os
import re
import sys
import uuid
import unicodedata
from pathlib import Path
from typing import Dict, List
import noisereduce as nr
import soundfile as sf

from phoneme_vectors import canonicalize_phoneme, phoneme_distance
import librosa
import torch
from flask import Flask, jsonify, render_template, request
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
MODEL_PATH = BASE_DIR / "model" / "my_wav2vec2_phoneme_model"
G2P_DIR = BASE_DIR / "g2p_pipeline_split_v2"
HETERONYMS_PATH = G2P_DIR / "heteronyms.json"
IPA_DICT_PATH = G2P_DIR / "cmudict-0.7b-ipa.txt"

UPLOAD_FOLDER.mkdir(exist_ok=True)

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

def transcribe_audio_to_phonemes(audio_path: Path) -> str:
    load_wav2vec_model()

    audio_array, sr = librosa.load(
        str(audio_path),
        sr=16000,
        mono=True
    )

    reduced_noise = nr.reduce_noise(
        y=audio_array,
        sr=sr
    )

    clean_path = audio_path.with_name(audio_path.stem + "_reduced.wav")
    sf.write(str(clean_path), reduced_noise, sr)

    inputs = processor(
        reduced_noise,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
    )

    input_values = inputs.input_values.to(device)

    with torch.no_grad():
        logits = model(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    predicted_ipa = processor.batch_decode(predicted_ids)[0]

    return normalize_ipa(predicted_ipa)

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


def align_phonemes(ref_seq, hyp_seq):
    n = len(ref_seq)
    m = len(hyp_seq)

    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    backtrack = [[None] * (m + 1) for _ in range(n + 1)]

    deletion_cost = 1.0
    insertion_cost = 1.0

    for i in range(1, n + 1):
        dp[i][0] = i * deletion_cost
        backtrack[i][0] = "UP"

    for j in range(1, m + 1):
        dp[0][j] = j * insertion_cost
        backtrack[0][j] = "LEFT"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            ref_ph = ref_seq[i - 1]
            hyp_ph = hyp_seq[j - 1]

            sub_cost = phoneme_distance(ref_ph, hyp_ph)

            diag = dp[i - 1][j - 1] + sub_cost
            up = dp[i - 1][j] + deletion_cost
            left = dp[i][j - 1] + insertion_cost

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
            dist = phoneme_distance(ref_ph, hyp_ph)

            aligned_ref.append(ref_ph)
            aligned_hyp.append(hyp_ph)
            distances.append(round(dist, 3))

            if dist == 0:
                operations.append("correct")
            elif dist <= 0.25:
                operations.append("minor_substitution")
            elif dist <= 0.50:
                operations.append("medium_substitution")
            else:
                operations.append("major_substitution")

            i -= 1
            j -= 1

        elif move == "UP":
            aligned_ref.append(ref_seq[i - 1])
            aligned_hyp.append("-")
            operations.append("deletion")
            distances.append(1.0)
            i -= 1

        elif move == "LEFT":
            aligned_ref.append("-")
            aligned_hyp.append(hyp_seq[j - 1])
            operations.append("insertion")
            distances.append(1.0)
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


def calculate_metrics(operations: List[str]):
    total_reference_units = len([op for op in operations if op != "insertion"])

    minor = operations.count("minor_substitution")
    medium = operations.count("medium_substitution")
    major = operations.count("major_substitution")

    substitutions = minor + medium + major
    deletions = operations.count("deletion")
    insertions = operations.count("insertion")
    correct = operations.count("correct")

    # Weighted phoneme error rate
    weighted_error = (
        minor * 0.33 +
        medium * 0.66 +
        major * 1.0 +
        deletions * 1.0 +
        insertions * 1.0
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
        "major_substitutions": major,
        "deletions": deletions,
        "insertions": insertions,
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
    })


@app.route("/g2p", methods=["POST"])
def g2p_route():
    try:
        data = request.get_json(silent=True) or {}
        text = str(data.get("text", "")).strip()
        if not text:
            return jsonify({"error": "Please send text."}), 400
        ipa = g2p_convert(text)
        return jsonify({"text": text, "ipa": ipa, "g2p_mode": g2p_mode})
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
        filename = f"{uuid.uuid4()}.webm"
        audio_path = UPLOAD_FOLDER / filename
        audio_file.save(str(audio_path))

        reference_ipa = g2p_convert(user_text)
        predicted_ipa = transcribe_audio_to_phonemes(audio_path)

        ref_seq = ipa_to_tokens(reference_ipa)
        hyp_seq = ipa_to_tokens(predicted_ipa)

        aligned_ref, aligned_hyp, operations, distances = align_phonemes(ref_seq, hyp_seq)
        metrics = calculate_metrics(operations)

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
            "g2p_mode": g2p_mode,
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
