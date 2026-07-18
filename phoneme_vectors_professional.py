"""
Single source of truth for phoneme-level policy in the pronunciation app.

This module owns EVERY authoritative phoneme decision the rest of the app
relies on, so there is exactly one implementation of each:

    * canonicalize_phoneme()            - IPA symbol normalization + aliasing
    * the canonical scoring inventory   - derived from the model vocab.json
    * phoneme_distance()                - articulatory distance in [0, 1]
    * is_vowel() / is_consonant()       - major-class classification
    * alignment_substitution_cost()     - cost used ONLY by DP alignment
    * substitution_cost_and_label()     - (alignment_cost, result_label)
    * classify_substitution()           - result label from ref/hyp/distance
    * mastery_observation()             - soft evidence in [0, 1] for mastery
    * validate_g2p_inventory()          - startup inventory sanity check
    * scoring_engine() / panphon_available()

Scientific note
---------------
The articulatory distance below comes from PanPhon feature vectors
(Mortensen et al., 2016, COLING). PanPhon describes how *similar* two
phonemes are; it is NOT a calibrated acoustic pronunciation score. The
mastery evidence derived from it is therefore labelled PROVISIONAL
throughout the app, and is never presented as CEFR-equivalent or as a
research-grade Goodness-of-Pronunciation (GOP) probability.

When PanPhon is not installed, this module falls back to a small, clearly
labelled articulatory-class feature model so the app keeps running, but it
reports ``scoring_engine() == "fallback_features"`` so callers can refuse to
update trusted mastery from those numbers (see app.py's mastery gate).
"""

from __future__ import annotations

import json
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

BASE_DIR = Path(__file__).resolve().parent
MODEL_VOCAB_PATH = BASE_DIR / "model" / "my_wav2vec2_phoneme_model" / "vocab.json"

# Feature order used by current PanPhon FeatureTable examples.
PANPHON_FEATURE_NAMES: Tuple[str, ...] = (
    "syl", "son", "cons", "cont", "delrel", "lat", "nas", "strid",
    "voi", "sg", "cg", "ant", "cor", "distr", "lab", "hi", "lo",
    "back", "round", "velaric", "tense", "long",
)

# -----------------------------------------------------------------------------
# Aliases: symbol normalization only, NOT feature definitions.
#
# The model vocab distinguishes /ɪ/ from /i/ and /ʊ/ from /u/ but does not
# contain the length symbol /ː/. So expected long vowels from the G2P
# (iː, uː, ɑː, ɔː) must collapse onto units the model can actually emit,
# otherwise they would be phonemes the acoustic model can never produce.
# -----------------------------------------------------------------------------
PHONEME_ALIASES: Dict[str, str] = {
    "g": "ɡ",       # ASCII g -> IPA script g (model vocab uses ɡ)
    "ɫ": "l",       # dark l -> l
    "r": "ɹ",       # ASCII r / trill -> English approximant (model vocab uses ɹ)
    "ɚ": "ɝ",       # r-coloured schwa -> stressed r-coloured vowel
    "ɜː": "ɝ",
    "ɜ:": "ɝ",
    "ə˞": "ɝ",
    "˞": "",
    "ɑː": "ɑ",      # long vowels -> model-emittable short symbol (no /ː/ in vocab)
    "ɑ:": "ɑ",
    "ɔː": "ɔ",
    "ɔ:": "ɔ",
    "iː": "i",
    "i:": "i",
    "uː": "u",
    "u:": "u",
    # Tie-bar affricates -> app tokens the DP alignment/mastery layers use.
    "t͡ʃ": "tʃ",
    "d͡ʒ": "dʒ",
}

# Application-level composite phonemes: a single scoring unit whose acoustic
# realisation is a sequence of model tokens. Kept composite so mastery and
# alignment treat e.g. /tʃ/ or /oʊ/ as one target, while PanPhon can still
# vectorize them as the mean of their component segments.
COMPOSITE_COMPONENTS: Mapping[str, Tuple[str, ...]] = {
    "tʃ": ("t", "ʃ"),
    "dʒ": ("d", "ʒ"),
    "aɪ": ("a", "ɪ"),
    "aʊ": ("a", "ʊ"),
    "eɪ": ("e", "ɪ"),
    "oʊ": ("o", "ʊ"),
    "ɔɪ": ("ɔ", "ɪ"),
}

# Non-phoneme model tokens that never take part in scoring.
_NON_PHONEME_TOKENS = {"|", "ˌ", "ˈ", "[PAD]", "[UNK]", "<s>", "</s>", "", " "}


@lru_cache(maxsize=1)
def _model_base_phonemes() -> Tuple[str, ...]:
    """Base phoneme inventory the acoustic model can emit, read from its
    vocab.json. Falls back to the known set if the file is missing so the
    module still imports in a stripped-down checkout."""
    try:
        with MODEL_VOCAB_PATH.open("r", encoding="utf-8") as f:
            vocab = json.load(f)
        tokens = [t for t in vocab.keys() if t not in _NON_PHONEME_TOKENS]
    except Exception:
        tokens = [
            "a", "b", "d", "e", "f", "h", "i", "j", "k", "l", "m", "n", "o",
            "p", "s", "t", "u", "v", "w", "z", "æ", "ð", "ŋ", "ɑ", "ɔ", "ə",
            "ɛ", "ɝ", "ɡ", "ɪ", "ɹ", "ʃ", "ʊ", "ʌ", "ʒ", "θ",
        ]
    return tuple(sorted(set(tokens)))


@lru_cache(maxsize=1)
def canonical_inventory() -> frozenset:
    """The canonical scoring inventory = every model-emittable base phoneme
    plus the application-level composite phonemes (affricates, diphthongs)."""
    base = set(_model_base_phonemes())
    return frozenset(base | set(COMPOSITE_COMPONENTS.keys()))


# Vowel / consonant membership over the canonical inventory. Used as the
# authoritative major-class split (and as the PanPhon-free fallback for
# is_vowel). These lists are canonical-inventory symbols only.
VOWEL_PHONEMES: frozenset = frozenset({
    "i", "ɪ", "e", "ɛ", "æ", "ɑ", "ɔ", "ʊ", "u", "ʌ", "ə", "ɝ", "o", "a",
    "aɪ", "aʊ", "eɪ", "oʊ", "ɔɪ",
})

CONSONANT_PHONEMES: frozenset = frozenset({
    "p", "b", "t", "d", "k", "ɡ", "f", "v", "θ", "ð", "s", "z", "ʃ", "ʒ", "h",
    "tʃ", "dʒ", "m", "n", "ŋ", "l", "ɹ", "w", "j",
})

# Backwards-compatible name used by older imports / tests.
KNOWN_IPA_PHONEMES: Tuple[str, ...] = tuple(sorted(VOWEL_PHONEMES | CONSONANT_PHONEMES))

# The set of phonemes that count toward level assessment and inventory
# coverage. It is the standard American-English phoneme inventory: all
# consonants, the monophthong vowels, and the diphthongs -- but NOT the bare
# vowels /a/, /e/, /o/, which exist here only as diphthong components and are
# never independent English targets. Coverage denominators use this set.
_DIPHTHONG_ONLY_COMPONENTS = frozenset({"a", "e", "o"})
ASSESSABLE_INVENTORY: frozenset = frozenset(
    (VOWEL_PHONEMES | CONSONANT_PHONEMES) - _DIPHTHONG_ONLY_COMPONENTS
)


def is_assessable(phoneme: str) -> bool:
    """True if the canonicalized phoneme is a standard English target that
    counts toward level assessment / inventory coverage."""
    return canonicalize_phoneme(phoneme) in ASSESSABLE_INVENTORY


def canonicalize_phoneme(phoneme: str) -> str:
    """Normalize an IPA token before any inventory lookup or distance call.

    THE single canonicalizer for the whole app. Strips stress marks and
    delimiters, converts ``:`` to ``ː``, then applies symbol aliases
    (idempotently) so every downstream module sees the same representation.
    """
    ph = unicodedata.normalize("NFC", str(phoneme)).strip()
    if not ph:
        return ph

    ph = ph.replace(":", "ː")
    ph = ph.replace("ˈ", "").replace("ˌ", "")
    ph = ph.replace("/", "").replace("[", "").replace("]", "")

    for _ in range(8):
        mapped = PHONEME_ALIASES.get(ph)
        if mapped is None or mapped == ph:
            break
        ph = mapped

    # Strip any residual length mark AFTER aliasing. Specific long vowels
    # (iː, ɜː, ...) were already collapsed by aliases; the model vocab has no
    # /ː/ at all, so a leftover length mark on any other symbol (e.g. ɝː) must
    # not survive into scoring. Only strip when something remains.
    if "ː" in ph and len(ph) > 1:
        stripped = ph.replace("ː", "")
        if stripped:
            ph = stripped
    return ph


def in_inventory(phoneme: str) -> bool:
    """True if the canonicalized phoneme is part of the scoring inventory."""
    return canonicalize_phoneme(phoneme) in canonical_inventory()


# Alias kept for readability at call sites in app.py.
is_known_phoneme = in_inventory


# -----------------------------------------------------------------------------
# PanPhon loading (real articulatory vectors when available)
# -----------------------------------------------------------------------------
def _patch_panphon_utf8_read() -> None:
    """PanPhon on Windows may open its CSVs with cp1252, which breaks on IPA
    bytes. Force UTF-8 in FeatureTable._read_bases."""
    try:
        import pandas as pd
        import panphon.featuretable as ft_mod
        from importlib.resources import files
    except Exception:
        return

    if getattr(ft_mod.FeatureTable, "_utf8_read_patch_applied", False):
        return

    def _read_bases_utf8(self, fn: str, weights):
        spec_to_int = {"+": 1, "0": 0, "-": -1}
        with files("panphon").joinpath(fn).open(encoding="utf-8") as f:
            df = pd.read_csv(f)
        df["ipa"] = df["ipa"].apply(self.normalize)
        feature_names = list(df.columns[1:])
        df[feature_names] = df[feature_names].map(lambda x: spec_to_int[x])
        segments = [
            (row["ipa"], ft_mod.Segment(feature_names, row[1:].to_dict(), weights=weights))
            for (_, row) in df.iterrows()
        ]
        return segments, dict(segments), feature_names

    ft_mod.FeatureTable._read_bases = _read_bases_utf8
    ft_mod.FeatureTable._utf8_read_patch_applied = True


@lru_cache(maxsize=1)
def _feature_table():
    try:
        import panphon
        _patch_panphon_utf8_read()
        return panphon.FeatureTable()
    except Exception:
        return None


def panphon_available() -> bool:
    """True only when the real PanPhon library can be imported and loaded."""
    return _feature_table() is not None


def scoring_engine() -> str:
    """Which distance engine is actually in use.

    ``"panphon"``           - real PanPhon articulatory vectors (trusted).
    ``"fallback_features"`` - the small built-in class model (degraded).

    app.py refuses to fold ``fallback_features`` results into trusted mastery
    so PanPhon being unavailable never silently produces "professional" scores.
    """
    return "panphon" if panphon_available() else "fallback_features"


# Feature weights used only to normalize the PanPhon vector distance.
FEATURE_WEIGHTS: Mapping[str, float] = {
    "syl": 2.0, "son": 1.5, "cons": 1.5, "cont": 1.0, "delrel": 0.75,
    "lat": 0.75, "nas": 1.0, "strid": 0.75, "voi": 0.75, "sg": 0.5,
    "cg": 0.5, "ant": 0.75, "cor": 0.75, "distr": 0.5, "lab": 0.75,
    "hi": 1.0, "lo": 1.0, "back": 1.0, "round": 0.75, "velaric": 0.25,
    "tense": 0.75, "long": 0.5,
}


def _as_numeric_vector(row: Sequence[object]) -> Tuple[float, ...]:
    out: List[float] = []
    for value in row:
        if isinstance(value, (int, float)):
            out.append(float(value))
        elif value == "+":
            out.append(1.0)
        elif value == "-":
            out.append(-1.0)
        else:
            out.append(0.0)
    return tuple(out)


def _mean_vectors(vectors: Sequence[Sequence[float]]) -> Tuple[float, ...]:
    if not vectors:
        raise ValueError("Cannot average an empty vector list.")
    width = len(vectors[0])
    return tuple(sum(float(v[i]) for v in vectors) / len(vectors) for i in range(width))


@lru_cache(maxsize=512)
def _segment_vector(segment: str) -> Tuple[float, ...]:
    ft = _feature_table()
    if ft is None:
        raise RuntimeError("PanPhon is unavailable.")
    rows = ft.word_to_vector_list(segment, numeric=True)
    if not rows:
        raise ValueError(f"PanPhon could not vectorize IPA segment: {segment!r}")
    return _mean_vectors([_as_numeric_vector(row) for row in rows])


@lru_cache(maxsize=512)
def phoneme_vector(phoneme: str) -> Tuple[float, ...]:
    """One fixed-width PanPhon articulatory vector per app phoneme (mean of
    component segment vectors for composites). Raises if PanPhon is absent."""
    ph = canonicalize_phoneme(phoneme)
    if not ph:
        raise ValueError("Empty phoneme cannot be vectorized.")
    components = COMPOSITE_COMPONENTS.get(ph)
    if components:
        return _mean_vectors([_segment_vector(c) for c in components])
    return _segment_vector(ph)


def phoneme_features(phoneme: str) -> Dict[str, float]:
    vector = phoneme_vector(phoneme)
    names = PANPHON_FEATURE_NAMES[: len(vector)]
    return dict(zip(names, vector))


def _weighted_l1_distance(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    width = min(len(vec_a), len(vec_b), len(PANPHON_FEATURE_NAMES))
    if width == 0:
        return 1.0
    weighted_diff = 0.0
    max_weighted_diff = 0.0
    for i in range(width):
        weight = FEATURE_WEIGHTS.get(PANPHON_FEATURE_NAMES[i], 1.0)
        weighted_diff += weight * abs(float(vec_a[i]) - float(vec_b[i]))
        max_weighted_diff += weight * 2.0  # PanPhon values in {-1, 0, +1}
    if max_weighted_diff <= 0:
        return 1.0
    return max(0.0, min(1.0, weighted_diff / max_weighted_diff))


# -----------------------------------------------------------------------------
# Fallback articulatory-class distance (used only when PanPhon is unavailable)
# -----------------------------------------------------------------------------
_FALLBACK_CLASSES: Dict[str, frozenset] = {
    "vowel": VOWEL_PHONEMES,
    "stop": frozenset({"p", "b", "t", "d", "k", "ɡ"}),
    "fricative": frozenset({"f", "v", "θ", "ð", "s", "z", "ʃ", "ʒ", "h"}),
    "affricate": frozenset({"tʃ", "dʒ"}),
    "nasal": frozenset({"m", "n", "ŋ"}),
    "liquid": frozenset({"l", "ɹ"}),
    "glide": frozenset({"w", "j"}),
    "lateral": frozenset({"l"}),
    "rhotic": frozenset({"ɹ", "ɝ"}),
    "voiced": frozenset({"b", "d", "ɡ", "v", "ð", "z", "ʒ", "dʒ", "m", "n",
                         "ŋ", "l", "ɹ", "w", "j"}),
    "bilabial": frozenset({"p", "b", "m"}),
    "labiodental": frozenset({"f", "v"}),
    "dental": frozenset({"θ", "ð"}),
    "alveolar": frozenset({"t", "d", "s", "z", "n", "l", "ɹ"}),
    "postalveolar": frozenset({"ʃ", "ʒ", "tʃ", "dʒ"}),
    "velar": frozenset({"k", "ɡ", "ŋ", "w"}),
    "front_vowel": frozenset({"i", "ɪ", "e", "ɛ", "æ", "eɪ"}),
    "central_vowel": frozenset({"ʌ", "ə", "ɝ", "a"}),
    "back_vowel": frozenset({"ɑ", "ɔ", "ʊ", "u", "o", "oʊ", "aʊ"}),
    "high_vowel": frozenset({"i", "ɪ", "u", "ʊ"}),
    "low_vowel": frozenset({"æ", "ɑ", "a"}),
}


def _fallback_vector(phoneme: str) -> List[int]:
    ph = canonicalize_phoneme(phoneme)
    return [1 if ph in members else 0 for members in _FALLBACK_CLASSES.values()]


def _fallback_distance(a: str, b: str) -> float:
    a = canonicalize_phoneme(a)
    b = canonicalize_phoneme(b)
    if a == b:
        return 0.0
    va, vb = _fallback_vector(a), _fallback_vector(b)
    if not any(va) or not any(vb):
        return 1.0
    diff = sum(1 for x, y in zip(va, vb) if x != y)
    return min(1.0, diff / max(len(va), 1))


def is_vowel(phoneme: str) -> bool:
    """Major-class check. Uses PanPhon's syllabic feature when available,
    otherwise the canonical vowel set."""
    ft = _feature_table()
    if ft is not None:
        try:
            return phoneme_features(phoneme).get("syl", 0.0) > 0.25
        except Exception:
            pass
    return canonicalize_phoneme(phoneme) in VOWEL_PHONEMES


def is_consonant(phoneme: str) -> bool:
    ph = canonicalize_phoneme(phoneme)
    if ph in CONSONANT_PHONEMES:
        return True
    if ph in VOWEL_PHONEMES:
        return False
    return not is_vowel(ph)


def articulatory_distance(a: str, b: str) -> float:
    """Normalized articulatory feature distance in [0, 1].

    0.0 = identical after canonicalization. 1.0 = maximally different or
    not vectorizable. Uses PanPhon vectors when available, otherwise the
    labelled fallback class model.

    This is a DESCRIPTIVE similarity, not a calibrated pronunciation score.
    """
    a = canonicalize_phoneme(a)
    b = canonicalize_phoneme(b)
    if a == b:
        return 0.0
    if not panphon_available():
        return _fallback_distance(a, b)
    try:
        return _weighted_l1_distance(phoneme_vector(a), phoneme_vector(b))
    except Exception:
        return _fallback_distance(a, b)


# Public name kept stable for existing callers.
phoneme_distance = articulatory_distance


# -----------------------------------------------------------------------------
# Alignment cost + substitution classification (single authoritative policy)
# -----------------------------------------------------------------------------
# Costs used ONLY by the dynamic-programming alignment. They are NOT distances
# and NOT scores: they are the price the alignment pays to line two phonemes
# up, tuned so a vowel/consonant swap is worse than opening a gap.
MATCH_COST = 0.0
DELETION_COST = 0.85
INSERTION_COST = 0.85

VERY_CLOSE_SUB_COST = 0.25
CLOSE_SUB_COST = 0.45
MEDIUM_SUB_COST = 0.75
MAJOR_SUB_COST = 1.05
VOWEL_CONSONANT_SUB_COST = 1.35
UNKNOWN_SUB_COST = 1.20

# Distance thresholds that map a raw articulatory distance to a severity band.
VERY_CLOSE_MAX_DISTANCE = 0.15
CLOSE_MAX_DISTANCE = 0.35
MEDIUM_MAX_DISTANCE = 0.65


def substitution_cost_and_label(ref_ph: str, hyp_ph: str) -> Tuple[str, float]:
    """Authoritative (result_label, alignment_cost) for one aligned pair.

    Returns the *alignment cost* (for the DP), never a distance. The caller
    gets the raw articulatory distance separately via ``articulatory_distance``.
    """
    ref_ph = canonicalize_phoneme(ref_ph)
    hyp_ph = canonicalize_phoneme(hyp_ph)

    if ref_ph == hyp_ph:
        return "correct", MATCH_COST

    if not in_inventory(ref_ph) or not in_inventory(hyp_ph):
        return "unknown_substitution", UNKNOWN_SUB_COST

    if is_vowel(ref_ph) != is_vowel(hyp_ph):
        return "vowel_consonant_substitution", VOWEL_CONSONANT_SUB_COST

    distance = articulatory_distance(ref_ph, hyp_ph)
    if distance <= VERY_CLOSE_MAX_DISTANCE:
        return "very_close_substitution", VERY_CLOSE_SUB_COST
    if distance <= CLOSE_MAX_DISTANCE:
        return "close_substitution", CLOSE_SUB_COST
    if distance <= MEDIUM_MAX_DISTANCE:
        return "medium_substitution", MEDIUM_SUB_COST
    return "major_substitution", MAJOR_SUB_COST


def alignment_substitution_cost(ref_ph: str, hyp_ph: str) -> float:
    """Just the DP alignment cost for a ref/hyp pair."""
    return substitution_cost_and_label(ref_ph, hyp_ph)[1]


def classify_substitution(ref_ph: str, hyp_ph: str, distance_value: float | None = None) -> str:
    """Result label for a ref/hyp pair (correct / *_substitution)."""
    return substitution_cost_and_label(ref_ph, hyp_ph)[0]


# -----------------------------------------------------------------------------
# Soft mastery evidence (provisional)
# -----------------------------------------------------------------------------
# Result labels that count as a hard miss (0.0 evidence) regardless of distance.
_ZERO_EVIDENCE_LABELS = {
    "deletion",
    "unknown_substitution",
    "vowel_consonant_substitution",
}


def mastery_observation(result_label: str, articulatory_distance_value: float) -> float:
    """Provisional soft evidence in [0, 1] for one aligned occurrence.

    correct                         -> 1.0
    deletion                        -> 0.0
    unknown substitution            -> 0.0
    vowel/consonant substitution    -> 0.0
    other substitution              -> clamp(1 - articulatory_distance, 0, 1)

    Deliberately isolated so it can later be swapped for a calibrated GOP
    posterior without touching the Beta-update machinery in mastery.py.
    Insertions have no expected phoneme and are never passed here.
    """
    if result_label == "correct":
        return 1.0
    if result_label in _ZERO_EVIDENCE_LABELS:
        return 0.0
    # very_close / close / medium / major substitution.
    return max(0.0, min(1.0, 1.0 - float(articulatory_distance_value)))


# -----------------------------------------------------------------------------
# Startup inventory validation
# -----------------------------------------------------------------------------
def validate_g2p_inventory(phonemes: Iterable[str]) -> Dict[str, object]:
    """Check that every supplied phoneme maps into the canonical scoring
    inventory. Unsupported phonemes must never silently enter scoring.

    Returns a report dict: ``ok`` (bool), ``unsupported`` (sorted list of the
    original tokens), and ``checked`` (count).
    """
    unsupported = []
    checked = 0
    for raw in phonemes:
        checked += 1
        if not canonicalize_phoneme(raw):
            continue
        if not in_inventory(raw):
            unsupported.append(raw)
    return {
        "ok": not unsupported,
        "unsupported": sorted(set(unsupported)),
        "checked": checked,
        "inventory_size": len(canonical_inventory()),
        "scoring_engine": scoring_engine(),
    }


def build_phoneme_vector_table(
    phonemes: Iterable[str] = KNOWN_IPA_PHONEMES,
) -> Dict[str, Tuple[float, ...]]:
    """Build PanPhon vectors for every phoneme in the inventory (diagnostic)."""
    table: Dict[str, Tuple[float, ...]] = {}
    for ph in phonemes:
        canon = canonicalize_phoneme(ph)
        if canon and canon not in table:
            table[canon] = phoneme_vector(canon)
    return table


def substitution_label(distance_value: float) -> str:
    """Backward-compatible distance-only label (prefer classify_substitution)."""
    if distance_value <= VERY_CLOSE_MAX_DISTANCE:
        return "minor_substitution"
    if distance_value <= CLOSE_MAX_DISTANCE:
        return "medium_substitution"
    return "major_substitution"


if __name__ == "__main__":
    print("scoring_engine:", scoring_engine())
    print("inventory size:", len(canonical_inventory()))
    print("inventory:", sorted(canonical_inventory()))
    print("distance(t, d)=", round(articulatory_distance("t", "d"), 3))
    print("distance(θ, s)=", round(articulatory_distance("θ", "s"), 3))
