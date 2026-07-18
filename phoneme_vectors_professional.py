"""
Professional IPA phoneme vectors for pronunciation feedback.

Primary source
--------------
PanPhon articulatory feature vectors:
Mortensen, Littell, Bharadwaj, Goyal, Dyer, and Levin (2016),
"PanPhon: A Resource for Mapping IPA Segments to Articulatory Feature Vectors",
COLING 2016.

This module does NOT hand-write phoneme feature vectors. It asks PanPhon for
IPA feature vectors and exposes a stable API for the Flask app:

    canonicalize_phoneme(ph)
    panphon_available()
    phoneme_vector(ph)
    phoneme_distance(a, b)
    substitution_label(distance)
    alignment_substitution_cost(a, b)
    classify_substitution(a, b, distance)

Install:
    pip install panphon

Notes
-----
PanPhon returns vectors for IPA segments. Some tokens used by this app are
composite phonemes, for example diphthongs like /aɪ/ or ASCII-style affricates
like /tʃ/. For a composite token, this module averages the PanPhon vectors of
its component IPA segments. This keeps one fixed-length vector per app phoneme.
"""

from __future__ import annotations

import math
import unicodedata
from functools import lru_cache
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

# Feature order used by current PanPhon FeatureTable examples.
# If your installed PanPhon exposes a different order, the vector values still
# come from PanPhon; this list is used for labels and robust class checks.
PANPHON_FEATURE_NAMES: Tuple[str, ...] = (
    "syl", "son", "cons", "cont", "delrel", "lat", "nas", "strid",
    "voi", "sg", "cg", "ant", "cor", "distr", "lab", "hi", "lo",
    "back", "round", "velaric", "tense", "long",
)

# The app mostly needs American-English IPA phonemes produced by your G2P and
# Wav2Vec2 phoneme model.
KNOWN_IPA_PHONEMES: Tuple[str, ...] = (
    "p", "b", "t", "d", "k", "ɡ", "g",
    "f", "v", "θ", "ð", "s", "z", "ʃ", "ʒ", "h",
    "tʃ", "dʒ",
    "m", "n", "ŋ",
    "l", "ɹ", "r", "w", "j",
    "i", "iː", "ɪ", "e", "ɛ", "æ", "ɑ", "ɔ", "ʊ", "u", "uː",
    "ʌ", "ə", "ɝ", "ɚ",
    "aɪ", "aʊ", "eɪ", "oʊ", "ɔɪ",
)

# Normalization only: these are symbol aliases, not feature definitions.
PHONEME_ALIASES: Dict[str, str] = {
    "g": "ɡ",
    "ɫ": "l",
    "r": "ɹ",
    "ɚ": "ɝ",
    "ɜː": "ɝ",
    "ɜ:": "ɝ",
    "ə˞": "ɝ",
    "ɑː": "ɑ",
    "ɑ:": "ɑ",
    "ɔː": "ɔ",
    "ɔ:": "ɔ",
    "i:": "iː",
    "u:": "uː",
    # Tie-bar versions are more IPA-canonical for affricates, but many ASR/G2P
    # systems output tʃ and dʒ. Keep the app tokens stable and let PanPhon parse.
    "t͡ʃ": "tʃ",
    "d͡ʒ": "dʒ",
}

# Tokens that should be represented as a single app phoneme but may be analyzed
# by PanPhon as multiple IPA segments. Vector = mean(component vectors).
COMPOSITE_COMPONENTS: Mapping[str, Tuple[str, ...]] = {
    "tʃ": ("t", "ʃ"),
    "dʒ": ("d", "ʒ"),
    "aɪ": ("a", "ɪ"),
    "aʊ": ("a", "ʊ"),
    "eɪ": ("e", "ɪ"),
    "oʊ": ("o", "ʊ"),
    "ɔɪ": ("ɔ", "ɪ"),
}

# Feature weights used only for vector-distance normalization. These are not
# hand-written phoneme vectors. They make major class differences count more in
# the distance while still relying on PanPhon for the actual feature values.
FEATURE_WEIGHTS: Mapping[str, float] = {
    "syl": 2.0,
    "son": 1.5,
    "cons": 1.5,
    "cont": 1.0,
    "delrel": 0.75,
    "lat": 0.75,
    "nas": 1.0,
    "strid": 0.75,
    "voi": 0.75,
    "sg": 0.5,
    "cg": 0.5,
    "ant": 0.75,
    "cor": 0.75,
    "distr": 0.5,
    "lab": 0.75,
    "hi": 1.0,
    "lo": 1.0,
    "back": 1.0,
    "round": 0.75,
    "velaric": 0.25,
    "tense": 0.75,
    "long": 0.5,
}


def _patch_panphon_utf8_read() -> None:
    """
    PanPhon on Windows can open its CSV files with locale encoding (cp1252),
    which breaks on IPA bytes. Patch FeatureTable._read_bases to force UTF-8.
    """
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
        seg_dict = dict(segments)
        return segments, seg_dict, feature_names

    ft_mod.FeatureTable._read_bases = _read_bases_utf8
    ft_mod.FeatureTable._utf8_read_patch_applied = True


def canonicalize_phoneme(phoneme: str) -> str:
    """Normalize an IPA token before vector lookup or distance scoring."""
    ph = unicodedata.normalize("NFC", str(phoneme)).strip()
    if not ph:
        return ph

    ph = ph.replace(":", "ː")
    ph = ph.replace("ˈ", "").replace("ˌ", "")
    ph = ph.replace("/", "").replace("[", "").replace("]", "")

    # Apply aliases recursively, but avoid infinite loops.
    for _ in range(8):
        mapped = PHONEME_ALIASES.get(ph)
        if mapped is None or mapped == ph:
            break
        ph = mapped
    return ph


@lru_cache(maxsize=1)
def _feature_table():
    try:
        import panphon
        _patch_panphon_utf8_read()
        return panphon.FeatureTable()
    except Exception:
        return None


@lru_cache(maxsize=1)
def _distance_object():
    try:
        from panphon.distance import Distance
        _patch_panphon_utf8_read()
        return Distance()
    except Exception:
        return None


def panphon_available() -> bool:
    """Return True only when the real PanPhon library can be imported."""
    return _feature_table() is not None and _distance_object() is not None


def require_panphon() -> None:
    if not panphon_available():
        raise RuntimeError(
            "PanPhon is required for professional phoneme vectors. "
            "Install it with: pip install panphon"
        )


def _as_numeric_vector(row: Sequence[object]) -> Tuple[float, ...]:
    """Convert a PanPhon vector row to numeric values -1, 0, +1."""
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
        raise ValueError("Cannot average empty vector list.")
    width = len(vectors[0])
    return tuple(sum(float(vec[i]) for vec in vectors) / len(vectors) for i in range(width))


@lru_cache(maxsize=512)
def _segment_vector(segment: str) -> Tuple[float, ...]:
    """Return PanPhon vector for one IPA segment, not an app-level phoneme."""
    require_panphon()
    ft = _feature_table()
    assert ft is not None

    # numeric=True is available in PanPhon FeatureTable.word_to_vector_list.
    rows = ft.word_to_vector_list(segment, numeric=True)
    if not rows:
        raise ValueError(f"PanPhon could not vectorize IPA segment: {segment!r}")

    # For a true segment this should be exactly one row. If PanPhon splits it,
    # average so the caller still receives one fixed-width vector.
    return _mean_vectors([_as_numeric_vector(row) for row in rows])


@lru_cache(maxsize=512)
def phoneme_vector(phoneme: str) -> Tuple[float, ...]:
    """
    Return one fixed-width articulatory vector for an app phoneme.

    Values come from PanPhon. Composite phonemes are represented by the mean of
    their component segment vectors.
    """
    ph = canonicalize_phoneme(phoneme)
    if not ph:
        raise ValueError("Empty phoneme cannot be vectorized.")

    components = COMPOSITE_COMPONENTS.get(ph)
    if components:
        return _mean_vectors([_segment_vector(component) for component in components])

    return _segment_vector(ph)


def phoneme_features(phoneme: str) -> Dict[str, float]:
    """Return a named PanPhon feature dictionary for one phoneme."""
    vector = phoneme_vector(phoneme)
    names = PANPHON_FEATURE_NAMES[: len(vector)]
    return dict(zip(names, vector))


def build_phoneme_vector_table(
    phonemes: Iterable[str] = KNOWN_IPA_PHONEMES,
) -> Dict[str, Tuple[float, ...]]:
    """Build vectors for every phoneme in the app inventory."""
    table: Dict[str, Tuple[float, ...]] = {}
    for ph in phonemes:
        canon = canonicalize_phoneme(ph)
        if canon and canon not in table:
            table[canon] = phoneme_vector(canon)
    return table


def _weighted_l1_distance(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    """Normalized weighted L1 distance over PanPhon feature vectors."""
    width = min(len(vec_a), len(vec_b), len(PANPHON_FEATURE_NAMES))
    if width == 0:
        return 1.0

    weighted_diff = 0.0
    max_weighted_diff = 0.0

    for i in range(width):
        feature_name = PANPHON_FEATURE_NAMES[i]
        weight = FEATURE_WEIGHTS.get(feature_name, 1.0)
        weighted_diff += weight * abs(float(vec_a[i]) - float(vec_b[i]))
        # PanPhon feature values are in {-1, 0, +1}, so maximum difference is 2.
        max_weighted_diff += weight * 2.0

    if max_weighted_diff <= 0:
        return 1.0
    return max(0.0, min(1.0, weighted_diff / max_weighted_diff))


def is_vowel(phoneme: str) -> bool:
    """Use PanPhon's syllabic feature rather than a hand-written vowel list."""
    try:
        features = phoneme_features(phoneme)
        return features.get("syl", 0.0) > 0.25
    except Exception:
        return False


def phoneme_distance(a: str, b: str) -> float:
    """
    Normalized phoneme distance in [0, 1] using PanPhon vectors.

    0.0 = identical after canonicalization.
    1.0 = maximally different or not vectorizable.
    """
    a = canonicalize_phoneme(a)
    b = canonicalize_phoneme(b)

    if a == b:
        return 0.0

    vec_a = phoneme_vector(a)
    vec_b = phoneme_vector(b)
    return _weighted_l1_distance(vec_a, vec_b)


def alignment_substitution_cost(ref_ph: str, hyp_ph: str) -> float:
    """
    Cost for dynamic-programming alignment, separate from display distance.

    This prevents bad shifted alignments where a vowel is aligned to a consonant
    just because the raw feature distance is numerically small.
    """
    ref_ph = canonicalize_phoneme(ref_ph)
    hyp_ph = canonicalize_phoneme(hyp_ph)

    if ref_ph == hyp_ph:
        return 0.0

    distance = phoneme_distance(ref_ph, hyp_ph)

    # Vowel/consonant substitutions should almost always be worse than a gap.
    if is_vowel(ref_ph) != is_vowel(hyp_ph):
        return 1.35

    if distance <= 0.15:
        return 0.25
    if distance <= 0.30:
        return 0.45
    if distance <= 0.50:
        return 0.75
    if distance <= 0.75:
        return 1.05
    return 1.20


def classify_substitution(ref_ph: str, hyp_ph: str, distance_value: float | None = None) -> str:
    """Classify a substitution using PanPhon features plus major class guard."""
    ref_ph = canonicalize_phoneme(ref_ph)
    hyp_ph = canonicalize_phoneme(hyp_ph)

    if ref_ph == hyp_ph:
        return "correct"

    if is_vowel(ref_ph) != is_vowel(hyp_ph):
        return "major_substitution"

    distance_value = phoneme_distance(ref_ph, hyp_ph) if distance_value is None else distance_value

    if distance_value <= 0.15:
        return "minor_substitution"
    if distance_value <= 0.35:
        return "medium_substitution"
    return "major_substitution"


def substitution_label(distance_value: float) -> str:
    """
    Backward-compatible label function for app.py.

    Prefer classify_substitution(ref_ph, hyp_ph, distance) when both phonemes
    are available, because it can guard vowel/consonant errors.
    """
    if distance_value <= 0.15:
        return "minor_substitution"
    if distance_value <= 0.35:
        return "medium_substitution"
    return "major_substitution"


if __name__ == "__main__":
    require_panphon()
    print("PanPhon is available.")
    print("Feature count:", len(phoneme_vector("p")))
    for ph in ["p", "b", "t", "d", "aɪ", "w", "ʌ", "n", "tʃ"]:
        print(ph, phoneme_vector(ph))
    print("distance(t, d)=", round(phoneme_distance("t", "d"), 3))
    print("distance(aɪ, w)=", round(phoneme_distance("aɪ", "w"), 3))
