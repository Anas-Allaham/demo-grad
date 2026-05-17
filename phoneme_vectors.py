"""
Research-based IPA phoneme distance for pronunciation feedback.

Primary method:
- PanPhon articulatory feature vectors and weighted feature edit distance.
- Paper: Mortensen et al. (2016), "PanPhon: A Resource for Mapping IPA Segments
  to Articulatory Feature Vectors".

Fallback:
- A very small feature fallback is included only so the app can still run if PanPhon
  is not installed. For final/report use, install panphon.
"""

from __future__ import annotations

import itertools
import unicodedata
from functools import lru_cache
from typing import Dict, List

# The app mostly needs American-English IPA phonemes produced by the bundled G2P
# and your Wav2Vec2 model.
KNOWN_IPA_PHONEMES = [
    "p", "b", "t", "d", "k", "ɡ", "g",
    "f", "v", "θ", "ð", "s", "z", "ʃ", "ʒ", "h",
    "tʃ", "dʒ",
    "m", "n", "ŋ",
    "l", "ɹ", "r", "w", "j",
    "i", "iː", "ɪ", "e", "ɛ", "æ", "ɑ", "ɔ", "ʊ", "u", "uː", "ʌ", "ə", "ɝ", "ɚ",
    "aɪ", "aʊ", "eɪ", "oʊ", "ɔɪ",
]

PHONEME_ALIASES: Dict[str, str] = {
    # Common Unicode/IPA symbol variants.
    "g": "ɡ",
    "ɚ": "ɝ",
    "ɜː": "ɝ",
    "ɜ:": "ɝ",
    "ə˞": "ɝ",
    "˞": "",
    "ɑː": "ɑ",
    "ɑ:": "ɑ",
    "ɔː": "ɔ",
    "ɔ:": "ɔ",
    "u:": "uː",
    "i:": "iː",
    "ɫ": "l",
}

# Tiny fallback feature set. This is not used when PanPhon is installed.
FALLBACK_CLASSES = {
    "vowels": {"i", "iː", "ɪ", "e", "ɛ", "æ", "ɑ", "ɔ", "ʊ", "u", "uː", "ʌ", "ə", "ɝ", "aɪ", "aʊ", "eɪ", "oʊ", "ɔɪ"},
    "stops": {"p", "b", "t", "d", "k", "ɡ"},
    "fricatives": {"f", "v", "θ", "ð", "s", "z", "ʃ", "ʒ", "h"},
    "affricates": {"tʃ", "dʒ"},
    "nasals": {"m", "n", "ŋ"},
    "liquids": {"l", "ɹ", "r"},
    "glides": {"w", "j"},
    "lateral": {"l"},
    "rhotic": {"ɹ", "r"},
    "voiced": {"b", "d", "ɡ", "v", "ð", "z", "ʒ", "dʒ", "m", "n", "ŋ", "l", "ɹ", "r", "w", "j"},
    "bilabial": {"p", "b", "m"},
    "labiodental": {"f", "v"},
    "dental": {"θ", "ð"},
    "alveolar": {"t", "d", "s", "z", "n", "l", "ɹ", "r"},
    "postalveolar": {"ʃ", "ʒ", "tʃ", "dʒ"},
    "velar": {"k", "ɡ", "ŋ", "w"},
    "front_vowels": {"i", "iː", "ɪ", "e", "ɛ", "æ", "eɪ"},
    "central_vowels": {"ʌ", "ə", "ɝ"},
    "back_vowels": {"ɑ", "ɔ", "ʊ", "u", "uː", "oʊ"},
}


def canonicalize_phoneme(phoneme: str) -> str:
    """Normalize one IPA token before distance scoring."""
    ph = unicodedata.normalize("NFC", str(phoneme)).strip()
    if not ph:
        return ph
    ph = ph.replace(":", "ː")
    for _ in range(4):
        mapped = PHONEME_ALIASES.get(ph)
        if mapped is None or mapped == ph:
            break
        ph = mapped
    return ph


@lru_cache(maxsize=1)
def _panphon_distance_object():
    try:
        from panphon.distance import Distance
        return Distance()
    except Exception:
        return None


def panphon_available() -> bool:
    return _panphon_distance_object() is not None


@lru_cache(maxsize=1)
def _max_panphon_substitution_distance() -> float:
    """Normalize PanPhon substitution distances over the app's IPA inventory."""
    distance = _panphon_distance_object()
    if distance is None:
        return 1.0

    phonemes = sorted({canonicalize_phoneme(p) for p in KNOWN_IPA_PHONEMES if canonicalize_phoneme(p)})
    max_dist = 0.0

    for a, b in itertools.combinations(phonemes, 2):
        try:
            raw = float(distance.weighted_feature_edit_distance(a, b))
            if raw > max_dist:
                max_dist = raw
        except Exception:
            continue

    return max(max_dist, 1.0)


def _fallback_vector(phoneme: str) -> List[int]:
    phoneme = canonicalize_phoneme(phoneme)
    return [1 if phoneme in members else 0 for members in FALLBACK_CLASSES.values()]


def _fallback_distance(a: str, b: str) -> float:
    a = canonicalize_phoneme(a)
    b = canonicalize_phoneme(b)
    if a == b:
        return 0.0

    va = _fallback_vector(a)
    vb = _fallback_vector(b)
    if not any(va) or not any(vb):
        return 1.0

    diff = sum(1 for x, y in zip(va, vb) if x != y)
    return min(1.0, diff / max(len(va), 1))


def phoneme_distance(a: str, b: str) -> float:
    """
    Return a normalized distance between two IPA phonemes.

    0.0 means identical phoneme.
    1.0 means maximally different or unknown.
    """
    a = canonicalize_phoneme(a)
    b = canonicalize_phoneme(b)

    if a == b:
        return 0.0

    distance = _panphon_distance_object()
    if distance is None:
        return _fallback_distance(a, b)

    try:
        raw = float(distance.weighted_feature_edit_distance(a, b))
        normalized = raw / _max_panphon_substitution_distance()
        return max(0.0, min(1.0, normalized))
    except Exception:
        return 1.0


def substitution_label(distance_value: float) -> str:
    """Map a normalized distance to an interpretable substitution label."""
    if distance_value <= 0.30:
        return "minor_substitution"
    if distance_value <= 0.60:
        return "medium_substitution"
    return "major_substitution"
