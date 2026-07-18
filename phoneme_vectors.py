"""
Compatibility shim.

``phoneme_vectors`` used to be a second, competing phoneme-distance
implementation. It is now a thin wrapper that re-exports the single source of
truth in ``phoneme_vectors_professional`` so nothing in the project keeps two
divergent canonicalizers / distance functions alive. Prefer importing from
``phoneme_vectors_professional`` directly in new code.
"""

from __future__ import annotations

from phoneme_vectors_professional import (  # noqa: F401
    ASSESSABLE_INVENTORY,
    CONSONANT_PHONEMES,
    KNOWN_IPA_PHONEMES,
    PHONEME_ALIASES,
    VOWEL_PHONEMES,
    articulatory_distance,
    canonical_inventory,
    canonicalize_phoneme,
    classify_substitution,
    is_assessable,
    is_consonant,
    is_vowel,
    panphon_available,
    phoneme_distance,
    scoring_engine,
    substitution_label,
    validate_g2p_inventory,
)

__all__ = [
    "ASSESSABLE_INVENTORY",
    "CONSONANT_PHONEMES",
    "KNOWN_IPA_PHONEMES",
    "PHONEME_ALIASES",
    "VOWEL_PHONEMES",
    "articulatory_distance",
    "canonical_inventory",
    "canonicalize_phoneme",
    "classify_substitution",
    "is_assessable",
    "is_consonant",
    "is_vowel",
    "panphon_available",
    "phoneme_distance",
    "scoring_engine",
    "substitution_label",
    "validate_g2p_inventory",
]
