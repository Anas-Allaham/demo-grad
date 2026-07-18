"""Test 17: every module shares ONE canonicalizer implementation."""

import assessment
import content
import mastery
import phoneme_vectors
import phoneme_vectors_professional as pro
import tokenization


def test_all_modules_use_the_same_canonicalizer():
    canonical = pro.canonicalize_phoneme
    assert mastery.canonicalize_phoneme is canonical
    assert tokenization.canonicalize_phoneme is canonical
    assert content.canonicalize_phoneme is canonical
    assert assessment.canonicalize_phoneme is canonical
    # The old module is now a compatibility wrapper re-exporting the same one.
    assert phoneme_vectors.canonicalize_phoneme is canonical


def test_compat_wrapper_reexports_professional_distance():
    assert phoneme_vectors.phoneme_distance is pro.phoneme_distance
    assert phoneme_vectors.articulatory_distance is pro.articulatory_distance
    assert phoneme_vectors.panphon_available is pro.panphon_available


def test_single_distance_and_cost_policy():
    # scoring.py imports its policy from the professional module (no local copy).
    import scoring
    assert scoring.articulatory_distance is pro.articulatory_distance
    assert scoring.substitution_cost_and_label is pro.substitution_cost_and_label
