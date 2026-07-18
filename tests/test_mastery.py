"""Tests 6-10: soft evidence, per-recording updates, decay."""

from datetime import datetime, timedelta, timezone

import mastery
from phoneme_vectors_professional import mastery_observation

FIXED_NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


# 6. Very-close substitutions produce partial (not zero, not full) evidence.
def test_very_close_substitution_partial_evidence():
    obs = mastery_observation("very_close_substitution", 0.1)
    assert obs == 0.9
    assert 0.0 < obs < 1.0

    rows = [{"expected": "θ", "spoken": "s", "result": "close_substitution",
             "articulatory_distance": 0.2}]
    stats = mastery.update_mastery_for_recording({}, rows, FIXED_NOW)
    stat = stats["θ"]
    # alpha += 0.8, beta += 0.2  -> partial credit, not a flat miss.
    assert abs(stat.alpha - 1.8) < 1e-9
    assert abs(stat.beta - 1.2) < 1e-9


# 7. Deletions produce zero evidence.
def test_deletion_zero_evidence():
    assert mastery_observation("deletion", 0.0) == 0.0
    rows = [{"expected": "θ", "spoken": "-", "result": "deletion",
             "articulatory_distance": None}]
    stats = mastery.update_mastery_for_recording({}, rows, FIXED_NOW)
    stat = stats["θ"]
    assert abs(stat.alpha - 1.0) < 1e-9   # no success mass added
    assert abs(stat.beta - 2.0) < 1e-9    # one full miss


def test_unknown_and_vowel_consonant_substitutions_are_zero_evidence():
    assert mastery_observation("unknown_substitution", 0.05) == 0.0
    assert mastery_observation("vowel_consonant_substitution", 0.05) == 0.0


# 8. Three occurrences in one recording -> ONE independent attempt.
def test_three_occurrences_one_recording_increments_attempts_once():
    rows = [
        {"expected": "θ", "spoken": "θ", "result": "correct", "articulatory_distance": 0.0},
        {"expected": "θ", "spoken": "θ", "result": "correct", "articulatory_distance": 0.0},
        {"expected": "θ", "spoken": "θ", "result": "correct", "articulatory_distance": 0.0},
    ]
    stats = mastery.update_mastery_for_recording({}, rows, FIXED_NOW)
    stat = stats["θ"]
    assert stat.independent_attempts == 1     # NOT 3
    assert stat.occurrence_count == 3
    # One Beta update from the mean observation (1.0): alpha 1->2, beta stays 1.
    assert abs(stat.alpha - 2.0) < 1e-9
    assert abs(stat.beta - 1.0) < 1e-9
    # A single recording must not mark the phoneme mastered.
    assert not mastery.is_mastered(stat, FIXED_NOW)


# 9. Half-life retains exactly 50% of the evidence.
def test_half_life_retains_exactly_fifty_percent():
    assert mastery.decay_factor(mastery.DECAY_HALF_LIFE_DAYS) == 0.5
    stat = mastery.PhonemeStat(alpha=5.0, beta=3.0, independent_attempts=4,
                               last_practiced_at=FIXED_NOW)
    later = FIXED_NOW + timedelta(days=mastery.DECAY_HALF_LIFE_DAYS)
    decayed = mastery.decayed_stat(stat, later)
    # Evidence above the prior (1,1) is halved.
    assert abs((decayed.alpha - 1.0) - (5.0 - 1.0) * 0.5) < 1e-9
    assert abs((decayed.beta - 1.0) - (3.0 - 1.0) * 0.5) < 1e-9


# 10. Stale phonemes decay during ranking, surfacing for re-test.
def test_stale_phoneme_decays_during_ranking():
    # 's' is weak but practiced now; 'z' is strong but very old. (Both are real
    # inventory phonemes so ranking considers them.)
    stats = {
        "s": mastery.PhonemeStat(alpha=2.0, beta=1.5, independent_attempts=3,
                                 last_practiced_at=FIXED_NOW),
        "z": mastery.PhonemeStat(alpha=20.0, beta=2.0, independent_attempts=10,
                                 last_practiced_at=FIXED_NOW - timedelta(days=400)),
    }
    # Rank far in the future: 'z' has decayed back toward the prior, so its
    # lower-confidence bound drops and it ranks as weak/uncertain.
    future = FIXED_NOW + timedelta(days=1)
    ranked = mastery.rank_weak_phonemes(stats, top_k=1, epsilon=0.0, now=future)
    assert ranked == ["z"]

    # Without decay (now=None) the strong, stale 'z' would NOT rank first.
    ranked_no_decay = mastery.rank_weak_phonemes(stats, top_k=1, epsilon=0.0, now=None)
    assert ranked_no_decay == ["s"]
