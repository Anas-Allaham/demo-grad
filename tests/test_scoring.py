"""Test 5: articulatory distance, alignment cost, and score are separate."""

from phoneme_vectors_professional import (
    DELETION_COST,
    MATCH_COST,
    alignment_substitution_cost,
    articulatory_distance,
    mastery_observation,
    substitution_cost_and_label,
)
from scoring import align_phonemes, calculate_metrics


def test_distance_cost_and_score_are_separate_concepts():
    # A substitution row exposes THREE distinct numbers.
    rows = align_phonemes(["θ"], ["s"])
    assert len(rows) == 1
    row = rows[0]
    assert row["expected"] == "θ" and row["spoken"] == "s"

    art = row["articulatory_distance"]          # raw feature distance [0,1]
    cost = row["alignment_cost"]                # DP cost, from a cost table
    obs = mastery_observation(row["result"], art)  # soft evidence [0,1]

    # They are computed differently and generally differ in value.
    assert 0.0 <= art <= 1.0
    assert cost in {0.25, 0.45, 0.75, 1.05, 1.20, 1.35}   # a cost band, not a distance
    assert cost != art
    # Score is derived from the distance, not equal to it or the cost.
    assert abs(obs - (1.0 - art)) < 1e-9


def test_correct_pair_has_zero_distance_and_zero_cost():
    label, cost = substitution_cost_and_label("t", "t")
    assert label == "correct"
    assert cost == MATCH_COST == 0.0
    assert articulatory_distance("t", "t") == 0.0


def test_alignment_cost_is_not_a_distance_for_gaps():
    # Deletion uses the alignment cost, which is not an articulatory distance.
    rows = align_phonemes(["t", "s"], ["t"])
    deletion = [r for r in rows if r["result"] == "deletion"][0]
    assert deletion["articulatory_distance"] is None
    assert deletion["alignment_cost"] == round(DELETION_COST, 3)
    assert alignment_substitution_cost("s", "s") == 0.0


def test_metrics_names_preserve_insertion_overflow():
    # 1 reference phoneme, 3 insertions -> weighted PER can exceed 100.
    rows = align_phonemes(["s"], ["s", "s", "s", "s"])
    m = calculate_metrics(rows)
    assert m["raw_weighted_per"] > 100.0
    assert m["display_error_percent"] == 100.0
    assert m["display_accuracy_percent"] == 0.0
    # phoneme_error_rate stays the capped, backward-compatible field.
    assert m["phoneme_error_rate"] == m["display_error_percent"]
