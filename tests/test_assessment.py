"""Tests 11-12: evidence-aware level + continuing diagnostic coverage."""

from datetime import datetime, timezone

import assessment as A
import mastery

FIXED_NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


# 11. One observed phoneme cannot produce an "established" user level.
def test_one_phoneme_cannot_be_established():
    # Even with strong, multi-recording, multi-prompt evidence for ONE phoneme,
    # inventory coverage (1/40) is far too low to establish a level.
    stats = {"θ": mastery.PhonemeStat(alpha=6.0, beta=1.0, independent_attempts=5,
                                      occurrence_count=9, last_practiced_at=FIXED_NOW)}
    context = {"θ": {"recordings": 5, "distinct_prompts": 3, "occurrences": 9}}
    result = A.assess_user_level(stats, context, independent_recording_count=5, now=FIXED_NOW)

    assert result["assessment_status"] != "established"
    assert result["assessment_status"] == "insufficient_evidence"
    assert result["overall_level"] == "unknown"
    assert result["pronunciation_score"] is None
    assert result["eligible_phoneme_count"] == 1


def test_unknown_phonemes_are_not_counted_as_weak():
    stats = {"θ": mastery.PhonemeStat(alpha=6.0, beta=1.0, independent_attempts=5,
                                      occurrence_count=9, last_practiced_at=FIXED_NOW)}
    context = {"θ": {"recordings": 5, "distinct_prompts": 3, "occurrences": 9}}
    result = A.assess_user_level(stats, context, independent_recording_count=5, now=FIXED_NOW)
    weak_symbols = {w["phoneme"] for w in result["weak_phonemes"]}
    # Untracked/under-observed phonemes are "unknown", never automatically weak.
    assert "s" in result["unknown_phonemes"]
    assert "s" not in weak_symbols


# 12. Diagnostic mode continues while inventory coverage is incomplete.
def test_diagnostic_continues_until_coverage_reached():
    # Only 3 phonemes have enough contexts -> still diagnostic, and the
    # remaining inventory is offered as "uncovered" (drives next selection).
    sparse_context = {
        p: {"recordings": 3, "distinct_prompts": 2, "occurrences": 3}
        for p in ["s", "t", "n"]
    }
    diag = A.diagnostic_status(sparse_context)
    assert diag["in_diagnostic"] is True
    assert diag["covered_count"] == 3
    assert len(diag["uncovered"]) > 0
    assert "θ" in diag["uncovered"]


def test_diagnostic_ends_when_enough_phonemes_covered():
    from phoneme_vectors_professional import ASSESSABLE_INVENTORY
    # Cover more than the required fraction of the inventory.
    covered = sorted(ASSESSABLE_INVENTORY)[: int(len(ASSESSABLE_INVENTORY) * 0.7) + 1]
    rich_context = {p: {"recordings": 3, "distinct_prompts": 2, "occurrences": 3} for p in covered}
    diag = A.diagnostic_status(rich_context)
    assert diag["in_diagnostic"] is False
