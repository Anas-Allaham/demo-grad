"""
Phoneme alignment and metric scoring.

Keeps three concepts that the old code conflated strictly separate:

  1. articulatory_distance  - raw normalized PanPhon feature distance in [0, 1]
                              (descriptive similarity; provisional, not GOP).
  2. alignment_cost         - cost used ONLY by the DP alignment.
  3. mastery_observation    - soft evidence, computed later in mastery.py.

All alignment/substitution policy is imported from
``phoneme_vectors_professional`` - this module owns none of it, so there is a
single authoritative implementation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from phoneme_vectors_professional import (
    DELETION_COST,
    INSERTION_COST,
    MATCH_COST,
    articulatory_distance,
    substitution_cost_and_label,
)


def align_phonemes(ref_seq: Sequence[str], hyp_seq: Sequence[str]) -> List[Dict[str, Any]]:
    """Needleman-Wunsch alignment of reference vs hypothesis phonemes.

    Returns a list of alignment rows, each:
        {
            "expected": "θ" | "-",
            "spoken":   "s" | "-",
            "result":   "close_substitution" | "correct" | "deletion" | ...,
            "articulatory_distance": 0.18 | None,   # None for gaps
            "alignment_cost": 0.45,                  # DP cost, NOT a distance
        }
    """
    n, m = len(ref_seq), len(hyp_seq)
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    backtrack: List[List[Optional[str]]] = [[None] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = i * DELETION_COST
        backtrack[i][0] = "UP"
    for j in range(1, m + 1):
        dp[0][j] = j * INSERTION_COST
        backtrack[0][j] = "LEFT"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            _, sub_cost = substitution_cost_and_label(ref_seq[i - 1], hyp_seq[j - 1])
            diag = dp[i - 1][j - 1] + sub_cost
            up = dp[i - 1][j] + DELETION_COST
            left = dp[i][j - 1] + INSERTION_COST
            best = min(diag, up, left)
            dp[i][j] = best
            # Prefer exact match, then gaps, then substitution on ties.
            if sub_cost == MATCH_COST and diag == best:
                backtrack[i][j] = "DIAG"
            elif up == best:
                backtrack[i][j] = "UP"
            elif left == best:
                backtrack[i][j] = "LEFT"
            else:
                backtrack[i][j] = "DIAG"

    rows: List[Dict[str, Any]] = []
    i, j = n, m
    while i > 0 or j > 0:
        move = backtrack[i][j]
        if move == "DIAG":
            ref_ph, hyp_ph = ref_seq[i - 1], hyp_seq[j - 1]
            label, cost = substitution_cost_and_label(ref_ph, hyp_ph)
            dist = 0.0 if label == "correct" else articulatory_distance(ref_ph, hyp_ph)
            rows.append({
                "expected": ref_ph,
                "spoken": hyp_ph,
                "result": label,
                "articulatory_distance": round(dist, 3),
                "alignment_cost": round(cost, 3),
            })
            i -= 1
            j -= 1
        elif move == "UP":
            rows.append({
                "expected": ref_seq[i - 1],
                "spoken": "-",
                "result": "deletion",
                "articulatory_distance": None,
                "alignment_cost": round(DELETION_COST, 3),
            })
            i -= 1
        else:  # LEFT
            rows.append({
                "expected": "-",
                "spoken": hyp_seq[j - 1],
                "result": "insertion",
                "articulatory_distance": None,
                "alignment_cost": round(INSERTION_COST, 3),
            })
            j -= 1

    rows.reverse()
    return rows


def to_api_alignment(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Add the legacy ``distance`` field (== articulatory_distance) so the
    existing UI table keeps working, without losing the new explicit fields."""
    api_rows = []
    for row in rows:
        api_row = dict(row)
        api_row["distance"] = row["articulatory_distance"]
        api_rows.append(api_row)
    return api_rows


def calculate_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute error/accuracy metrics from alignment rows.

    Metric naming (see README): standard/weighted PER can exceed 100% because
    insertions add cost without adding reference units. That information is
    preserved rather than silently clamped:

        raw_weighted_per         - weighted PER, may exceed 100.
        display_error_percent    - raw_weighted_per clamped to [0, 100].
        display_accuracy_percent - clamp(100 - raw_weighted_per, 0, 100).

    The weighted metric uses PROVISIONAL articulatory/alignment penalties and
    is NOT a calibrated GOP score.
    """
    results = [r["result"] for r in rows]
    total_reference_units = sum(1 for r in results if r != "insertion")

    def _count(label: str) -> int:
        return results.count(label)

    very_close = _count("very_close_substitution")
    close = _count("close_substitution")
    medium = _count("medium_substitution")
    major = _count("major_substitution")
    vowel_consonant = _count("vowel_consonant_substitution")
    unknown = _count("unknown_substitution")
    deletions = _count("deletion")
    insertions = _count("insertion")
    correct = _count("correct")
    substitutions = sum(1 for r in results if r.endswith("_substitution"))

    weighted_error = sum(r["alignment_cost"] for r in rows if r["result"] != "correct")

    if total_reference_units > 0:
        raw_weighted_per = (weighted_error / total_reference_units) * 100.0
    else:
        raw_weighted_per = 0.0

    display_error_percent = max(0.0, min(100.0, raw_weighted_per))
    display_accuracy_percent = max(0.0, min(100.0, 100.0 - raw_weighted_per))

    return {
        "correct": correct,
        "substitutions": substitutions,
        "minor_substitutions": very_close + close,
        "medium_substitutions": medium,
        "major_substitutions": major + vowel_consonant + unknown,
        "very_close_substitutions": very_close,
        "close_substitutions": close,
        "vowel_consonant_substitutions": vowel_consonant,
        "unknown_substitutions": unknown,
        "deletions": deletions,
        "insertions": insertions,
        "weighted_error": round(weighted_error, 3),
        "raw_weighted_per": round(raw_weighted_per, 2),
        "display_error_percent": round(display_error_percent, 2),
        "display_accuracy_percent": round(display_accuracy_percent, 2),
        # Backward-compatible field the current UI reads. Equals the capped
        # display error percent.
        "phoneme_error_rate": round(display_error_percent, 2),
    }


def align_and_score(ref_seq: Sequence[str], hyp_seq: Sequence[str]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Convenience: align, then compute metrics in one call."""
    rows = align_phonemes(ref_seq, hyp_seq)
    return rows, calculate_metrics(rows)
