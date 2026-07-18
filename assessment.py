"""
Evidence-aware pronunciation level assessment and confusion analysis.

Two responsibilities:

  1. ``assess_user_level`` - a level assessment that refuses to invent a level
     from thin evidence. A phoneme is level-eligible only after enough
     independent recordings across enough distinct prompts; the level is a
     macro-average of a CONSERVATIVE (lower-bound) posterior over those
     eligible phonemes, and the status is reported honestly
     (insufficient_evidence / provisional / established).

  2. Confusion helpers - aggregate the learner's real substitution pairs
     (e.g. θ->s, ð->d, ɪ->i, v->f) and pick an exercise type appropriate to a
     phoneme's current mastery.

Scientific honesty: the numbers below are PROVISIONAL. They are derived from
articulatory-distance soft evidence, not calibrated GOP probabilities, and are
NOT a CEFR level. Thresholds are configurable and labelled provisional.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import mastery
from phoneme_vectors_professional import (
    ASSESSABLE_INVENTORY,
    canonicalize_phoneme,
    is_assessable,
)

# ---- Eligibility (configurable, provisional) --------------------------------
MIN_RECORDINGS_FOR_ELIGIBLE = 3      # independent recordings before a phoneme counts
MIN_DISTINCT_PROMPTS = 2             # across at least this many different texts

# ---- Coverage -> assessment status ------------------------------------------
PROVISIONAL_COVERAGE = 0.25          # below this: insufficient_evidence
ESTABLISHED_COVERAGE = 0.60          # at/above this: established

# ---- Level thresholds on the 0-100 provisional score ------------------------
BEGINNER_MAX_SCORE = 55.0
INTERMEDIATE_MAX_SCORE = 78.0

# ---- Weak / strong classification (on the conservative estimate) ------------
WEAK_MASTERY_THRESHOLD = 0.50
STRONG_MASTERY_THRESHOLD = 0.80

# ---- Cold-start diagnostic phase (#9) ---------------------------------------
# A phoneme is "covered" for diagnostic purposes once it has been recorded in
# at least this many independent contexts. The diagnostic phase keeps serving
# broad, high-gain sentences until this fraction of the assessable inventory is
# covered, instead of narrowing to one sentence's phonemes after a single try.
DIAGNOSTIC_MIN_CONTEXTS = 2
DIAGNOSTIC_COVERAGE_FRACTION = 0.5


def diagnostic_status(context_stats: Dict[str, Dict[str, int]]) -> Dict[str, Any]:
    """Return the cold-start diagnostic state.

    ``in_diagnostic``   - True while inventory coverage is still incomplete.
    ``uncovered``       - assessable phonemes still lacking enough contexts
                          (these drive max-gain diagnostic sentence selection).
    ``covered_count``   - phonemes that have enough independent contexts.
    """
    covered = set()
    for phoneme, ctx in context_stats.items():
        canon = canonicalize_phoneme(phoneme)
        if is_assessable(canon) and ctx.get("recordings", 0) >= DIAGNOSTIC_MIN_CONTEXTS:
            covered.add(canon)
    uncovered = ASSESSABLE_INVENTORY - covered
    needed = int(round(DIAGNOSTIC_COVERAGE_FRACTION * len(ASSESSABLE_INVENTORY)))
    return {
        "in_diagnostic": len(covered) < needed and bool(uncovered),
        "uncovered": sorted(uncovered),
        "covered_count": len(covered),
        "coverage_target": needed,
        "inventory_size": len(ASSESSABLE_INVENTORY),
    }


def _level_from_score(score: Optional[float]) -> str:
    if score is None:
        return "unknown"
    if score < BEGINNER_MAX_SCORE:
        return "beginner"
    if score < INTERMEDIATE_MAX_SCORE:
        return "intermediate"
    return "advanced"


def assess_user_level(
    stats: Dict[str, "mastery.PhonemeStat"],
    context_stats: Dict[str, Dict[str, int]],
    independent_recording_count: int,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Produce an evidence-aware level assessment.

    ``stats``: {canonical phoneme -> PhonemeStat}.
    ``context_stats``: {phoneme -> {"recordings", "distinct_prompts", ...}}
        (from db.get_phoneme_context_stats), used for eligibility.
    ``independent_recording_count``: total scorable recordings by the user.
    """
    # Canonicalize/merge stats and context onto the assessable inventory only.
    tracked: Dict[str, "mastery.PhonemeStat"] = {}
    for phoneme, stat in stats.items():
        canon = canonicalize_phoneme(phoneme)
        if is_assessable(canon):
            tracked[canon] = stat

    context: Dict[str, Dict[str, int]] = {}
    for phoneme, ctx in context_stats.items():
        canon = canonicalize_phoneme(phoneme)
        if not is_assessable(canon):
            continue
        acc = context.setdefault(canon, {"recordings": 0, "distinct_prompts": 0, "occurrences": 0})
        acc["recordings"] += ctx.get("recordings", 0)
        acc["distinct_prompts"] = max(acc["distinct_prompts"], ctx.get("distinct_prompts", 0))
        acc["occurrences"] += ctx.get("occurrences", 0)

    eligible: List[str] = []
    for phoneme, stat in tracked.items():
        ctx = context.get(phoneme, {})
        recordings = max(ctx.get("recordings", 0), stat.independent_attempts)
        prompts = ctx.get("distinct_prompts", 0)
        if recordings >= MIN_RECORDINGS_FOR_ELIGIBLE and prompts >= MIN_DISTINCT_PROMPTS:
            eligible.append(phoneme)

    weak: List[Dict[str, Any]] = []
    strong: List[str] = []
    conservative_scores: List[float] = []
    mean_scores: List[float] = []

    for phoneme in eligible:
        stat = tracked[phoneme]
        lcb = mastery.lower_confidence_bound(stat, now=now)   # conservative
        mean = mastery.posterior_mean(stat, now=now)
        conservative_scores.append(lcb)
        mean_scores.append(mean)
        if mean >= STRONG_MASTERY_THRESHOLD:
            strong.append(phoneme)
        elif lcb < WEAK_MASTERY_THRESHOLD:
            weak.append({
                "phoneme": phoneme,
                "mastery": round(mean, 3),
                "lower_confidence_bound": round(lcb, 3),
            })

    weak.sort(key=lambda w: w["lower_confidence_bound"])

    # Unknown = assessable phonemes we cannot yet judge (untracked OR not
    # eligible). Explicitly NOT counted as weak.
    unknown = sorted(ASSESSABLE_INVENTORY - set(eligible))

    eligible_count = len(eligible)
    inventory_coverage = eligible_count / max(len(ASSESSABLE_INVENTORY), 1)

    if eligible_count == 0 or inventory_coverage < PROVISIONAL_COVERAGE:
        status = "insufficient_evidence"
    elif inventory_coverage >= ESTABLISHED_COVERAGE:
        status = "established"
    else:
        status = "provisional"

    if conservative_scores and status != "insufficient_evidence":
        # Macro (per-phoneme equal-weight) average of the conservative estimate.
        pronunciation_score: Optional[float] = round(
            100.0 * sum(conservative_scores) / len(conservative_scores), 1
        )
        ci_low = round(100.0 * min(conservative_scores), 1)
        ci_high = round(100.0 * (sum(mean_scores) / len(mean_scores)), 1)
        confidence_interval: Optional[List[float]] = [ci_low, ci_high]
    else:
        pronunciation_score = None
        confidence_interval = None

    return {
        "pronunciation_score": pronunciation_score,
        "overall_level": _level_from_score(pronunciation_score),
        "assessment_status": status,
        "inventory_coverage": round(inventory_coverage, 3),
        "eligible_phoneme_count": eligible_count,
        "tracked_phoneme_count": len(tracked),
        "independent_recording_count": independent_recording_count,
        "confidence_interval": confidence_interval,
        "weak_phonemes": weak,
        "unknown_phonemes": unknown,
        "strong_phonemes": sorted(strong),
        "provisional": True,
        "note": "Provisional articulatory-distance score; not a calibrated GOP or CEFR level.",
    }


# -----------------------------------------------------------------------------
# Confusion analysis (#11)
# -----------------------------------------------------------------------------
def canonical_confusions(confusion_pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Canonicalize and re-aggregate raw confusion rows, dropping legacy noise
    (punctuation, alias-only "substitutions" like r->ɹ where both canonicalize
    to the same phoneme, and anything outside the assessable inventory)."""
    merged: Dict[tuple, int] = {}
    for pair in confusion_pairs:
        exp = canonicalize_phoneme(pair.get("expected", ""))
        spo = canonicalize_phoneme(pair.get("spoken", ""))
        if not exp or not spo or exp == spo:
            continue
        if not is_assessable(exp):
            continue
        merged[(exp, spo)] = merged.get((exp, spo), 0) + int(pair.get("count", 0))
    out = [
        {"expected": exp, "spoken": spo, "count": count}
        for (exp, spo), count in merged.items()
        if count > 0
    ]
    out.sort(key=lambda p: p["count"], reverse=True)
    return out


def main_confusion_for(phoneme: str, confusion_pairs: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The most frequent thing ``phoneme`` gets substituted with."""
    phoneme = canonicalize_phoneme(phoneme)
    best: Optional[Dict[str, Any]] = None
    for pair in canonical_confusions(confusion_pairs):
        if pair["expected"] == phoneme:
            if best is None or pair["count"] > best["count"]:
                best = pair
    return best


def confusions_for_weak_phonemes(
    weak_phonemes: List[Any], confusion_pairs: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Return the main confusion for each weak phoneme, if any."""
    result = []
    for weak in weak_phonemes:
        phoneme = weak["phoneme"] if isinstance(weak, dict) else weak
        confusion = main_confusion_for(phoneme, confusion_pairs)
        if confusion is not None:
            result.append(confusion)
    return result


# -----------------------------------------------------------------------------
# Exercise type by mastery (#11)
# -----------------------------------------------------------------------------
def exercise_type_for_mastery(mastery_value: Optional[float], is_unknown: bool = False) -> str:
    """Pick the practice format appropriate to a phoneme's mastery.

    unknown        -> diagnostic sentence
    < 0.40         -> isolated words / minimal pairs
    0.40 - 0.70    -> short phrases
    0.70 - 0.85    -> natural targeted sentences
    > 0.85         -> maintenance / connected speech
    """
    if is_unknown or mastery_value is None:
        return "diagnostic"
    if mastery_value < 0.40:
        return "minimal_pairs"
    if mastery_value < 0.70:
        return "short_phrase"
    if mastery_value <= 0.85:
        return "targeted_sentence"
    return "maintenance"
