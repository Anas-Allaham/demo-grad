"""
Evidence-aware pronunciation level assessment and confusion analysis.

Two responsibilities:

  1. ``assess_user_level`` - a level assessment that refuses to invent a level
     from thin evidence. A phoneme is level-eligible only after enough
     independent, quality-weighted recordings across enough distinct prompts;
     the score is a posterior macro-average and the level is assigned only
     when its credible interval stays inside one band. Status is reported honestly
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
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import mastery
from phoneme_vectors_professional import (
    ASSESSABLE_INVENTORY,
    canonicalize_phoneme,
    is_assessable,
)

# ---- Posterior credible interval (Monte Carlo) ------------------------------
# The macro-average pronunciation score is a function of several independent
# Beta posteriors, so its own distribution has no closed form. We estimate it by
# Monte Carlo: draw from each phoneme's (decayed) Beta posterior, average across
# phonemes per draw, and take percentiles of that macro-average distribution.
# Deterministic given the seed, so tests are reproducible.
MC_SAMPLES = 4000
MC_SEED = 20260718
CREDIBLE_MASS = 0.95


def _posterior_macro_interval(
    alpha_betas: Sequence[Tuple[float, float]],
    seed: int = MC_SEED,
    n_samples: int = MC_SAMPLES,
    credible_mass: float = CREDIBLE_MASS,
) -> Tuple[float, float, float]:
    """Monte-Carlo posterior mean and credible interval of the macro-average
    over independent Beta posteriors. Returns (mean, low, high) in [0, 1]."""
    if not alpha_betas:
        return (0.0, 0.0, 0.0)
    rng = np.random.default_rng(seed)
    draws = np.stack([rng.beta(max(a, 1e-6), max(b, 1e-6), size=n_samples) for a, b in alpha_betas])
    macro = draws.mean(axis=0)
    tail = (1.0 - credible_mass) / 2.0
    return (float(macro.mean()), float(np.quantile(macro, tail)), float(np.quantile(macro, 1.0 - tail)))

# ---- Eligibility (configurable, provisional) --------------------------------
MIN_RECORDINGS_FOR_ELIGIBLE = 3      # independent recordings before a phoneme counts
MIN_EFFECTIVE_RECORDINGS_FOR_ELIGIBLE = 2.0  # quality-weighted recording mass
MIN_DISTINCT_PROMPTS = 2             # across at least this many different texts

# Utterance-level epenthesis posterior. Its influence ramps up over the first
# three effective recordings and is capped so phoneme mastery remains the main
# profile signal while systematic insertions can still lower the result.
UTTERANCE_PROFILE_MAX_WEIGHT = 0.20
UTTERANCE_FULL_WEIGHT_EVIDENCE = 3.0

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
        recordings = ctx.get("recordings", 0)
        effective = ctx.get("effective_recordings", recordings)
        if (
            is_assessable(canon)
            and recordings >= DIAGNOSTIC_MIN_CONTEXTS
            and effective >= DIAGNOSTIC_MIN_CONTEXTS
        ):
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


def _level_from_interval(interval: Optional[Sequence[float]]) -> Tuple[str, List[str]]:
    """Conservative level decision from a posterior credible interval."""
    if interval is None:
        return "unknown", []
    low, high = float(interval[0]), float(interval[1])
    if high < BEGINNER_MAX_SCORE:
        return "beginner", []
    if low >= INTERMEDIATE_MAX_SCORE:
        return "advanced", []
    if low >= BEGINNER_MAX_SCORE and high < INTERMEDIATE_MAX_SCORE:
        return "intermediate", []

    possible: List[str] = []
    if low < BEGINNER_MAX_SCORE:
        possible.append("beginner")
    if high >= BEGINNER_MAX_SCORE and low < INTERMEDIATE_MAX_SCORE:
        possible.append("intermediate")
    if high >= INTERMEDIATE_MAX_SCORE:
        possible.append("advanced")
    return "uncertain", possible


def _posterior_profile_interval(
    alpha_betas: Sequence[Tuple[float, float]],
    utterance_state: Optional[Dict[str, Any]] = None,
    seed: int = MC_SEED,
    n_samples: int = MC_SAMPLES,
    credible_mass: float = CREDIBLE_MASS,
) -> Tuple[float, float, float, float]:
    """Posterior interval for phoneme macro mastery plus epenthesis state.

    Returns ``(mean, low, high, utterance_weight)`` in [0, 1].
    """
    if not alpha_betas:
        return (0.0, 0.0, 0.0, 0.0)
    rng = np.random.default_rng(seed)
    phoneme_draws = np.stack([
        rng.beta(max(alpha, 1e-6), max(beta, 1e-6), size=n_samples)
        for alpha, beta in alpha_betas
    ]).mean(axis=0)

    effective = float((utterance_state or {}).get("effective_recordings", 0.0))
    utterance_weight = UTTERANCE_PROFILE_MAX_WEIGHT * min(
        1.0, effective / UTTERANCE_FULL_WEIGHT_EVIDENCE
    )
    profile_draws = phoneme_draws
    if utterance_weight > 0:
        alpha = max(float(utterance_state.get("alpha", 1.0)), 1e-6)
        beta = max(float(utterance_state.get("beta", 1.0)), 1e-6)
        utterance_draws = rng.beta(alpha, beta, size=n_samples)
        profile_draws = (1.0 - utterance_weight) * phoneme_draws + utterance_weight * utterance_draws

    tail = (1.0 - credible_mass) / 2.0
    return (
        float(profile_draws.mean()),
        float(np.quantile(profile_draws, tail)),
        float(np.quantile(profile_draws, 1.0 - tail)),
        float(utterance_weight),
    )


def assess_user_level(
    stats: Dict[str, "mastery.PhonemeStat"],
    context_stats: Dict[str, Dict[str, Any]],
    independent_recording_count: int,
    now: Optional[datetime] = None,
    effective_recording_count: Optional[float] = None,
    utterance_state: Optional[Dict[str, Any]] = None,
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

    context: Dict[str, Dict[str, Any]] = {}
    for phoneme, ctx in context_stats.items():
        canon = canonicalize_phoneme(phoneme)
        if not is_assessable(canon):
            continue
        acc = context.setdefault(
            canon,
            {"recordings": 0, "effective_recordings": 0.0, "distinct_prompts": 0, "occurrences": 0},
        )
        acc["recordings"] += ctx.get("recordings", 0)
        acc["effective_recordings"] += ctx.get("effective_recordings", ctx.get("recordings", 0))
        acc["distinct_prompts"] = max(acc["distinct_prompts"], ctx.get("distinct_prompts", 0))
        acc["occurrences"] += ctx.get("occurrences", 0)

    eligible: List[str] = []
    for phoneme, stat in tracked.items():
        ctx = context.get(phoneme, {})
        recordings = max(ctx.get("recordings", 0), stat.independent_attempts)
        effective_recordings = float(ctx.get("effective_recordings", recordings))
        prompts = ctx.get("distinct_prompts", 0)
        if (
            recordings >= MIN_RECORDINGS_FOR_ELIGIBLE
            and effective_recordings >= MIN_EFFECTIVE_RECORDINGS_FOR_ELIGIBLE
            and prompts >= MIN_DISTINCT_PROMPTS
        ):
            eligible.append(phoneme)

    weak: List[Dict[str, Any]] = []
    strong: List[str] = []
    eligible_alpha_betas: List[Tuple[float, float]] = []

    for phoneme in eligible:
        stat = tracked[phoneme]
        decayed = mastery.decayed_stat(stat, now)
        eligible_alpha_betas.append((decayed.alpha, decayed.beta))
        lcb = mastery.lower_confidence_bound(stat, now=now)   # conservative
        mean = mastery.posterior_mean(stat, now=now)
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

    if eligible_alpha_betas and status != "insufficient_evidence":
        # Real posterior of the macro-average plus the separate utterance-level
        # epenthesis state. Both use quality-weighted Beta evidence.
        mc_mean, mc_low, mc_high, utterance_weight = _posterior_profile_interval(
            eligible_alpha_betas, utterance_state=utterance_state
        )
        pronunciation_score: Optional[float] = round(100.0 * mc_mean, 1)
        credible_interval: Optional[List[float]] = [round(100.0 * mc_low, 1), round(100.0 * mc_high, 1)]
    else:
        pronunciation_score = None
        credible_interval = None
        utterance_weight = 0.0

    overall_level, borderline_levels = _level_from_interval(credible_interval)
    exercise_level = _level_from_score(credible_interval[0] if credible_interval else None)
    utterance_alpha = float((utterance_state or {}).get("alpha", 1.0))
    utterance_beta = float((utterance_state or {}).get("beta", 1.0))
    utterance_mean = utterance_alpha / max(utterance_alpha + utterance_beta, 1e-9)

    return {
        "pronunciation_score": pronunciation_score,
        "overall_level": overall_level,
        "borderline_levels": borderline_levels,
        "level_decision": "borderline" if overall_level == "uncertain" else overall_level,
        "exercise_level": exercise_level,
        "assessment_status": status,
        "inventory_coverage": round(inventory_coverage, 3),
        "eligible_phoneme_count": eligible_count,
        "tracked_phoneme_count": len(tracked),
        "independent_recording_count": independent_recording_count,
        "effective_recording_count": round(
            float(effective_recording_count if effective_recording_count is not None else independent_recording_count),
            3,
        ),
        # A real Bayesian posterior credible interval for the macro-average
        # (Monte Carlo over the eligible Beta posteriors). Named accurately --
        # it is NOT a frequentist confidence interval.
        "credible_interval": credible_interval,
        "credible_mass": CREDIBLE_MASS,
        "interval_method": "quality_weighted_beta_posterior_monte_carlo",
        "level_decision_basis": "credible_interval",
        "utterance_epenthesis_state": {
            "alpha": round(utterance_alpha, 6),
            "beta": round(utterance_beta, 6),
            "posterior_mean": round(utterance_mean, 3),
            "effective_recordings": round(float((utterance_state or {}).get("effective_recordings", 0.0)), 3),
            "insertion_count": int((utterance_state or {}).get("insertion_count", 0)),
            "profile_weight": round(utterance_weight, 3),
        },
        "weak_phonemes": weak,
        "unknown_phonemes": unknown,
        "strong_phonemes": sorted(strong),
        "provisional": True,
        "assessment_source": "saved_profile_evidence",
        "raw_stateless_metrics": False,
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
