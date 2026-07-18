"""
Per-phoneme mastery model: a Beta-Bernoulli posterior per (user, phoneme)
with time-decay, updated incrementally from each `/analyze` attempt's
alignment result.

Design (see PLAN for the full rationale):
- One conjugate Beta(alpha, beta) posterior per phoneme, prior (1, 1).
- Each observed phoneme (correct / not) is one Bernoulli trial.
- Before folding in a new observation, existing (alpha, beta) are decayed
  back toward the prior based on time elapsed since last practiced. This
  both models forgetting and widens the credible interval for stale
  phonemes, so they naturally resurface for re-testing.
- Phonemes are ranked for practice by the *lower* confidence bound of the
  posterior (pessimistic estimate), not the raw mean, so a single unlucky
  attempt on a barely-practiced phoneme doesn't outrank a phoneme that is
  consistently wrong across many attempts.

This module is intentionally pure/DB-free so it can be unit-tested against
hand-built alignment lists with no audio, G2P, or database involved.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

from phoneme_vectors import canonicalize_phoneme

try:
    from scipy.stats import beta as _beta_dist
except Exception:
    _beta_dist = None

PRIOR_ALPHA = 1.0
PRIOR_BETA = 1.0
DECAY_HALF_LIFE_DAYS = 25.0
LCB_QUANTILE = 0.10
WEAK_TOP_K = 2  # focus each exercise on the 2 weakest sounds
MAINTENANCE_EPSILON = 0.15
MASTERED_THRESHOLD = 0.80


@dataclass
class PhonemeStat:
    alpha: float = PRIOR_ALPHA
    beta: float = PRIOR_BETA
    attempts_count: int = 0
    last_practiced_at: Optional[datetime] = None


def score_from_operation(operation: str) -> float:
    """Binary evidence: 1.0 only for an exact match. Substitutions of any
    severity and deletions all count as a miss for this phoneme."""
    return 1.0 if operation == "correct" else 0.0


def _decay(stat: PhonemeStat, now: datetime) -> PhonemeStat:
    if stat.last_practiced_at is None:
        return stat
    elapsed_days = max(0.0, (now - stat.last_practiced_at).total_seconds() / 86400.0)
    gamma = math.exp(-elapsed_days / DECAY_HALF_LIFE_DAYS)
    return PhonemeStat(
        alpha=PRIOR_ALPHA + gamma * (stat.alpha - PRIOR_ALPHA),
        beta=PRIOR_BETA + gamma * (stat.beta - PRIOR_BETA),
        attempts_count=stat.attempts_count,
        last_practiced_at=stat.last_practiced_at,
    )


def apply_observation(stat: PhonemeStat, success: float, now: datetime) -> PhonemeStat:
    decayed = _decay(stat, now)
    return PhonemeStat(
        alpha=decayed.alpha + success,
        beta=decayed.beta + (1.0 - success),
        attempts_count=decayed.attempts_count + 1,
        last_practiced_at=now,
    )


def posterior_mean(stat: PhonemeStat) -> float:
    return stat.alpha / (stat.alpha + stat.beta)


def lower_confidence_bound(stat: PhonemeStat, quantile: float = LCB_QUANTILE) -> float:
    """Pessimistic mastery estimate used for ranking. Widens (drops) for
    low-evidence or stale phonemes, which is what makes this ranking do
    double duty as both a weak-spot detector and a spaced-repetition cue."""
    if _beta_dist is not None:
        try:
            return float(_beta_dist.ppf(quantile, stat.alpha, stat.beta))
        except Exception:
            pass
    # Fallback with no scipy: shrink the mean by an uncertainty margin that
    # grows as total evidence (alpha + beta) shrinks toward the prior.
    n = stat.alpha + stat.beta
    mean = posterior_mean(stat)
    margin = 1.0 / math.sqrt(max(n, 1e-6))
    return max(0.0, mean - margin)


def update_mastery_for_attempt(
    existing_stats: Dict[str, PhonemeStat],
    alignment: List[dict],
    now: datetime,
) -> Dict[str, PhonemeStat]:
    """Fold one `/analyze` attempt's alignment rows into per-phoneme stats.

    `alignment` rows are the same `{expected, spoken, result, distance}`
    dicts already returned by the `/analyze` route. Insertions (no target
    phoneme, `expected in (None, "-")`) are skipped — they still count
    toward the overall phoneme_error_rate shown elsewhere, just not toward
    any single phoneme's mastery.
    """
    updated = dict(existing_stats)
    for row in alignment:
        expected = row.get("expected")
        if not expected or expected == "-":
            continue
        phoneme = canonicalize_phoneme(expected)
        if not phoneme:
            continue
        success = score_from_operation(row.get("result", ""))
        current = updated.get(phoneme, PhonemeStat())
        updated[phoneme] = apply_observation(current, success, now)
    return updated


def rank_weak_phonemes(
    stats: Dict[str, PhonemeStat],
    top_k: int = WEAK_TOP_K,
    epsilon: float = MAINTENANCE_EPSILON,
    rng: Optional[random.Random] = None,
) -> List[str]:
    """Return up to `top_k` phonemes to target next, ranked by ascending
    LCB (weakest/least-certain first). With probability `epsilon`, swaps
    one slot for a "maintenance probe": the phoneme with the highest
    mastery that hasn't been practiced in the longest time, so mastered
    sounds occasionally get re-tested instead of only ever drilling
    weaknesses. Returns [] when `stats` is empty (cold start) so the
    caller can fall back to a diagnostic sentence instead of ranking
    nothing.
    """
    if not stats:
        return []

    scored = sorted(
        ((phoneme, lower_confidence_bound(stat)) for phoneme, stat in stats.items()),
        key=lambda pair: pair[1],
    )
    targets = [phoneme for phoneme, _ in scored[:top_k]]

    rng = rng or random
    if targets and rng.random() < epsilon:
        mastered = [
            (phoneme, stat)
            for phoneme, stat in stats.items()
            if posterior_mean(stat) > MASTERED_THRESHOLD and phoneme not in targets
        ]
        if mastered:
            mastered.sort(key=lambda pair: pair[1].last_practiced_at or datetime.min.replace(tzinfo=timezone.utc))
            targets[-1] = mastered[0][0]

    return targets


def get_overmastered_phonemes(stats: Dict[str, PhonemeStat], min_evidence: float = 6.0) -> set:
    """Phonemes the user has already demonstrated strong, well-evidenced
    mastery on — used to steer new exercises away from padding sentences
    with sounds that don't need more practice."""
    return {
        phoneme
        for phoneme, stat in stats.items()
        if posterior_mean(stat) > MASTERED_THRESHOLD and (stat.alpha + stat.beta) >= min_evidence
    }
