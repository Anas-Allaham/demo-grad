"""
Per-phoneme mastery model: a Beta posterior per (user, phoneme) updated
ONCE PER RECORDING from soft, provisional pronunciation evidence, with a
correct half-life time decay applied on read.

Key properties (see the task spec):

* Soft evidence, not binary. A close substitution is partial credit
  (``1 - articulatory_distance``), not a flat miss. The evidence function
  lives in ``phoneme_vectors_professional.mastery_observation`` and is
  deliberately isolated so it can later be replaced by calibrated GOP
  probabilities without touching the Beta machinery here.

* One Beta update per recording, not per occurrence. Three /θ/ tokens in one
  sentence contribute ONE Beta update (from their mean observation) and
  ``occurrence_count += 3``, ``independent_attempts += 1`` -- so a single
  recording can never mark a phoneme "mastered" through sheer repetition.

* Correct half-life decay. ``gamma = 0.5 ** (elapsed_days / half_life)`` so
  exactly one half-life retains 50% of the evidence, applied whenever the
  posterior is read (ranking, display, level assessment) rather than only at
  the next observation -- stale skills decay in real time.

Pure/DB-free so it unit-tests against hand-built alignment rows with fixed
datetimes, no audio/G2P/database involved.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

from phoneme_vectors_professional import (
    canonicalize_phoneme,
    in_inventory,
    mastery_observation,
)

try:
    from scipy.stats import beta as _beta_dist
except Exception:
    _beta_dist = None

PRIOR_ALPHA = 1.0
PRIOR_BETA = 1.0
DECAY_HALF_LIFE_DAYS = 25.0
LCB_QUANTILE = 0.10
WEAK_TOP_K = 2
MAINTENANCE_EPSILON = 0.15
MASTERED_THRESHOLD = 0.80
# Evidence (alpha + beta above the prior) required before "overmastered"
# steering trusts a phoneme as well-practiced.
OVERMASTERED_MIN_EVIDENCE = 6.0


@dataclass
class PhonemeStat:
    alpha: float = PRIOR_ALPHA
    beta: float = PRIOR_BETA
    independent_attempts: int = 0   # number of distinct recordings
    occurrence_count: int = 0       # number of phoneme occurrences seen
    last_practiced_at: Optional[datetime] = None


def decay_factor(elapsed_days: float, half_life_days: float = DECAY_HALF_LIFE_DAYS) -> float:
    """Fraction of evidence retained after ``elapsed_days``. At exactly one
    half-life this is 0.5."""
    if half_life_days <= 0:
        return 1.0
    return 0.5 ** (max(0.0, elapsed_days) / half_life_days)


def decayed_stat(stat: PhonemeStat, now: Optional[datetime]) -> PhonemeStat:
    """Return ``stat`` with its Beta evidence decayed back toward the prior
    based on time since it was last practiced. Counters are preserved."""
    if now is None or stat.last_practiced_at is None:
        return stat
    elapsed_days = max(0.0, (now - stat.last_practiced_at).total_seconds() / 86400.0)
    gamma = decay_factor(elapsed_days)
    return PhonemeStat(
        alpha=PRIOR_ALPHA + gamma * (stat.alpha - PRIOR_ALPHA),
        beta=PRIOR_BETA + gamma * (stat.beta - PRIOR_BETA),
        independent_attempts=stat.independent_attempts,
        occurrence_count=stat.occurrence_count,
        last_practiced_at=stat.last_practiced_at,
    )


def posterior_mean(stat: PhonemeStat, now: Optional[datetime] = None) -> float:
    """Decayed posterior mean mastery in [0, 1]."""
    s = decayed_stat(stat, now)
    return s.alpha / (s.alpha + s.beta)


def lower_confidence_bound(
    stat: PhonemeStat, quantile: float = LCB_QUANTILE, now: Optional[datetime] = None
) -> float:
    """Pessimistic (lower-tail) mastery estimate, after decay. Widens for
    low-evidence or stale phonemes, so ranking by it doubles as weak-spot
    detection and spaced-repetition."""
    s = decayed_stat(stat, now)
    if _beta_dist is not None:
        try:
            return float(_beta_dist.ppf(quantile, s.alpha, s.beta))
        except Exception:
            pass
    n = s.alpha + s.beta
    mean = s.alpha / n
    margin = 1.0 / math.sqrt(max(n, 1e-6))
    return max(0.0, mean - margin)


def _grouped_observations(alignment: List[dict]) -> Dict[str, List[float]]:
    """Group alignment rows by expected canonical phoneme -> list of soft
    observations. Insertions (no expected phoneme) are skipped."""
    grouped: Dict[str, List[float]] = defaultdict(list)
    for row in alignment:
        expected = row.get("expected")
        if not expected or expected == "-":
            continue
        phoneme = canonicalize_phoneme(expected)
        if not phoneme:
            continue
        # Support both the new explicit field and the legacy "distance".
        dist = row.get("articulatory_distance", row.get("distance"))
        obs = mastery_observation(row.get("result", ""), float(dist) if dist is not None else 1.0)
        grouped[phoneme].append(obs)
    return grouped


def update_mastery_for_recording(
    existing_stats: Dict[str, PhonemeStat],
    alignment: List[dict],
    now: datetime,
    quality_weight: float = 1.0,
) -> Dict[str, PhonemeStat]:
    """Fold ONE recording's alignment into per-phoneme stats.

    For each expected phoneme in this recording:
      1. Mean of its soft observations -> a single Bernoulli-style evidence.
      2. Decay the existing posterior to ``now``.
      3. Apply exactly one Beta update, SCALED by ``quality_weight``:
             alpha += w * mean;  beta += w * (1 - mean)
         Lower-quality audio therefore contributes LESS total evidence
         (a wider, less-certain update) WITHOUT biasing the mean toward
         failure -- alpha and beta are scaled by the same factor, so a noisy
         recording never turns into a pronunciation "miss". A perfect
         recording (w=1.0) is a full Bernoulli trial.
      4. occurrence_count += number of occurrences; independent_attempts += 1.
    """
    weight = max(0.0, min(1.0, float(quality_weight)))
    updated = dict(existing_stats)
    for phoneme, observations in _grouped_observations(alignment).items():
        if not observations:
            continue
        mean_obs = sum(observations) / len(observations)
        occurrences = len(observations)
        current = updated.get(phoneme, PhonemeStat())
        decayed = decayed_stat(current, now)
        updated[phoneme] = PhonemeStat(
            alpha=decayed.alpha + weight * mean_obs,
            beta=decayed.beta + weight * (1.0 - mean_obs),
            independent_attempts=decayed.independent_attempts + 1,
            occurrence_count=decayed.occurrence_count + occurrences,
            last_practiced_at=now,
        )
    return updated


# Backwards-compatible alias for the previous name.
update_mastery_for_attempt = update_mastery_for_recording


def rank_weak_phonemes(
    stats: Dict[str, PhonemeStat],
    top_k: int = WEAK_TOP_K,
    epsilon: float = MAINTENANCE_EPSILON,
    now: Optional[datetime] = None,
    rng: Optional[random.Random] = None,
) -> List[str]:
    """Up to ``top_k`` phonemes to target next, ranked by ascending decayed
    LCB (weakest/least-certain first). Decay is applied at ranking time so
    stale phonemes resurface without waiting for a new observation. With
    probability ``epsilon`` one slot becomes a maintenance probe (the
    highest-mastery, longest-unpracticed phoneme). Returns [] on cold start."""
    if not stats:
        return []

    # Only ever target real inventory phonemes; legacy/non-canonical keys can
    # still be displayed but must never be selected as a practice target.
    rankable = {p: s for p, s in stats.items() if in_inventory(p)}
    if not rankable:
        return []

    scored = sorted(
        ((phoneme, lower_confidence_bound(stat, now=now)) for phoneme, stat in rankable.items()),
        key=lambda pair: pair[1],
    )
    targets = [phoneme for phoneme, _ in scored[:top_k]]

    rng = rng or random
    if targets and rng.random() < epsilon:
        mastered = [
            (phoneme, stat)
            for phoneme, stat in stats.items()
            if posterior_mean(stat, now=now) > MASTERED_THRESHOLD and phoneme not in targets
        ]
        if mastered:
            mastered.sort(key=lambda pair: pair[1].last_practiced_at or datetime.min.replace(tzinfo=timezone.utc))
            targets[-1] = mastered[0][0]

    return targets


def get_overmastered_phonemes(
    stats: Dict[str, PhonemeStat],
    min_evidence: float = OVERMASTERED_MIN_EVIDENCE,
    now: Optional[datetime] = None,
) -> set:
    """Phonemes with strong, well-evidenced (decayed) mastery -- used to steer
    new exercises away from padding with sounds that need no more practice."""
    result = set()
    for phoneme, stat in stats.items():
        s = decayed_stat(stat, now)
        if posterior_mean(s) > MASTERED_THRESHOLD and (s.alpha + s.beta) >= min_evidence:
            result.add(phoneme)
    return result


def is_mastered(stat: PhonemeStat, now: Optional[datetime] = None) -> bool:
    return posterior_mean(stat, now=now) >= MASTERED_THRESHOLD
