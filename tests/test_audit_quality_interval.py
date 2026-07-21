"""Audit #4 & #5: fractional quality weight, and a real MC credible interval."""

from datetime import datetime, timezone

import assessment as A
import mastery

FIXED_NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


# ---- #4: quality_weight as fractional Beta evidence, NOT a failure ----
def test_quality_weight_scales_evidence_without_failure_bias():
    rows = [{"expected": "s", "spoken": "s", "result": "correct", "articulatory_distance": 0.0}]
    full = mastery.update_mastery_for_recording({}, rows, FIXED_NOW, quality_weight=1.0)["s"]
    half = mastery.update_mastery_for_recording({}, rows, FIXED_NOW, quality_weight=0.5)["s"]
    # Lower quality adds LESS total evidence...
    assert (full.alpha + full.beta) > (half.alpha + half.beta)
    # ...and it is applied only to alpha (a success), never invented as a miss.
    assert full.beta == half.beta == 1.0
    assert half.alpha < full.alpha


def test_low_quality_partial_credit_preserves_observation_mean():
    rows = [{"expected": "s", "spoken": "z", "result": "close_substitution", "articulatory_distance": 0.2}]
    st = mastery.update_mastery_for_recording({}, rows, FIXED_NOW, quality_weight=0.3)["s"]
    added_alpha = st.alpha - 1.0
    added_beta = st.beta - 1.0
    # Evidence direction is the observation (0.8), scaled by weight 0.3 -> 0.24 / 0.06.
    assert abs(added_alpha / (added_alpha + added_beta) - 0.8) < 1e-9


# ---- #5: real posterior credible interval (deterministic Monte Carlo) ----
def test_credible_interval_is_real_and_deterministic():
    ab = [(6.0, 2.0), (4.0, 3.0), (8.0, 1.0)]
    m1, lo1, hi1 = A._posterior_macro_interval(ab)
    m2, lo2, hi2 = A._posterior_macro_interval(ab)
    assert (m1, lo1, hi1) == (m2, lo2, hi2)          # deterministic (fixed seed)
    assert lo1 < m1 < hi1                            # a genuine interval, not a point
    # Interval brackets the analytic macro-average of the means.
    analytic = sum(a / (a + b) for a, b in ab) / len(ab)
    assert lo1 <= analytic <= hi1


def test_assessment_reports_credible_interval_not_confidence():
    stats, context = {}, {}
    for ph in ["s", "z", "t", "d", "n", "m", "k", "p", "f", "l", "ɹ", "i"]:
        stats[ph] = mastery.PhonemeStat(alpha=5.0, beta=2.0, independent_attempts=4,
                                        occurrence_count=6, last_practiced_at=FIXED_NOW)
        context[ph] = {"recordings": 4, "distinct_prompts": 2, "occurrences": 6}
    result = A.assess_user_level(stats, context, independent_recording_count=8, now=FIXED_NOW)
    assert "credible_interval" in result
    assert "confidence_interval" not in result
    assert result["interval_method"] == "quality_weighted_beta_posterior_monte_carlo"
    lo, hi = result["credible_interval"]
    assert 0.0 <= lo <= result["pronunciation_score"] <= hi <= 100.0
