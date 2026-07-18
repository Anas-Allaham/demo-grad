"""Tests 13 & 18: audio-quality gate and PanPhon-unavailable safety."""

from datetime import datetime, timezone

import numpy as np

import mastery
from audio_quality import analyze_audio_quality, should_update_mastery
from phoneme_vectors_professional import panphon_available, scoring_engine

SR = 16000
FIXED_NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _noise(seconds, level=0.3, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, level, int(seconds * SR)).astype(np.float32)


# 13. Non-scorable audio does not update mastery.
def test_non_scorable_audio_blocks_mastery_update():
    silent = np.zeros(SR, dtype=np.float32)
    decision = analyze_audio_quality(silent, SR)
    assert decision.scorable is False

    # The gate: an unscorable recording never updates mastery, even if the
    # engine is trusted.
    assert should_update_mastery(decision.scorable, scoring_trusted=True) is False

    # And structurally: the app produces NO alignment for an unscorable
    # recording, so even an update call would be a no-op.
    before = {"θ": mastery.PhonemeStat(alpha=3.0, beta=1.0, independent_attempts=2)}
    after = mastery.update_mastery_for_recording(before, [], now=FIXED_NOW)
    assert after["θ"].independent_attempts == 2
    assert after["θ"].alpha == 3.0


def test_scorable_audio_passes_gate():
    good = np.concatenate([np.zeros(int(0.2 * SR), dtype=np.float32),
                           _noise(1.6), np.zeros(int(0.2 * SR), dtype=np.float32)])
    decision = analyze_audio_quality(good, SR)
    assert decision.scorable is True
    assert should_update_mastery(decision.scorable, scoring_trusted=True) is True


# 18. PanPhon unavailable must not silently produce trusted scores.
def test_panphon_unavailable_is_not_trusted():
    # Engine label is consistent with availability.
    if panphon_available():
        assert scoring_engine() == "panphon"
    else:
        assert scoring_engine() == "fallback_features"

    # The mastery gate treats an untrusted engine as non-updating regardless of
    # audio quality -- so fallback scores can never silently update mastery.
    assert should_update_mastery(scorable=True, scoring_trusted=False) is False
    trusted = scoring_engine() == "panphon"
    assert should_update_mastery(scorable=True, scoring_trusted=trusted) is trusted
