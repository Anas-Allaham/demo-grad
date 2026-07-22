"""Tests 13 & 18: audio-quality evidence and PanPhon-unavailable safety."""

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

import mastery
from audio_quality import analyze_audio_quality, should_update_mastery
from phoneme_vectors_professional import panphon_available, scoring_engine

SR = 16000
FIXED_NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)
SPEECH_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "speech_sample.wav"


def _load_speech_fixture():
    audio, sr = sf.read(str(SPEECH_FIXTURE))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return np.asarray(audio, dtype=np.float64), int(sr)


# 13. A quality warning does not update mastery.
def test_low_quality_audio_blocks_mastery_update_only():
    silent = np.zeros(SR, dtype=np.float32)
    decision = analyze_audio_quality(silent, SR)
    assert decision.scorable is False

    # The app may still display a score, but uncertain audio never updates
    # mastery even if the scoring engine is trusted.
    assert should_update_mastery(decision.scorable, scoring_trusted=True) is False

    # An empty alignment remains a no-op independently of the HTTP behavior.
    before = {"θ": mastery.PhonemeStat(alpha=3.0, beta=1.0, independent_attempts=2)}
    after = mastery.update_mastery_for_recording(before, [], now=FIXED_NOW)
    assert after["θ"].independent_attempts == 2
    assert after["θ"].alpha == 3.0


def test_real_speech_fixture_passes_gate():
    audio, sr = _load_speech_fixture()
    decision = analyze_audio_quality(audio, sr)
    assert decision.scorable is True
    assert decision.reasons == []
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
