"""Audit #1: the audio gate rejects loud non-speech, accepts real speech."""

from pathlib import Path

import numpy as np
import soundfile as sf

from audio_quality import analyze_audio_quality

SR = 16000
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "speech_sample.wav"


def _rejected(x, sr=SR):
    return analyze_audio_quality(np.asarray(x, dtype=np.float64), sr)


def test_white_noise_rejected():
    rng = np.random.default_rng(0)
    d = _rejected(rng.normal(0, 0.3, 2 * SR))
    assert d.scorable is False
    assert "noise_like_spectrum" in d.reasons or "no_speech_modulation" in d.reasons


def test_sine_tone_rejected():
    t = np.arange(2 * SR) / SR
    d = _rejected(0.5 * np.sin(2 * np.pi * 440 * t))
    assert d.scorable is False
    assert "tonal_not_speech" in d.reasons


def test_dc_signal_rejected():
    d = _rejected(0.5 * np.ones(2 * SR))
    assert d.scorable is False
    assert "dc_or_subsonic" in d.reasons or "tonal_not_speech" in d.reasons


def test_silence_rejected():
    d = _rejected(np.zeros(2 * SR))
    assert d.scorable is False


def test_clipping_rejected():
    t = np.arange(2 * SR) / SR
    d = _rejected(np.clip(3.0 * np.sin(2 * np.pi * 300 * t), -1, 1))
    assert d.scorable is False
    assert "clipping" in d.reasons


def test_real_speech_accepted():
    audio, sr = sf.read(str(FIXTURE))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    d = analyze_audio_quality(np.asarray(audio, dtype=np.float64), int(sr))
    assert d.scorable is True
    assert d.reasons == []
    # And it exposes the speech-presence features it judged on.
    assert d.metrics["envelope_modulation"] > 0.2
    assert d.metrics["spectral_flatness"] < 0.55


def _speech():
    audio, sr = sf.read(str(FIXTURE))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return np.asarray(audio, dtype=np.float64), int(sr)


def test_real_speech_with_dc_offset_still_accepted():
    """Regression: a DC-biased mic must NOT cause a false 'tonal_not_speech'
    rejection of genuine speech (the DC offset is removed before analysis)."""
    audio, sr = _speech()
    d = analyze_audio_quality(audio + 0.15, sr)
    assert d.scorable is True
    assert "tonal_not_speech" not in d.reasons


def test_real_speech_with_mains_hum_still_accepted():
    """Regression: sub-60 Hz mains hum must not falsely reject real speech."""
    audio, sr = _speech()
    hum = 0.25 * np.sin(2 * np.pi * 60 * np.arange(len(audio)) / sr)
    d = analyze_audio_quality(audio + hum, sr)
    assert d.scorable is True
