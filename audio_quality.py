"""
Audio-quality diagnostics and mastery-evidence gate.

Turns a recording into an ``AudioQualityDecision``:

    {
        "scorable": bool,
        "quality_weight": float in [0, 1],
        "reasons": [str, ...],
        "metrics": {...},
    }

If ``scorable`` is False, the caller may still enhance, transcribe, align, and
show a score. It must not use that uncertain score to update phoneme mastery.
This keeps recording acceptance separate from evidence quality.

Design choices that matter linguistically:
  * Leading/trailing silence never invalidates a recording (people breathe
    before/after speaking).
  * A short silence is NOT a dropout. Natural pauses and stop closures are
    normal speech; only sustained *internal* gaps count.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# ---- Configurable thresholds (documented, not magic numbers) ----------------
MIN_DURATION_S = 0.35          # shorter than this can't hold a real utterance
MAX_DURATION_S = 40.0          # implausibly long for a practice sentence
MIN_RMS = 0.006                # below this the mic essentially captured nothing
CLIPPING_LEVEL = 0.98          # |sample| at/above this counts as clipped
MAX_CLIPPING_RATIO = 0.02      # >2% clipped samples = distorted
MIN_VOICED_RATIO = 0.12        # need at least this fraction of voiced frames
MIN_SPEECH_SPAN_S = 0.25       # trimmed speech must span at least this long
DROPOUT_MIN_S = 0.30           # an internal silence must exceed this to count
MAX_INTERNAL_SILENCE_RATIO = 0.55  # too much silence *inside* speech = dropout

FRAME_S = 0.020                # 20 ms analysis window
HOP_S = 0.010                  # 10 ms hop
SILENCE_REL_TO_PEAK = 0.08     # frame is "silent" below 8% of peak frame RMS
SILENCE_ABS_FLOOR = 1e-4

# ---- Speech-presence thresholds ---------------------------------------------
# Energy alone cannot tell speech from white noise, a sine tone, or a DC
# offset -- all can be "loud". These spectral/temporal features do, validated
# against a real speech fixture vs synthetic noise/tone/DC/clipping:
#   * envelope modulation  - real speech has strong syllable-rate amplitude
#     modulation (~0.5-0.7); steady signals (noise/tone/DC/clipping) are ~0.0.
#   * spectral flatness    - white noise ~1.0; pure tone/DC ~0.0; speech ~0.1-0.4.
#   * spectral bandwidth    - a pure tone/DC is near-zero bandwidth; speech is wide.
#   * zero-crossing rate    - a DC/sub-sonic signal barely crosses zero.
SPEECH_MODULATION_MIN = 0.20   # below this = no speech-like modulation
SPEECH_FLATNESS_MAX = 0.55     # above this = broadband noise, not speech
SPEECH_FLATNESS_MIN = 0.012    # below this = pure tone / DC, not speech
SPEECH_BANDWIDTH_MIN = 0.02    # normalized to Nyquist; below = tonal, not speech
SPEECH_ZCR_MIN = 0.004         # below = DC / sub-sonic, not speech
SPEECH_FRAME_S = 0.025         # 25 ms window for spectral analysis
HIGHPASS_HZ = 60.0             # remove DC + mains hum before speech analysis


@dataclass
class AudioQualityDecision:
    scorable: bool
    quality_weight: float
    reasons: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scorable": self.scorable,
            "quality_weight": round(self.quality_weight, 3),
            "reasons": self.reasons,
            "metrics": self.metrics,
        }


def _remove_dc_and_rumble(audio: np.ndarray, sr: int) -> np.ndarray:
    """Remove the DC offset and sub-~60 Hz rumble/mains hum before speech
    analysis. Real speech is AC energy above ~60 Hz; a DC bias or low-frequency
    hum (both common with real mics / browser capture) otherwise dumps energy at
    0 Hz and makes genuine speech look tonal / sub-sonic to the gate."""
    audio = audio - float(np.mean(audio))  # remove DC offset
    try:
        from scipy.signal import butter, sosfiltfilt
        nyquist = sr / 2.0
        cutoff = min(HIGHPASS_HZ, nyquist * 0.9) / nyquist
        if 0 < cutoff < 1 and len(audio) > 27:
            sos = butter(2, cutoff, btype="highpass", output="sos")
            audio = sosfiltfilt(sos, audio).astype(np.float64)
    except Exception:
        pass  # DC removal alone still helps if scipy is unavailable
    return audio


def _speech_presence_features(audio: np.ndarray, sr: int) -> Optional[Dict[str, float]]:
    """Spectral/temporal features that distinguish real speech from white
    noise, a sine tone, a DC offset, or clipping. Returns None if the signal
    is too short to analyze."""
    win = max(8, int(SPEECH_FRAME_S * sr))
    hop = max(1, int(HOP_S * sr))
    if len(audio) < win:
        return None

    n_frames = 1 + (len(audio) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n_frames)[:, None]
    window = np.hanning(win)[None, :]
    frames = audio[idx] * window

    frame_energy = np.sqrt(np.mean(frames ** 2, axis=1))
    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    avg_power = power.mean(axis=0) + 1e-12

    # Spectral flatness: geometric mean / arithmetic mean of the power spectrum.
    flatness = float(np.exp(np.mean(np.log(avg_power))) / np.mean(avg_power))

    freqs = np.fft.rfftfreq(win, 1.0 / sr)
    nyquist = sr / 2.0
    centroid = float(np.sum(freqs * avg_power) / np.sum(avg_power))
    bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * avg_power) / np.sum(avg_power)))

    zcr = float(np.mean(np.abs(np.diff(np.sign(audio))) > 0)) if len(audio) > 1 else 0.0

    # Envelope modulation: coefficient of variation of voiced-frame energy.
    voiced = frame_energy > max(float(frame_energy.max()) * SILENCE_REL_TO_PEAK, SILENCE_ABS_FLOOR)
    voiced_energy = frame_energy[voiced] if voiced.any() else frame_energy
    modulation = float(np.std(voiced_energy) / (np.mean(voiced_energy) + 1e-9))

    return {
        "spectral_flatness": round(flatness, 4),
        "spectral_centroid": round(centroid / nyquist, 4),
        "spectral_bandwidth": round(bandwidth / nyquist, 4),
        "zero_crossing_rate": round(zcr, 4),
        "envelope_modulation": round(modulation, 4),
    }


def _frame_rms(audio: np.ndarray, sr: int) -> np.ndarray:
    win = max(1, int(FRAME_S * sr))
    hop = max(1, int(HOP_S * sr))
    if len(audio) < win:
        return np.array([float(np.sqrt(np.mean(audio ** 2)))]) if len(audio) else np.array([0.0])
    frames = []
    for start in range(0, len(audio) - win + 1, hop):
        seg = audio[start:start + win]
        frames.append(float(np.sqrt(np.mean(seg ** 2))))
    return np.asarray(frames, dtype=np.float64)


def analyze_audio_quality(audio: np.ndarray, sr: int) -> AudioQualityDecision:
    """Score a mono waveform. Returns an AudioQualityDecision."""
    reasons: List[str] = []
    audio = np.asarray(audio, dtype=np.float64).ravel()

    # ---- Empty / non-finite ----
    if audio.size == 0:
        return AudioQualityDecision(False, 0.0, ["empty_audio"], {"num_samples": 0})
    if not np.all(np.isfinite(audio)):
        finite_ratio = float(np.mean(np.isfinite(audio)))
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        reasons.append("non_finite_samples")
    else:
        finite_ratio = 1.0

    duration = len(audio) / float(sr)
    # Clipping is judged on the RAW samples (it is about hitting the rails).
    peak = float(np.max(np.abs(audio)))
    clipping_ratio = float(np.mean(np.abs(audio) >= CLIPPING_LEVEL))
    dc_offset = float(np.mean(audio))

    # Everything speech-related is judged on the DC/rumble-removed signal so a
    # mic DC bias or mains hum can't make real speech look tonal/sub-sonic.
    audio = _remove_dc_and_rumble(audio, sr)
    rms = float(np.sqrt(np.mean(audio ** 2)))

    # ---- Frame-level voiced analysis ----
    frame_rms = _frame_rms(audio, sr)
    peak_frame = float(np.max(frame_rms)) if frame_rms.size else 0.0
    silence_threshold = max(peak_frame * SILENCE_REL_TO_PEAK, SILENCE_ABS_FLOOR)
    voiced_mask = frame_rms > silence_threshold
    voiced_ratio = float(np.mean(voiced_mask)) if voiced_mask.size else 0.0

    # Trim leading/trailing silence -> speech span (never penalized).
    voiced_idx = np.where(voiced_mask)[0]
    internal_silence_ratio = 0.0
    max_internal_gap_s = 0.0
    speech_span_s = 0.0
    if voiced_idx.size:
        first, last = voiced_idx[0], voiced_idx[-1]
        span = voiced_mask[first:last + 1]
        speech_span_s = len(span) * HOP_S
        internal_silence_ratio = float(np.mean(~span)) if span.size else 0.0
        # Longest run of internal silence.
        max_run = run = 0
        for v in span:
            run = 0 if v else run + 1
            max_run = max(max_run, run)
        max_internal_gap_s = max_run * HOP_S

    # ---- Speech-presence analysis (rejects loud non-speech) ----
    speech = _speech_presence_features(audio, sr)

    metrics = {
        "duration_seconds": round(duration, 3),
        "peak_amplitude": round(peak, 6),
        "overall_rms": round(rms, 6),
        "clipping_ratio": round(clipping_ratio, 4),
        "voiced_ratio": round(voiced_ratio, 3),
        "speech_span_seconds": round(speech_span_s, 3),
        "internal_silence_ratio": round(internal_silence_ratio, 3),
        "max_internal_gap_seconds": round(max_internal_gap_s, 3),
        "finite_ratio": round(finite_ratio, 4),
        "dc_offset": round(dc_offset, 5),
    }
    if speech is not None:
        metrics.update(speech)

    # ---- Hard gates ----
    if duration < MIN_DURATION_S:
        reasons.append("too_short")
    if duration > MAX_DURATION_S:
        reasons.append("too_long")
    if rms < MIN_RMS:
        reasons.append("very_low_level")
    if peak <= SILENCE_ABS_FLOOR:
        reasons.append("silent")
    if clipping_ratio > MAX_CLIPPING_RATIO:
        reasons.append("clipping")
    if voiced_ratio < MIN_VOICED_RATIO:
        reasons.append("insufficient_voiced_speech")
    if speech_span_s < MIN_SPEECH_SPAN_S:
        reasons.append("insufficient_speech_span")
    if max_internal_gap_s > DROPOUT_MIN_S and internal_silence_ratio > MAX_INTERNAL_SILENCE_RATIO:
        reasons.append("excessive_internal_dropout")

    # ---- Speech-presence gates (energy alone is not speech) ----
    # Each discriminator is independent. In particular, amplitude modulation
    # can make noise or a sine tone look speech-like in the envelope domain;
    # it must not disable the spectral/tonality gates.
    if speech is not None:
        if speech["envelope_modulation"] < SPEECH_MODULATION_MIN:
            reasons.append("no_speech_modulation")
        if speech["spectral_flatness"] > SPEECH_FLATNESS_MAX:
            reasons.append("noise_like_spectrum")
        if (
            speech["spectral_flatness"] < SPEECH_FLATNESS_MIN
            or speech["spectral_bandwidth"] < SPEECH_BANDWIDTH_MIN
        ):
            reasons.append("tonal_not_speech")
        if speech["zero_crossing_rate"] < SPEECH_ZCR_MIN:
            reasons.append("dc_or_subsonic")

    # Fatal reasons make the recording unscorable outright.
    fatal = {
        "empty_audio", "silent", "too_short", "too_long", "very_low_level",
        "clipping",
        "insufficient_voiced_speech", "insufficient_speech_span",
        "excessive_internal_dropout",
        "no_speech_modulation", "noise_like_spectrum", "tonal_not_speech",
        "dc_or_subsonic",
    }
    scorable = not (set(reasons) & fatal)

    # ---- Soft quality weight (only meaningful when scorable) ----
    weight = 1.0
    if clipping_ratio > 0:
        weight -= min(0.5, clipping_ratio * 10.0)
    if voiced_ratio < 0.35:
        weight -= (0.35 - voiced_ratio)
    if "non_finite_samples" in reasons:
        weight -= 0.2
    if internal_silence_ratio > 0.35:
        weight -= min(0.3, internal_silence_ratio - 0.35)
    weight = max(0.0, min(1.0, weight))
    if not scorable:
        weight = 0.0

    return AudioQualityDecision(scorable, weight, reasons, metrics)


def should_update_mastery(scorable: bool, scoring_trusted: bool) -> bool:
    """The mastery-update gate. Trusted mastery is only updated when BOTH the
    audio was scorable AND the scoring engine is trusted (PanPhon-backed).
    A pure function so the gate is unit-testable without the Flask app."""
    return bool(scorable and scoring_trusted)


def check_audio_file(path: str | Path, sr: int = 16000) -> AudioQualityDecision:
    """Load a WAV/audio file and score it. Kept import-light: soundfile is
    imported lazily so importing this module never requires it."""
    try:
        import soundfile as sf
        audio, file_sr = sf.read(str(path))
        sr = int(file_sr)
    except Exception:
        try:
            import librosa
            audio, sr = librosa.load(str(path), sr=sr, mono=True)
        except Exception as exc:
            return AudioQualityDecision(False, 0.0, ["unreadable_audio"], {"error": repr(exc)})
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    return analyze_audio_quality(audio, sr)
