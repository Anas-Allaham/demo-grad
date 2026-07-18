"""
Safe audio preprocessing for pronunciation analysis.
This file avoids hard noise gates and avoids deleting quiet speech.

Use it AFTER you confirm that the raw browser recording has no dropouts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt

try:
    import noisereduce as nr
except Exception:  # noisereduce is optional
    nr = None


def _to_mono_float(audio: np.ndarray) -> np.ndarray:
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32)


def bandpass_speech(audio: np.ndarray, sr: int, low_hz: float = 60.0, high_hz: float = 7600.0) -> np.ndarray:
    """Mild speech bandpass. Keeps consonants better than a 4 kHz cutoff."""
    nyquist = sr / 2
    low = max(1.0, low_hz) / nyquist
    high = min(high_hz, nyquist - 100.0) / nyquist

    if not 0 < low < high < 1:
        return audio

    sos = butter(4, [low, high], btype="bandpass", output="sos")
    return sosfiltfilt(sos, audio).astype(np.float32)


def peak_normalize(audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak <= 1e-8:
        return audio
    return (audio * min(target_peak / peak, 10.0)).astype(np.float32)


def mild_noise_reduce(audio: np.ndarray, sr: int, prop_decrease: float = 0.35) -> np.ndarray:
    """
    Mild noisereduce wrapper.

    Uses a low ``prop_decrease`` and a non-stationary estimate so quiet
    phonemes survive. The first 250 ms is passed as the noise profile
    (``y_noise``) rather than discarded, which is the intended behaviour.
    """
    if nr is None:
        return audio

    duration = len(audio) / sr
    if duration < 4.0:
        # Short clips often do not contain a reliable noise profile.
        return audio

    # Use only the first 250 ms as a possible noise profile. If the user
    # starts speaking immediately this is imperfect, so keep prop_decrease mild.
    noise_clip = audio[: int(0.25 * sr)]
    return nr.reduce_noise(
        y=audio,
        sr=sr,
        y_noise=noise_clip,
        stationary=False,
        prop_decrease=prop_decrease,
    ).astype(np.float32)


def process_audio_file(
    input_path: str | Path,
    output_path: str | Path,
    use_noise_reduce: bool = False,
) -> str:
    input_path = Path(input_path)
    output_path = Path(output_path)

    audio, sr = sf.read(input_path)
    audio = _to_mono_float(audio)

    audio = bandpass_speech(audio, sr)

    if use_noise_reduce:
        audio = mild_noise_reduce(audio, sr)

    audio = peak_normalize(audio)
    sf.write(output_path, audio, sr)
    return str(output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--noise-reduce", action="store_true")
    args = parser.parse_args()

    result = process_audio_file(args.input, args.output, use_noise_reduce=args.noise_reduce)
    print(f"Saved: {result}")
