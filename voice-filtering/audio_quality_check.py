"""
Audio dropout checker for pronunciation recordings.

Usage:
    python audio_quality_check.py path/to/audio.wav

It prints silent/dropout regions and saves two plots next to the input file:
    <name>_waveform_dropout_check.png
    <name>_rms_envelope.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf


def find_regions(mask: np.ndarray, sr: int, min_duration_s: float) -> list[tuple[float, float, float]]:
    regions: list[tuple[float, float, float]] = []
    i = 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j < len(mask) and mask[j]:
                j += 1
            duration = (j - i) / sr
            if duration >= min_duration_s:
                regions.append((i / sr, j / sr, duration))
            i = j
        else:
            i += 1
    return regions


def analyze_audio(audio_path: str | Path, save_plots: bool = True) -> dict:
    audio_path = Path(audio_path)
    audio, sr = sf.read(audio_path)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    audio = audio.astype(np.float64)
    duration = len(audio) / sr
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    rms = float(np.sqrt(np.mean(audio ** 2))) if len(audio) else 0.0

    # Near silence threshold: -60 dB below file peak, but never below 1e-5.
    threshold = max(peak * 10 ** (-60 / 20), 1e-5)
    near_silent = np.abs(audio) < threshold
    silent_regions = find_regions(near_silent, sr, min_duration_s=0.03)

    exact_zero_regions = find_regions(audio == 0, sr, min_duration_s=0.01)

    waveform_path = None
    rms_path = None

    if save_plots:
        # Force a headless backend to avoid Tkinter/threading crashes in web servers.
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        time = np.arange(len(audio)) / sr

        # Plot waveform with dropout highlights.
        plt.figure(figsize=(12, 4))
        plt.plot(time, audio, linewidth=0.8)
        for start, end, _ in silent_regions:
            plt.axvspan(start, end, alpha=0.2)
        plt.title("Waveform: highlighted regions are near-silent/dropout sections")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude")
        plt.tight_layout()
        waveform_path = audio_path.with_name(audio_path.stem + "_waveform_dropout_check.png")
        plt.savefig(waveform_path, dpi=160)
        plt.close()

        # RMS envelope, 20 ms windows.
        win = int(0.02 * sr)
        hop = int(0.01 * sr)
        rms_vals = []
        rms_times = []
        for start in range(0, max(0, len(audio) - win + 1), hop):
            segment = audio[start:start + win]
            rms_vals.append(float(np.sqrt(np.mean(segment ** 2))))
            rms_times.append((start + win / 2) / sr)

        plt.figure(figsize=(12, 4))
        plt.plot(rms_times, rms_vals, linewidth=1.0)
        plt.title("20 ms RMS loudness envelope")
        plt.xlabel("Time (seconds)")
        plt.ylabel("RMS amplitude")
        plt.tight_layout()
        rms_path = audio_path.with_name(audio_path.stem + "_rms_envelope.png")
        plt.savefig(rms_path, dpi=160)
        plt.close()

    return {
        "file": str(audio_path),
        "sample_rate": sr,
        "duration_seconds": round(duration, 3),
        "peak_amplitude": round(peak, 6),
        "overall_rms": round(rms, 6),
        "exact_zero_sample_percentage": round(float(np.mean(audio == 0) * 100), 2),
        "near_silent_regions_over_30ms": [
            (round(s, 3), round(e, 3), round(d, 3)) for s, e, d in silent_regions
        ],
        "exact_zero_regions_over_10ms": [
            (round(s, 3), round(e, 3), round(d, 3)) for s, e, d in exact_zero_regions
        ],
        "waveform_plot": str(waveform_path) if waveform_path else None,
        "rms_plot": str(rms_path) if rms_path else None,
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python audio_quality_check.py path/to/audio.wav")

    result = analyze_audio(sys.argv[1])
    for key, value in result.items():
        print(f"{key}: {value}")
