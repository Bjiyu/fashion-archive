#!/usr/bin/env python3
"""Ambient audio generator.

Creates a subtle "service worker calm" ambient loop — a relaxation-style
audio texture (gentle pink noise, soft low hums, slow swells). This is an
entertainment/relaxation texture, not a real behavior-control frequency.

Usage:
    python tools/ambient_audio_generator.py [output_path]

If no output path is given, the file is written to ./output/ next to the repo
root. Requires numpy and scipy (see tools/requirements.txt).
"""

import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile


def generate(duration: int = 300, sr: int = 44100, seed: int = 1212) -> tuple[int, np.ndarray]:
    """Generate the stereo ambient texture and return (sample_rate, int16 stereo)."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    rng = np.random.default_rng(seed)

    # Gentle pink-ish noise via frequency-domain shaping.
    white = rng.normal(0, 1, len(t))
    freqs = np.fft.rfftfreq(len(white), 1 / sr)
    spectrum = np.fft.rfft(white)
    shape = 1 / np.maximum(freqs, 20) ** 0.55  # pink shaping, avoid divide by zero
    pink = np.fft.irfft(spectrum * shape, n=len(white))
    pink = pink / np.max(np.abs(pink))

    # Soft low hums, intentionally subtle and non-obvious.
    base_174 = np.sin(2 * np.pi * 174 * t) * 0.045
    base_220 = np.sin(2 * np.pi * 220 * t + 0.7) * 0.025
    soft_432 = np.sin(2 * np.pi * 432 * t + 1.3) * 0.018

    # Very slow calming pulse, almost unnoticeable.
    slow_pulse = 0.82 + 0.18 * np.sin(2 * np.pi * 0.08 * t)

    # Light shimmer, quiet enough to feel like shop ambience.
    shimmer = (
        np.sin(2 * np.pi * 864 * t + 0.4) * 0.006
        + np.sin(2 * np.pi * 1296 * t + 1.1) * 0.004
    )

    # Soft breath-like swell every ~16 seconds.
    breath = 0.5 + 0.5 * np.sin(2 * np.pi * (1 / 16) * t - np.pi / 2)
    breath = breath ** 2

    audio = (
        pink * 0.055
        + (base_174 + base_220 + soft_432) * slow_pulse
        + shimmer * breath
    )

    # Fade in/out to avoid abrupt cuts.
    fade_len = int(sr * 4)
    audio[:fade_len] *= np.linspace(0, 1, fade_len)
    audio[-fade_len:] *= np.linspace(1, 0, fade_len)

    # Normalize softly, keep low volume.
    audio = audio / np.max(np.abs(audio)) * 0.35

    # Stereo: tiny difference between channels for space, not strong binaural.
    left = audio
    right = audio * 0.97 + np.roll(audio, int(sr * 0.012)) * 0.03
    stereo = np.column_stack([left, right])

    stereo_int16 = np.int16(np.clip(stereo, -1, 1) * 32767)
    return sr, stereo_int16


def main() -> None:
    if len(sys.argv) > 1:
        out_path = Path(sys.argv[1])
    else:
        out_dir = Path(__file__).resolve().parent.parent / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "진상_퇴치_주파수_서비스직_진정용_5분.wav"

    sr, stereo_int16 = generate()
    wavfile.write(out_path, sr, stereo_int16)
    print(out_path.as_posix())


if __name__ == "__main__":
    main()
