from __future__ import annotations

import numpy as np
import soxr

_HALF_WIDTH = 96
_CUTOFF = 7200 / 48000
_OFFSETS = np.arange(-_HALF_WIDTH, _HALF_WIDTH + 1)
_LOW_PASS = (np.sinc(2 * _CUTOFF * _OFFSETS) * np.kaiser(len(_OFFSETS), 7)).astype(np.float32)
_LOW_PASS /= _LOW_PASS.sum()


class InputResampler:
    """Continuous 48 to 16 kHz conversion with a two-millisecond filter delay."""

    delay_ms = _HALF_WIDTH / 48

    def __init__(self) -> None:
        # QQ returns an exact count for every 3:1 block. Use a linear-phase
        # low-pass before it to avoid aliasing without libsoxr's FFT buffering.
        self._resampler = soxr.ResampleStream(48000, 16000, 1, quality="QQ")
        self._history = np.zeros(2 * _HALF_WIDTH, dtype=np.float32)

    def process(self, block: np.ndarray) -> np.ndarray:
        extended = np.concatenate((self._history, block))
        self._history[:] = extended[-len(self._history) :]
        filtered = np.convolve(extended, _LOW_PASS, mode="valid")
        converted = self._resampler.resample_chunk(filtered)
        if len(converted) != len(block) // 3:
            raise RuntimeError("麦克风重采样后长度不正确")
        return converted


class OutputResampler:
    """Convert an overlapping synthesis window on its own time axis."""

    def __init__(self, sample_rate: int) -> None:
        self._resampler = (
            soxr.ResampleStream(sample_rate, 48000, 1, quality="MQ")
            if sample_rate != 48000 else None
        )

    def process(self, window: np.ndarray) -> np.ndarray:
        if self._resampler is None:
            return window
        # Each window repeats overlap from the last one. Flush the filter for
        # this segment so repeated samples never advance a streaming clock.
        self._resampler.clear()
        return self._resampler.resample_chunk(window, last=True)
