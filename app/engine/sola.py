from __future__ import annotations

import numpy as np


class Sola:
    def __init__(self, block: int, overlap: int, search: int) -> None:
        self.block_size = block
        self.overlap = overlap
        self.search = search
        phase = np.linspace(0, np.pi / 2, overlap, dtype=np.float32)
        self.fade_in = np.square(np.sin(phase))
        self.fade_out = 1 - self.fade_in
        self._previous = np.zeros(overlap, dtype=np.float32)

    def join(self, window: np.ndarray) -> np.ndarray:
        if len(window) != self.block_size + self.overlap + self.search:
            raise ValueError("音色模型返回的拼接窗口长度不正确")
        segment = window[: self.overlap + self.search]
        numerator = np.correlate(segment, self._previous, mode="valid")
        power = np.concatenate(([0.0], np.cumsum(segment * segment, dtype=np.float64)))
        denominator = np.sqrt(power[self.overlap :] - power[: -self.overlap] + 1e-8)
        offset = int(np.argmax(numerator / denominator))
        aligned = window[offset : offset + self.block_size + self.overlap]
        output = aligned[: self.block_size].copy()
        output[: self.overlap] *= self.fade_in
        output[: self.overlap] += self._previous * self.fade_out
        self._previous[:] = aligned[self.block_size :]
        return output
