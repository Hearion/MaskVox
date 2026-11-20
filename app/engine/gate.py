from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .vocoder import LATENCY, SAMPLE_RATE

FRAME = 256
_HOLD_FRAMES = round(0.3 * SAMPLE_RATE / FRAME)
_HYSTERESIS = 10 ** (-6 / 10)
# The vocoder delays its output by this many frames. Looking that far ahead
# on the microphone timeline opens the gate just before the first sound.
_LOOKAHEAD_FRAMES = LATENCY // FRAME
_RAMP = (np.arange(FRAME, dtype=np.float32) + 1) / FRAME


class NoiseGate:
    """Mute the processed voice while the microphone is idle.

    The decision uses the raw microphone level. A gate that zeroes single
    quiet frames removes consonants and word endings, so this one stays open
    through a hold time and closes at a lower level than it opens.
    """

    def __init__(self) -> None:
        self._hold = 0
        self._flags = np.zeros(_LOOKAHEAD_FRAMES, dtype=bool)
        self._gain = np.float32(0)

    def detect(self, block: np.ndarray, threshold_db: float) -> None:
        frames = block.reshape(-1, FRAME)
        active = np.ones(len(frames), dtype=bool)
        if threshold_db <= -60:
            self._hold = _HOLD_FRAMES
        else:
            open_power = 10 ** (threshold_db / 10)
            close_power = open_power * _HYSTERESIS
            for index, value in enumerate(np.mean(frames * frames, axis=1)):
                if value >= open_power or (self._hold and value >= close_power):
                    self._hold = _HOLD_FRAMES
                elif self._hold:
                    self._hold -= 1
                active[index] = self._hold > 0
        self._flags = np.concatenate((self._flags[-_LOOKAHEAD_FRAMES:], active))

    def apply(self, output: np.ndarray) -> np.ndarray:
        """Apply the gain for the block passed to the latest detect call."""
        targets = sliding_window_view(self._flags, _LOOKAHEAD_FRAMES + 1).any(axis=1).astype(np.float32)
        starts = np.concatenate(([self._gain], targets[:-1]))
        self._gain = targets[-1]
        if starts[0] == 1 and targets.all():
            return output
        gains = starts[:, None] + (targets - starts)[:, None] * _RAMP
        return output * gains.reshape(-1)
