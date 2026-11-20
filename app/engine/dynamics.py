from __future__ import annotations

import math

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

FRAME = 480  # 10 ms at 48 kHz
_HOLD_FRAMES = 30
_HYSTERESIS = 10 ** (-6 / 10)
# The converted output trails the microphone by five to six frames. Looking
# that far ahead on the input timeline opens the gate before the first sound.
_LOOKAHEAD_FRAMES = 6
_RAMP = (np.arange(FRAME, dtype=np.float32) + 1) / FRAME
_UNGATED_SPEECH_POWER = 10 ** (-40 / 10)


def frame_power(block: np.ndarray) -> np.ndarray:
    frames = block.reshape(-1, FRAME)
    return np.mean(frames * frames, axis=1)


def speech_power(threshold_db: float) -> float:
    """Frames at or above this power count as speech."""
    return _UNGATED_SPEECH_POWER if threshold_db <= -60 else 10 ** (threshold_db / 10)


class NoiseGate:
    """Mute the converted voice while the microphone is idle.

    The decision uses the raw microphone level. A gate that zeroes single
    quiet frames removes consonants and word endings, so this one stays open
    through a hold time and closes at a lower level than it opens. The gain
    is applied to the output, which leaves the model input continuous.
    """

    def __init__(self) -> None:
        self._hold = 0
        self._flags = np.zeros(_LOOKAHEAD_FRAMES, dtype=bool)
        self._gain = np.float32(0)

    def detect(self, power: np.ndarray, threshold_db: float) -> None:
        active = np.ones(len(power), dtype=bool)
        if threshold_db <= -60:
            self._hold = _HOLD_FRAMES
        else:
            open_power = speech_power(threshold_db)
            close_power = open_power * _HYSTERESIS
            for index, value in enumerate(power):
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


class InputLeveler:
    """Raise a quiet microphone toward the level the models were trained on.

    The level follows speech frames only, so pauses do not pump the gain. It
    never attenuates a signal that is already loud enough.
    """

    _TARGET_POWER = 10 ** (-20 / 10)
    _MAX_GAIN = 8.0
    _PEAK = 0.95
    _SMOOTHING = 1 - math.exp(-0.01 / 1.5)

    def __init__(self) -> None:
        self._power = self._TARGET_POWER
        self._gain = 1.0

    def process(self, block: np.ndarray, power: np.ndarray, minimum_power: float) -> np.ndarray:
        for value in power[power >= minimum_power]:
            self._power += self._SMOOTHING * (float(value) - self._power)
        gain = min(max(math.sqrt(self._TARGET_POWER / self._power), 1.0), self._MAX_GAIN)
        peak = float(np.max(np.abs(block)))
        if peak * gain > self._PEAK:
            # Follow a sudden loud sound at once and release slowly afterwards.
            gain = max(1.0, self._PEAK / peak)
            self._power = max(self._power, self._TARGET_POWER / gain**2)
        previous, self._gain = self._gain, gain
        if previous == 1.0 and gain == 1.0:
            return block
        gains = np.full(len(block), gain, dtype=np.float32)
        ramp = FRAME if gain < previous else len(block)
        gains[:ramp] = np.linspace(previous, gain, ramp, endpoint=False, dtype=np.float32)
        return np.clip(block * gains, -1, 1)
