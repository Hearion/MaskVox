"""Time-domain effects applied after the vocoder."""

from __future__ import annotations

import math

import numpy as np

from .vocoder import SAMPLE_RATE

_TWO_PI = 2 * np.pi
_ECHO_SAMPLES = round(0.26 * SAMPLE_RATE)
_TREMOLO_HZ = 5.5
_REVERB_FFT = 65536
_REVERB_SAMPLES = round(1.3 * SAMPLE_RATE)
_KNEE = 0.8


def _impulse_response() -> np.ndarray:
    """A diffuse room: decaying noise that darkens as it rings out."""
    random = np.random.default_rng(7)
    time = np.arange(_REVERB_SAMPLES) / SAMPLE_RATE
    response = random.standard_normal(_REVERB_SAMPLES) * np.exp(-6.9 * time / 1.1)
    response[: round(0.012 * SAMPLE_RATE)] = 0
    spectrum = np.fft.rfft(response)
    hz = np.fft.rfftfreq(_REVERB_SAMPLES, 1 / SAMPLE_RATE)
    spectrum *= (hz / np.hypot(hz, 180)) / np.hypot(1, hz / 4500)
    response = np.fft.irfft(spectrum, _REVERB_SAMPLES)
    return response / math.sqrt(float(np.sum(response * response)))


_ROOM = np.fft.rfft(_impulse_response(), _REVERB_FFT)


def soft_limit(samples: np.ndarray) -> np.ndarray:
    """Leave normal levels alone and round off peaks before they clip."""
    size = np.abs(samples)
    if size.max(initial=0) <= _KNEE:
        return samples
    over = _KNEE + (1 - _KNEE) * np.tanh((size - _KNEE) / (1 - _KNEE))
    return np.where(size > _KNEE, np.sign(samples) * over, samples).astype(np.float32)


class Effects:
    def __init__(self) -> None:
        self._ring_phase = 0.0
        self._tremolo_phase = 0.0
        self._delay = np.zeros(_ECHO_SAMPLES, dtype=np.float32)
        self._delay_position = 0
        self._tail = np.zeros(_REVERB_FFT)
        self._tail_blocks = 0

    def _oscillator(self, attribute: str, hz: float, count: int) -> np.ndarray:
        step = _TWO_PI * hz / SAMPLE_RATE
        start = getattr(self, attribute)
        setattr(self, attribute, (start + step * count) % _TWO_PI)
        return np.sin(start + step * np.arange(count))

    def _echo(self, samples: np.ndarray, amount: float) -> np.ndarray:
        count = len(samples)
        positions = (self._delay_position + np.arange(count)) % _ECHO_SAMPLES
        delayed = self._delay[positions]
        self._delay[positions] = samples * min(amount * 2, 1) + delayed * (0.2 + 0.4 * amount)
        self._delay_position = (self._delay_position + count) % _ECHO_SAMPLES
        return samples + delayed * 0.6

    def _reverb(self, samples: np.ndarray, amount: float) -> np.ndarray:
        count = len(samples)
        if amount > 0:
            if count + _REVERB_SAMPLES - 1 > _REVERB_FFT:
                raise ValueError("音频块过长，混响无法处理")
            self._tail += np.fft.irfft(np.fft.rfft(samples * (0.7 * amount), _REVERB_FFT) * _ROOM)
            self._tail_blocks = math.ceil(_REVERB_FFT / count)
        wet = self._tail[:count].copy()
        self._tail[:-count] = self._tail[count:]
        self._tail[-count:] = 0
        self._tail_blocks -= 1
        return samples * (1 - 0.3 * amount) + wet

    def process(self, samples: np.ndarray, params: dict) -> np.ndarray:
        count = len(samples)
        ring = float(params.get("ring_hz", 0))
        if ring > 0:
            # Multiplying by a sine halves the power. Make that up.
            samples = samples * (1.4 * self._oscillator("_ring_phase", ring, count))
        tremolo = float(params.get("tremolo", 0))
        if tremolo > 0:
            wobble = self._oscillator("_tremolo_phase", _TREMOLO_HZ, count)
            samples = samples * (1 - tremolo * (0.5 + 0.5 * wobble))
        drive = float(params.get("drive", 0))
        if drive > 0:
            gain = 1 + 8 * drive
            # Quiet parts come up a little and peaks flatten, at about the same loudness.
            samples = np.tanh(samples * gain) / gain**0.75
        echo = float(params.get("echo", 0))
        if echo > 0 or self._delay.any():
            samples = self._echo(samples, echo)
        reverb = float(params.get("reverb", 0))
        if reverb > 0 or self._tail_blocks > 0:
            samples = self._reverb(samples, reverb)
        return np.asarray(samples, dtype=np.float32)
