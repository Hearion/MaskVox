"""Phase-vocoder voice shaping: pitch, formants and the excitation itself."""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

SAMPLE_RATE = 48000
FFT = 2048
HOP = 512
# Each frame is zero-padded before the transform. The finer bin spacing lets
# a shifted peak land closer to its true frequency, which removes roughness.
_SIZE = 2 * FFT
BINS = _SIZE // 2 + 1
# A sample leaves the overlap-add buffer this long after it enters.
LATENCY = FFT - HOP

_TWO_PI = 2 * np.pi
_WINDOW = np.hanning(FFT + 1)[:-1].astype(np.float32)
# Squared Hann windows at 75 % overlap add up to 1.5.
_SYNTHESIS = _WINDOW / 1.5
_BIN = np.arange(BINS)
# Phase advance over one hop for a sinusoid that sits one bin up.
_HOP_PHASE = _TWO_PI * HOP / _SIZE
_EXPECTED = _HOP_PHASE * _BIN
# Zero phase puts each frame's pulse at sample 0, where the window is zero.
# A linear phase moves it to the middle of the frame.
_CENTERED = np.exp(-1j * _TWO_PI * _BIN * (FFT // 2) / _SIZE)
_PEAK_REACH = 3
_ENVELOPE_PASSES = 4
# Quefrencies below the lifter describe the vocal tract, the rest the pitch.
# A longer lifter resolves formants better but must stay under the pitch
# period, so it follows the speaker's pitch between these bounds.
_MIN_LIFTER, _DEFAULT_LIFTER, _MAX_LIFTER = 64, 120, 220
_MIN_LAG = SAMPLE_RATE // 600
_MAX_LAG = SAMPLE_RATE // 65
_SCALE_STEPS = np.array([0, 2, 4, 5, 7, 9, 11, 12])


def _band(low: float, high: float) -> np.ndarray:
    hz = _BIN * SAMPLE_RATE / _SIZE
    mask = np.ones(BINS)
    if low:
        mask *= np.clip((hz - 0.7 * low) / (0.3 * low), 0, 1)
    if high:
        mask *= np.clip((1.3 * high - hz) / (0.3 * high), 0, 1)
    return mask * mask * (3 - 2 * mask)


def _ramp(low: float, high: float) -> np.ndarray:
    position = np.clip((_BIN * SAMPLE_RATE / _SIZE - low) / (high - low), 0, 1)
    return position * position * (3 - 2 * position)


BANDS = {
    "full": None,
    # No chest rumble below a woman's pitch, and a little more air on top.
    "feminine": _band(150, 0) * (1 + 0.5 * _ramp(2500, 5000)),
    "phone": _band(300, 3400),
    "radio": _band(500, 2600),
    "muffled": _band(0, 700),
}
MODES = ("normal", "robot", "whisper")
# Breath noise replaces part of the harmonics above the first formants.
_BREATH_BAND = _ramp(1500, 4000)
_BREATH_SMOOTHING = np.ones(25) / 25
_PITCH_MEMORY = 1 - np.exp(-HOP / (2.0 * SAMPLE_RATE))


def _envelope(magnitude: np.ndarray, lifter: int) -> np.ndarray:
    """Smooth spectral envelope from the low part of the cepstrum."""
    # Without a floor the gaps between harmonics drag the envelope down.
    floor = max(1e-7, 1e-3 * float(magnitude.max()))
    target = np.log(np.maximum(magnitude, floor))
    smooth = target
    # Plain smoothing averages peaks with the gaps between harmonics and
    # lands below the peaks. Raising the input to the last estimate and
    # smoothing again pulls the envelope up onto them.
    for _ in range(_ENVELOPE_PASSES):
        cepstrum = np.fft.irfft(np.maximum(target, smooth))
        cepstrum[lifter : -lifter + 1] = 0
        smooth = np.fft.rfft(cepstrum).real
    return np.exp(smooth)


def _pitch(magnitude: np.ndarray) -> float:
    """Fundamental frequency from the autocorrelation, or 0 when unvoiced."""
    correlation = np.fft.irfft(magnitude * magnitude)[: _MAX_LAG + 2]
    lag = int(np.argmax(correlation[_MIN_LAG : _MAX_LAG + 1])) + _MIN_LAG
    if lag in (_MIN_LAG, _MAX_LAG) or correlation[lag] < 0.35 * correlation[0]:
        return 0.0
    before, peak, after = correlation[lag - 1 : lag + 2]
    curvature = before - 2 * peak + after
    offset = 0.5 * (before - after) / curvature if curvature < 0 else 0.0
    return SAMPLE_RATE / (lag + offset)


def _snap(frequency: float) -> float:
    """Nearest note of the C major scale."""
    octave, step = divmod(69 + 12 * np.log2(frequency / 440), 12)
    nearest = _SCALE_STEPS[np.argmin(np.abs(_SCALE_STEPS - step))]
    return 440 * 2 ** ((octave * 12 + nearest - 69) / 12)


class Vocoder:
    """Streaming short-time Fourier processor. Blocks are multiples of HOP."""

    def __init__(self) -> None:
        self._input = np.zeros(FFT, dtype=np.float32)
        self._output = np.zeros(FFT, dtype=np.float64)
        self._last_phase = np.zeros(BINS)
        self._rotation = np.zeros(BINS)
        self._tune = 1.0
        self._bend = 1.0
        self._usual_pitch = 0.0
        self._lifter = _DEFAULT_LIFTER
        self._random = np.random.default_rng()

    def _shift(
        self, spectrum: np.ndarray, magnitude: np.ndarray, frequency: np.ndarray,
        ratio: float, envelope: np.ndarray, shaped: np.ndarray,
    ) -> np.ndarray:
        """Move every spectral peak, with the bins around it, to a new pitch.

        Keeping each peak's neighbours together preserves their relative
        phases, which is what keeps a shifted voice from sounding hollow.
        Each bin trades the envelope where it was for the one where it lands.
        """
        neighbourhood = sliding_window_view(np.pad(magnitude, _PEAK_REACH), 2 * _PEAK_REACH + 1)
        peaks = np.flatnonzero((magnitude == neighbourhood.max(axis=1)) & (magnitude > 0))
        if not len(peaks):
            return np.zeros(BINS, dtype=np.complex128)
        region = np.searchsorted((peaks[:-1] + peaks[1:] + 1) // 2, _BIN, side="right")
        offset = frequency[peaks] * (ratio - 1)
        rotation = (self._rotation[peaks] + offset * _HOP_PHASE)[region] % _TWO_PI
        self._rotation = rotation
        target = _BIN + np.rint(offset).astype(np.int64)[region]
        audible = (target >= 0) & (target < BINS)
        target = target[audible]
        output = np.zeros(BINS, dtype=np.complex128)
        moved = spectrum[audible] / envelope[audible] * shaped[target] * np.exp(1j * rotation[audible])
        np.add.at(output, target, moved)
        # A new pitch puts a different number of harmonics under the
        # envelope. Keep the loudness of the frame that came in.
        energy = float(np.sum(output.real**2 + output.imag**2))
        gain = np.sqrt(float(np.sum(magnitude**2)) / energy) if energy > 0 else 1.0
        return output * min(max(gain, 0.25), 4.0)

    def _intonation(self, voiced: float, amount: float) -> float:
        """Widen the rise and fall of the speaker's pitch around its usual value.

        A man's speech melody moves less than a woman's. Shifted up
        unchanged it sounds flat.
        """
        if voiced:
            note = float(np.log(voiced))
            self._usual_pitch = self._usual_pitch or note
            self._usual_pitch += _PITCH_MEMORY * (note - self._usual_pitch)
            wanted = float(np.clip(np.exp(amount * (note - self._usual_pitch)), 0.8, 1.25))
            # Smoothing keeps a single misread pitch from yanking the voice.
            self._bend += 0.3 * (wanted - self._bend)
        return self._bend

    def _breathe(self, output: np.ndarray, amount: float) -> np.ndarray:
        """Trade some of the upper harmonics for noise at the same loudness."""
        power = output.real**2 + output.imag**2
        local = np.sqrt(np.convolve(power, _BREATH_SMOOTHING, mode="same"))
        noise = local * np.exp(1j * self._random.uniform(0, _TWO_PI, BINS))
        return output * (1 - 0.4 * amount * _BREATH_BAND) + noise * (amount * _BREATH_BAND)

    def _frame(self, params: dict) -> np.ndarray:
        spectrum = np.fft.rfft(self._input * _WINDOW, _SIZE)
        magnitude = np.abs(spectrum)
        phase = np.angle(spectrum)
        delta = phase - self._last_phase - _EXPECTED
        self._last_phase = phase
        delta -= _TWO_PI * np.round(delta / _TWO_PI)
        # Each bin's measured frequency, in bins.
        frequency = _BIN + delta / _HOP_PHASE

        mode = params.get("mode", "normal")
        ratio = 2 ** (float(params.get("pitch", 0)) / 12)
        formant = 2 ** (float(params.get("formant", 0)) / 12)
        tune = bool(params.get("tune")) and mode == "normal"
        if not tune:
            self._tune = 1.0

        if mode == "normal" and ratio == 1 and formant == 1 and not tune:
            output = spectrum
            self._rotation[:] = 0
        else:
            voiced = _pitch(magnitude)
            if voiced:
                self._lifter = int(np.clip(0.6 * SAMPLE_RATE / voiced, _MIN_LIFTER, _MAX_LIFTER))
                if tune:
                    self._tune = _snap(voiced * ratio) / (voiced * ratio)
            ratio *= self._tune
            intonation = float(params.get("intonation", 0))
            if intonation > 0 and mode == "normal":
                ratio *= self._intonation(voiced, intonation)
            envelope = _envelope(magnitude, self._lifter)
            shaped = envelope if formant == 1 else np.interp(_BIN / formant, _BIN, envelope, right=0)
            if mode != "normal":
                # The envelope runs below the harmonic peaks. Restore the
                # frame's energy for the synthetic excitations.
                shaped = shaped * np.sqrt(np.sum(magnitude**2) / max(float(np.sum(shaped**2)), 1e-12))
            if mode == "robot":
                output = shaped * _CENTERED
            elif mode == "whisper":
                # Frames with unrelated phases add up weaker than aligned ones.
                output = 1.3 * shaped * np.exp(1j * self._random.uniform(0, _TWO_PI, BINS))
            else:
                output = self._shift(spectrum, magnitude, frequency, ratio, envelope, shaped)
                breath = float(params.get("breath", 0))
                if breath > 0:
                    output = self._breathe(output, breath)

        mask = BANDS[params.get("band", "full")]
        if mask is not None:
            output = output * mask
        return np.fft.irfft(output)[:FFT] * _SYNTHESIS

    def process(self, block: np.ndarray, params: dict) -> np.ndarray:
        if len(block) % HOP:
            raise ValueError("音频块长度必须是 512 的整数倍")
        result = np.empty(len(block), dtype=np.float32)
        for start in range(0, len(block), HOP):
            self._input[:-HOP] = self._input[HOP:]
            self._input[-HOP:] = block[start : start + HOP]
            self._output += self._frame(params)
            result[start : start + HOP] = self._output[:HOP]
            self._output[:-HOP] = self._output[HOP:]
            self._output[-HOP:] = 0
        return result
