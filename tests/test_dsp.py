from __future__ import annotations

import time
import unittest

import numpy as np

from app.engine.chain import BLOCK, VoiceChain
from app.engine.effects import Effects, soft_limit
from app.engine.gate import FRAME, NoiseGate
from app.engine.presets import ADJUSTABLE, BY_ID, PRESETS, chain_params
from app.engine.vocoder import FFT, LATENCY, SAMPLE_RATE, Vocoder

RATE = SAMPLE_RATE


def _voice(hz: float, seconds: float = 1.0, formant: float = 700.0) -> np.ndarray:
    """Harmonics of one pitch under a single resonance, like a held vowel."""
    time_axis = np.arange(round(seconds * RATE / BLOCK) * BLOCK) / RATE
    wave = np.zeros(len(time_axis))
    for harmonic in range(1, int(8000 / hz)):
        weight = 1 / (1 + ((harmonic * hz - formant) / 250) ** 2)
        wave += weight * np.sin(2 * np.pi * harmonic * hz * time_axis + harmonic)
    return (0.3 * wave / np.max(np.abs(wave))).astype(np.float32)


def _run(processor, signal: np.ndarray, params: dict) -> np.ndarray:
    blocks = [processor.process(signal[start : start + BLOCK], params) for start in range(0, len(signal), BLOCK)]
    return np.concatenate(blocks)


def _spectrum(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tail = signal[-RATE // 2 :]
    return np.fft.rfftfreq(len(tail), 1 / RATE), np.abs(np.fft.rfft(tail * np.hanning(len(tail))))


def _periodicity(signal: np.ndarray) -> tuple[float, float]:
    """Pitch from the autocorrelation, and how strongly the signal repeats."""
    tail = signal[-RATE // 2 :].astype(np.float64)
    correlation = np.fft.irfft(np.abs(np.fft.rfft(tail, 2 * len(tail))) ** 2)[: RATE // 60]
    correlation /= correlation[0]
    first = RATE // 600
    peak = correlation[first:].max()
    lag = first + int(np.argmax(correlation[first:] >= 0.92 * peak))
    lag += int(np.argmax(correlation[lag : lag + 12]))
    return RATE / lag, float(peak)


def _fundamental(signal: np.ndarray) -> float:
    return _periodicity(signal)[0]


def _resonance(signal: np.ndarray) -> float:
    hz, magnitude = _spectrum(signal)
    return float(hz[np.argmax(magnitude)])


class VocoderTests(unittest.TestCase):
    def test_untouched_voice_passes_through_exactly(self) -> None:
        signal = _voice(140)
        output = _run(Vocoder(), signal, {})
        np.testing.assert_allclose(output[LATENCY + FFT :], signal[FFT:-LATENCY], atol=1e-5)

    def test_pitch_moves_and_formant_stays(self) -> None:
        signal = _voice(120)
        for semitones in (9, -5):
            output = _run(Vocoder(), signal, {"pitch": semitones})
            self.assertAlmostEqual(_fundamental(output), 120 * 2 ** (semitones / 12), delta=4)
            self.assertAlmostEqual(_resonance(output), 700, delta=110)
            self.assertAlmostEqual(float(np.std(output[RATE // 2 :])) / float(np.std(signal)), 1, delta=0.35)

    def test_formant_moves_and_pitch_stays(self) -> None:
        output = _run(Vocoder(), _voice(120), {"formant": 6})
        self.assertAlmostEqual(_fundamental(output), 120, delta=4)
        self.assertAlmostEqual(_resonance(output), 700 * 2**0.5, delta=130)

    def test_robot_speaks_on_one_note(self) -> None:
        for hz in (110, 190):
            output = _run(Vocoder(), _voice(hz), {"mode": "robot"})
            self.assertAlmostEqual(_fundamental(output), RATE / 512, delta=3)

    def test_whisper_has_no_pitch(self) -> None:
        output = _run(Vocoder(), _voice(150), {"mode": "whisper"})
        self.assertGreater(_periodicity(_voice(150))[1], 0.8)
        self.assertLess(_periodicity(output)[1], 0.3)
        self.assertGreater(float(np.std(output[RATE // 2 :])), 0.02)

    def test_tune_snaps_to_the_scale(self) -> None:
        # 150 Hz lies between D3 and D#3. The nearest scale note is D3.
        output = _run(Vocoder(), _voice(150), {"tune": True})
        self.assertAlmostEqual(_fundamental(output), 146.83, delta=2.5)

    def test_band_limits_remove_highs(self) -> None:
        signal = (0.2 * np.random.default_rng(0).standard_normal(RATE)).astype(np.float32)[: RATE // BLOCK * BLOCK]
        hz, magnitude = _spectrum(_run(Vocoder(), signal, {"band": "phone"}))
        self.assertLess(magnitude[hz > 6000].mean(), 0.01 * magnitude[(hz > 800) & (hz < 3000)].mean())
        self.assertLess(magnitude[hz < 150].mean(), 0.05 * magnitude[(hz > 800) & (hz < 3000)].mean())


class EffectTests(unittest.TestCase):
    def test_echo_repeats_after_the_delay(self) -> None:
        signal = np.zeros(BLOCK * 30, dtype=np.float32)
        signal[:200] = 0.5
        output = _run(Effects(), signal, {"echo": 0.5})
        np.testing.assert_allclose(output[:200], 0.5)
        delay = round(0.26 * RATE)
        self.assertGreater(float(output[delay : delay + 200].mean()), 0.2)
        self.assertLess(float(np.abs(output[400:delay]).max()), 1e-6)

    def test_reverb_rings_out_and_then_goes_idle(self) -> None:
        effects = Effects()
        burst = (0.3 * np.random.default_rng(1).standard_normal(BLOCK)).astype(np.float32)
        silence = np.zeros(BLOCK, dtype=np.float32)
        first = effects.process(burst, {"reverb": 0.5})
        tail = [effects.process(silence, {"reverb": 0}) for _ in range(70)]
        self.assertTrue(np.isfinite(first).all())
        self.assertGreater(float(np.abs(tail[3]).max()), 1e-3)
        self.assertGreater(float(np.abs(tail[3]).max()), 5 * float(np.abs(tail[40]).max()))
        np.testing.assert_array_equal(tail[-1], 0)
        self.assertLessEqual(effects._tail_blocks, 0)

    def test_limiter_only_touches_peaks(self) -> None:
        quiet = np.linspace(-0.8, 0.8, 64, dtype=np.float32)
        self.assertIs(soft_limit(quiet), quiet)
        loud = soft_limit(np.array([-3, 0.5, 0.9, 5], dtype=np.float32))
        self.assertLessEqual(np.abs(loud).max(), 1)
        self.assertLess(loud[2], np.float32(0.9))
        self.assertEqual(loud[1], np.float32(0.5))


class GateTests(unittest.TestCase):
    def _gains(self, gate: NoiseGate, levels: list[float], threshold: float) -> np.ndarray:
        block = np.repeat(np.array(levels, dtype=np.float32), FRAME)
        gate.detect(block, threshold)
        return gate.apply(np.ones(len(block), dtype=np.float32)).reshape(-1, FRAME)

    def test_gate_keeps_quiet_speech_frames_open(self) -> None:
        gate = NoiseGate()
        levels = [0.001] * 12 + [0.2] * 8 + [0.004] * 6 + [0.001] * 20 + [0.2] * 6
        gains = self._gains(gate, levels, -45)
        np.testing.assert_array_equal(gains[:12], 0)
        np.testing.assert_array_equal(gains[13:], 1)
        tail = self._gains(gate, [0.001] * 120, -45)
        np.testing.assert_array_equal(tail[:60], 1)
        np.testing.assert_array_equal(tail[64:], 0)
        self.assertLessEqual(np.abs(np.diff(tail.reshape(-1))).max(), 1 / FRAME + 1e-6)

    def test_gate_can_be_switched_off(self) -> None:
        np.testing.assert_array_equal(self._gains(NoiseGate(), [0.0001] * 8, -60)[1:], 1)


class ChainTests(unittest.TestCase):
    def test_every_preset_runs_and_stays_in_range(self) -> None:
        signal = _voice(130, 0.5)
        self.assertEqual(len(BY_ID), len(PRESETS))
        for preset in PRESETS:
            with self.subTest(preset=preset["id"]):
                settings = {"preset": preset["id"], "gate_db": -50, **{key: preset["params"][key] for key in ADJUSTABLE}}
                output = _run(VoiceChain(), signal, chain_params(settings))
                self.assertEqual(output.shape, signal.shape)
                self.assertEqual(output.dtype, np.float32)
                self.assertTrue(np.isfinite(output).all())
                self.assertLess(float(np.abs(output).max()), 1)
                self.assertGreater(float(np.std(output[RATE // 4 :])), 0.005)

    def test_sliders_override_the_preset(self) -> None:
        params = chain_params({"preset": "robot", "pitch": 3, "formant": -2, "echo": 0.2, "reverb": 0, "gate_db": -40})
        self.assertEqual((params["mode"], params["pitch"], params["formant"], params["echo"], params["gate_db"]),
                         ("robot", 3, -2, 0.2, -40))

    def test_heaviest_preset_is_far_faster_than_real_time(self) -> None:
        chain, block = VoiceChain(), _voice(130, 0.5)[:BLOCK]
        params = chain_params({"preset": "cave", "pitch": 7, "formant": 3, "echo": 0.4, "reverb": 0.5})
        for _ in range(5):
            chain.process(block, params)
        started = time.perf_counter()
        for _ in range(40):
            chain.process(block, params)
        self.assertLess((time.perf_counter() - started) / 40, 0.5 * BLOCK / RATE)


if __name__ == "__main__":
    unittest.main()
