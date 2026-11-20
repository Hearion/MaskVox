from __future__ import annotations

import threading
import time
import unittest

import numpy as np

from app.audio.ringbuffer import RingBuffer
from app.engine.resample import InputResampler, OutputResampler
from app.engine.sola import Sola


class RingBufferTests(unittest.TestCase):
    def test_wrap_trim_and_silence(self) -> None:
        ring = RingBuffer(5)
        self.assertEqual(ring.write(np.arange(8, dtype=np.float32)), 5)
        output = np.empty(3, dtype=np.float32)
        self.assertEqual(ring.read_into(output), 3)
        np.testing.assert_array_equal(output, [0, 1, 2])
        self.assertEqual(ring.write(np.array([5, 6, 7], dtype=np.float32)), 3)
        self.assertEqual(ring.trim(2), 3)
        self.assertEqual(ring.read_into(output), 2)
        np.testing.assert_array_equal(output, [6, 7, 0])

    def test_single_producer_and_consumer(self) -> None:
        ring = RingBuffer(4099)
        source = np.arange(160_000, dtype=np.float32)
        result = np.empty_like(source)
        finished = threading.Event()

        def produce() -> None:
            offset = 0
            while offset < len(source) and not finished.is_set():
                count = ring.write(source[offset : offset + 1201])
                offset += count
                if count == 0:
                    time.sleep(0.0002)

        writer = threading.Thread(target=produce)
        writer.start()
        offset = 0
        try:
            deadline = time.monotonic() + 3
            while offset < len(result) and time.monotonic() < deadline:
                count = ring.read_into(result[offset : offset + 1703])
                offset += count
                if count == 0:
                    time.sleep(0.0002)
        finally:
            finished.set()
            writer.join(timeout=1)
        self.assertEqual(offset, len(source))
        np.testing.assert_array_equal(result, source)


class AudioProcessingTests(unittest.TestCase):
    def test_sola_stitches_sine_after_independent_phase_shift(self) -> None:
        block = 960
        overlap = 480
        search = 240
        sola = Sola(block, overlap, search)
        phase = 2 * np.pi * 250 / 48000
        positions = np.arange(block + overlap + search)
        initial = np.sin(positions * phase).astype(np.float32)
        previous = sola.join(initial)
        for index in range(1, 8):
            jitter = 15 * index
            shifted = np.sin((positions + index * block - jitter) * phase).astype(np.float32)
            current = sola.join(shifted)
            expected = np.sin((positions[:block] + index * block) * phase)
            np.testing.assert_allclose(current, expected, atol=1e-5)
            self.assertLess(abs(float(current[0] - previous[-1])), 0.04)
            previous = current

    def test_resampling_keeps_an_exact_clock(self) -> None:
        for block in (5760, 9600, 16800):
            stream = InputResampler()
            chunks = []
            for index in range(16):
                positions = np.arange(index * block, (index + 1) * block)
                audio = np.sin(positions * (2 * np.pi * 500 / 48000)).astype(np.float32)
                chunks.append(stream.process(audio))
            result = np.concatenate(chunks)
            settled = round(InputResampler.delay_ms * 16) * 3
            positions = np.arange(settled, len(result)) - round(InputResampler.delay_ms * 16)
            expected = np.sin(positions * 2 * np.pi * 500 / 16000)
            self.assertEqual(len(result), 16 * block // 3)
            np.testing.assert_allclose(result[settled:], expected, atol=1e-3)
            ultrasound = np.sin(np.arange(block) * 2 * np.pi * 10000 / 48000).astype(np.float32)
            filtered = InputResampler().process(ultrasound)
            self.assertLess(np.max(np.abs(filtered[settled:])), 1e-3)
        for rate in (32000, 40000, 48000):
            window = np.ones(17 * rate // 100, dtype=np.float32)
            resampler = OutputResampler(rate)
            self.assertEqual(len(resampler.process(window)), 17 * 480)
            self.assertEqual(len(resampler.process(window)), 17 * 480)


if __name__ == "__main__":
    unittest.main()
