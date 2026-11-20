from __future__ import annotations

import threading
import time
import unittest

import numpy as np

from app.audio.ringbuffer import RingBuffer


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


if __name__ == "__main__":
    unittest.main()
