from __future__ import annotations

import numpy as np


class RingBuffer:
    """Single-producer, single-consumer float32 buffer on CPython.

    Each side publishes only its own cursor after the NumPy copy has finished.
    Copies may release the GIL. The published cursors keep the producer and
    consumer in disjoint memory while either copy is in progress.
    """

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("音频缓冲区容量必须大于零")
        self.capacity = capacity
        self._storage = np.empty(capacity, dtype=np.float32)
        self._read = 0
        self._write = 0

    @property
    def available(self) -> int:
        return self._write - self._read

    def write(self, samples: np.ndarray) -> int:
        """Write what fits, keeping the pending data safe from overwrite."""
        count = min(len(samples), self.capacity - self.available)
        start = self._write % self.capacity
        first = min(count, self.capacity - start)
        self._storage[start : start + first] = samples[:first]
        self._storage[: count - first] = samples[first:count]
        self._write += count
        return count

    def trim(self, limit: int) -> int:
        """Discard old data on the consumer side to catch up to real time."""
        dropped = max(0, self.available - limit)
        self._read += dropped
        return dropped

    def read_into(self, target: np.ndarray) -> int:
        """Fill preallocated output, writing silence for missing samples."""
        count = min(len(target), self.available)
        start = self._read % self.capacity
        first = min(count, self.capacity - start)
        target[:first] = self._storage[start : start + first]
        target[first:count] = self._storage[: count - first]
        target[count:] = 0
        self._read += count
        return count
