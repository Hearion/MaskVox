from __future__ import annotations

import numpy as np


class Passthrough:
    """Copy the microphone signal with the real worker and buffers."""

    backend = "直通"
    delay_ms = 0.0
    startup_delay_ms = 0.0

    def process(self, block: np.ndarray, params: dict) -> np.ndarray:
        return block
