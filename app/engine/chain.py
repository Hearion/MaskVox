from __future__ import annotations

import numpy as np

from .effects import Effects, soft_limit
from .gate import NoiseGate
from .vocoder import HOP, LATENCY, SAMPLE_RATE, Vocoder

BLOCK = 2 * HOP


class VoiceChain:
    """Noise gate, vocoder and effects for one mono 48 kHz stream."""

    delay_ms = LATENCY / SAMPLE_RATE * 1000

    def __init__(self) -> None:
        self._gate = NoiseGate()
        self._vocoder = Vocoder()
        self._effects = Effects()

    def process(self, block: np.ndarray, params: dict) -> np.ndarray:
        self._gate.detect(block, float(params.get("gate_db", -50)))
        voice = self._vocoder.process(block, params)
        # Gating ahead of the echo and reverb lets their tails ring out.
        return soft_limit(self._effects.process(self._gate.apply(voice), params))
