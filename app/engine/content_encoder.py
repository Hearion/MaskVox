from __future__ import annotations

from pathlib import Path

import numpy as np

from .ort_session import create_session


class ContentEncoder:
    def __init__(self, path: Path, gpu_index: int, audio_size: int, dimension: int) -> None:
        self.output_name = {256: "feats256", 768: "feats768"}[dimension]
        self.session = create_session(path, gpu_index, {"samples": audio_size})

    def encode(self, audio: np.ndarray) -> np.ndarray:
        features = self.session.run([self.output_name], {"audio": audio[None, :]})[0][0]
        # HuBERT's valid convolutions miss the rightmost hop. Extend it as RVC does.
        return np.concatenate((features, features[-1:]), axis=0)
