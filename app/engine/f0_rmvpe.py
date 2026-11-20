from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .ort_session import create_session

HOP = 160
RMVPE_MULTIPLE = 32
_MIN_MEL = 1127 * np.log1p(50 / 700)
_MAX_MEL = 1127 * np.log1p(1100 / 700)
_CENTS = 20 * np.arange(360, dtype=np.float32) + 1997.3794084376191
_NEIGHBORS = np.arange(-4, 5)


class MelSpectrogram:
    def __init__(self, path: Path) -> None:
        if not path.is_file():
            raise RuntimeError("基础模型缺少 rmvpe_mel.npz，请重新安装基础模型")
        with np.load(path, allow_pickle=False) as data:
            self.basis = np.ascontiguousarray(data["mel_basis"], dtype=np.float32)
            self.window = np.ascontiguousarray(data["window"], dtype=np.float32)
            self.hop = int(data["hop"].item())
            self.n_fft = int(data["n_fft"].item())
            sample_rate = int(data["sr"].item())
        if (
            sample_rate != 16000 or self.hop != HOP or self.n_fft != 1024
            or self.basis.shape != (128, 513) or self.window.shape != (1024,)
            or not np.isfinite(self.basis).all() or not np.isfinite(self.window).all()
        ):
            raise ValueError("RMVPE 梅尔谱常量无效，请重新安装基础模型")

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        audio = np.asarray(audio, dtype=np.float32)
        padded = np.pad(audio, self.n_fft // 2, mode="reflect")
        frames = sliding_window_view(padded, self.n_fft)[:: self.hop]
        magnitude = np.abs(np.fft.rfft(frames * self.window)).astype(np.float32, copy=False)
        mel = self.basis @ magnitude.T
        return np.ascontiguousarray(np.log(np.maximum(mel, 1e-5)))


def decode(salience: np.ndarray, threshold: float = 0.03) -> np.ndarray:
    peaks = np.argmax(salience, axis=1)
    neighbors = peaks[:, None] + _NEIGHBORS[None, :]
    valid = (neighbors >= 0) & (neighbors < len(_CENTS))
    indices = np.clip(neighbors, 0, len(_CENTS) - 1)
    weights = np.where(valid, np.take_along_axis(salience, indices, axis=1), 0)
    denominator = weights.sum(axis=1)
    cents = (weights * _CENTS[indices]).sum(axis=1) / np.maximum(denominator, 1e-12)
    frequencies = 10 * np.exp2(cents / 1200)
    return np.where(np.max(salience, axis=1) > threshold, frequencies, 0).astype(np.float32)


def f0_to_coarse(f0: np.ndarray) -> np.ndarray:
    mel = 1127 * np.log1p(np.maximum(f0, 0) / 700)
    scaled = 1 + 254 * (mel - _MIN_MEL) / (_MAX_MEL - _MIN_MEL)
    return np.rint(np.clip(scaled, 1, 255)).astype(np.int64)


class RMVPE:
    def __init__(self, assets_dir: Path, gpu_index: int, mel_frames: int) -> None:
        self.mel = MelSpectrogram(assets_dir / "rmvpe_mel.npz")
        self.session = create_session(assets_dir / "rmvpe.onnx", gpu_index, {"frames": mel_frames})
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def estimate(self, audio: np.ndarray) -> np.ndarray:
        mel = self.mel(audio)
        frames = mel.shape[1]
        missing = -frames % RMVPE_MULTIPLE
        if missing:
            mel = np.pad(mel, ((0, 0), (0, missing)))
        salience = self.session.run([self.output_name], {self.input_name: mel[None, :, :]})[0]
        return decode(salience[0, :frames])
