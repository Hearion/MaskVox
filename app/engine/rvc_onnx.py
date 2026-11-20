from __future__ import annotations

from pathlib import Path

import numpy as np

from .ort_session import create_session

_REQUIRED_INPUTS = {"phone", "phone_lengths", "pitch", "pitchf", "ds", "rnd"}
# RVC samples its latent prior at this scale. The exported graph takes the
# noise as given, and full-scale noise makes the voice rough and unstable.
_NOISE_SCALE = np.float32(0.66666)


class RvcModel:
    def __init__(self, voice_dir: Path, voice: dict, gpu_index: int, frames: int) -> None:
        self.sample_rate = int(voice["sr"])
        self.dimension = int(voice["vec_dim"])
        self.stride = self.sample_rate // 100
        self.frames = frames
        if self.sample_rate not in (32000, 40000, 48000) or self.dimension not in (256, 768):
            raise ValueError("音色采样率或版本不受支持")
        if not voice["f0"]:
            raise ValueError("暂不支持不带音高的音色")
        self.session = create_session(
            voice_dir / "model.onnx", gpu_index,
            {"frames": frames, "samples": frames * self.stride}, from_bytes=True,
        )
        meta = self.session.get_modelmeta().custom_metadata_map
        if meta.get("mv_format") != "1":
            raise ValueError("模型不是受支持的 MaskVox 音色格式")
        if any(meta.get(key) != str(voice[key]) for key in ("sr", "vec_dim", "version")):
            raise ValueError("音色模型与库中的配置不匹配，请重新导入音色")
        inputs = {node.name: node for node in self.session.get_inputs()}
        if set(inputs) not in (_REQUIRED_INPUTS, _REQUIRED_INPUTS | {"source_noise"}):
            raise ValueError("音色模型的推理接口不受支持，请重新转换音色")
        noise_dimension = inputs["rnd"].shape[1]
        self.noise_dimension = int(noise_dimension) if isinstance(noise_dimension, int) else 192
        self.source_noise = "source_noise" in inputs
        self.speaker = np.array([0], dtype=np.int64)
        self.length = np.array([frames], dtype=np.int64)
        self.random = np.random.default_rng()

    def synthesize(self, features: np.ndarray, pitch: np.ndarray, coarse: np.ndarray) -> np.ndarray:
        inputs = {
            "phone": features[None, :, :],
            "phone_lengths": self.length,
            "pitch": coarse[None, :],
            "pitchf": pitch[None, :],
            "ds": self.speaker,
            "rnd": _NOISE_SCALE * self.random.standard_normal(
                (1, self.noise_dimension, self.frames), dtype=np.float32,
            ),
        }
        if self.source_noise:
            inputs["source_noise"] = self.random.standard_normal(
                (1, self.frames * self.stride, 1), dtype=np.float32,
            )
        wave = self.session.run(["audio"], inputs)[0]
        if wave.shape != (1, 1, self.frames * self.stride):
            raise ValueError("音色模型输出长度与采样率不匹配，请重新转换音色")
        return wave[0, 0]
