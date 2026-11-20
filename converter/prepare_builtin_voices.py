"""Developer tool for rebuilding the redistributable preset voice bank."""

import argparse
import json
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
BUILTINS = ASSETS / "builtin-voices"
SOURCES = ASSETS / "sources" / "builtin-voices"
REVISION = "f9b58227fcb9e85b2435b514947d8c672e40dee1"
MODEL = "ampixa/nepali-rvc-voices-v1"
MODEL_URL = f"https://huggingface.co/{MODEL}/tree/{REVISION}"
DATA_URL = "https://huggingface.co/datasets/ai4bharat/indicvoices_r"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
CHECKSUMS = {
    "ne_rvc_m01": "765dfdcbcc9b3dc302f324f27a6b13ddd151e299d3f3f7caa0ce084ca10fee1f",
    "ne_rvc_m02": "b4f1b89364f0b08aebe5c0d6739cde14c354de53c2df280887e06df6af92ec90",
    "ne_rvc_f01": "69b2f39c8b4ddc5519f57f57818da8b75a399f486adf8672ba26c2110e689bce",
    "ne_rvc_f02": "627d1ea98dfc8e6299dfc0ae4220ebad9cf325ff2de55d5fbfa54edde6a3fe40",
}


@dataclass(frozen=True)
class Preset:
    id: str
    name: str
    gender: str
    description: str
    sources: tuple[str, str]
    example: str
    narrator: str
    clip: str


PRESETS = (
    Preset("b0010001", "沉稳男声", "male", "偏低的男声，适合日常通话。",
           ("ne_rvc_m01", "ne_rvc_m02"), "libri1", "Garth Comira", "5703-47212-0000"),
    Preset("b0010002", "清亮女声", "female", "偏亮的女声，适合日常通话。",
           ("ne_rvc_f01", "ne_rvc_f02"), "libri3", "Heather Barnett", "198-209-0000"),
)


def attribution(preset: Preset) -> dict[str, str]:
    return {
        "license": "CC BY 4.0",
        "license_url": LICENSE_URL,
        "creator": "Ampixa Labs",
        "source": f"Nepali RVC Voices v1 / {' + '.join(preset.sources)}",
        "source_url": MODEL_URL,
        "source_revision": REVISION,
        "dataset": "IndicVoices-R (AI4Bharat), CC BY 4.0",
        "dataset_url": DATA_URL,
        "modifications": "MaskVox 等权融合两份 RVC v2 权重，保留一个混合说话人，导出 ONNX。",
        "preview_source": f"LibriSpeech (Vassil Panayotov et al.), clip {preset.clip}, narrated by {preset.narrator}.",
        "preview_source_url": "https://www.openslr.org/12",
        "preview_audio_url": f"https://librosa.org/data/audio/{preset.clip}.ogg",
        "preview_modifications": "节选并通过混合音色变声，统一音量，转换为 48 kHz 单声道 PCM。",
    }


def build_models() -> None:
    import torch

    from .convert_voice import convert
    from .export_base_models import Source, ensure_source

    torch.set_num_threads(3)
    for donor, digest in CHECKSUMS.items():
        remote = f"{MODEL}/resolve/{REVISION}/voices/{donor}/{donor}_v1.pth"
        ensure_source(Source(f"{donor}.pth", remote, digest), SOURCES)
    BUILTINS.mkdir(parents=True, exist_ok=True)
    for preset in PRESETS:
        folder = BUILTINS / preset.id
        with tempfile.TemporaryDirectory(prefix=".builtin-", dir=BUILTINS) as temporary:
            scratch = Path(temporary)
            checkpoints = [torch.load(SOURCES / f"{donor}.pth", map_location="cpu", weights_only=True)
                           for donor in preset.sources]
            config = list(checkpoints[0]["config"])
            weights = [item["weight"] for item in checkpoints]
            if any(item["version"] != "v2" or item["f0"] != 1 or item["config"] != config
                   for item in checkpoints) or weights[0].keys() != weights[1].keys():
                raise ValueError("内置音色的原始模型结构不一致")
            merged = {}
            for key, first in weights[0].items():
                second = weights[1][key]
                if first.shape != second.shape:
                    raise ValueError(f"内置音色的原始张量形状不一致（{key}）")
                if key == "emb_g.weight":
                    first, second = first[:1], second[:1]
                merged[key] = torch.lerp(first.float(), second.float(), 0.5).half()
            config[-3] = 1
            pth = scratch / f"{preset.id}.pth"
            torch.save({"config": config, "weight": merged, "version": "v2", "f0": 1}, pth)
            convert(pth, scratch)
            meta_path = scratch / "meta.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta.update(
                id=preset.id, name=preset.name, gender=preset.gender, description=preset.description,
                has_index=False, source_filename="MaskVox bundled RVC blend", attribution=attribution(preset),
            )
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            folder.mkdir(exist_ok=True)
            for name in ("model.onnx", "meta.json"):
                (scratch / name).replace(folder / name)
    (BUILTINS / "NOTICE.md").write_text(notice(), encoding="utf-8")


def notice() -> str:
    return f"""# MaskVox 内置音色署名

沉稳男声（`b0010001`）融合 `ne_rvc_m01` 与 `ne_rvc_m02`。清亮女声（`b0010002`）融合 `ne_rvc_f01` 与 `ne_rvc_f02`。两组原始 RVC v2 模型来自 Ampixa Labs 的 Nepali RVC Voices v1，固定版本 `{REVISION}`。训练数据来自 AI4Bharat 的 IndicVoices-R。原始模型、训练数据和本次衍生音色按 CC BY 4.0 提供。

MaskVox 对男女两组权重分别等权融合，保留一个混合说话人并重新导出 ONNX。每个内置音色呈现混合音色，未随附单人权重或检索索引。

[Ampixa Labs 原始权重]({MODEL_URL}) · [AI4Bharat 训练数据]({DATA_URL}) · [CC BY 4.0 许可]({LICENSE_URL})

试听使用 LibriSpeech（Vassil Panayotov、Guoguo Chen、Daniel Povey 和 Sanjeev Khudanpur）的公开语音片段，按 CC BY 4.0 提供。男声输入来自 Garth Comira 朗读的 `5703-47212-0000`，女声输入来自 Heather Barnett 朗读的 `198-209-0000`。MaskVox 截取片段、通过对应混合音色变声并调整播放音量。

[LibriSpeech 数据集]({attribution(PRESETS[0])["preview_source_url"]}) · [男声原始输入]({attribution(PRESETS[0])["preview_audio_url"]}) · [女声原始输入]({attribution(PRESETS[1])["preview_audio_url"]})
"""


def prepare_inputs() -> None:
    import librosa
    import numpy as np

    for preset in PRESETS:
        audio, rate = librosa.load(librosa.example(preset.example), sr=None, duration=6.5)
        pcm = np.rint(audio.clip(-1, 1) * 32767).astype("<i2")
        with wave.open(str(SOURCES / f"{preset.id}-input.wav"), "wb") as file:
            file.setparams((1, 2, rate, 0, "NONE", "not compressed"))
            file.writeframes(pcm.tobytes())


def render_previews() -> None:
    import numpy as np

    from app.engine.realtime import RvcPipeline
    from scripts.offline_convert import convert_samples, read_wav, write_wav

    for preset in PRESETS:
        directory = BUILTINS / preset.id
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        samples = read_wav(SOURCES / f"{preset.id}-input.wav")
        pipeline = RvcPipeline(ASSETS, directory, meta, 20, -1, lambda *_: None)
        pipeline._voice.random = np.random.default_rng(113)
        result = convert_samples(pipeline, samples, {"gate_db": -60, "index_rate": 0, "pitch": 0})
        peak = float(np.max(np.abs(result)))
        if not np.isfinite(result).all() or peak < 0.03:
            raise ValueError(f"{preset.name} 试听输出无效")
        result *= 0.87 / peak
        fade = np.linspace(0.0, 1.0, 720, dtype=np.float32)
        result[:len(fade)] *= fade
        result[-len(fade):] *= fade[::-1]
        write_wav(directory / "preview.wav", result)
        print(f"已生成 {preset.name} 的试听。", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="下载许可明确的原始权重，构建 MaskVox 内置混合音色及试听")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--models-only", action="store_true", help="只生成模型与署名")
    action.add_argument("--render-only", action="store_true", help="在主程序虚拟环境中生成试听")
    args = parser.parse_args()
    if args.render_only:
        render_previews()
        return
    build_models()
    if not args.models_only:
        prepare_inputs()
        python = ROOT / ".venv" / "Scripts" / "python.exe"
        subprocess.run([str(python), "-X", "utf8", "-m", "converter.prepare_builtin_voices", "--render-only"],
                       cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
