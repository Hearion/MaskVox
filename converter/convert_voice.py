"""Convert an RVC community checkpoint in the optional CPU environment."""

import argparse
import json
import math
import os
import sys
import tempfile
import traceback
from collections.abc import Mapping
from pathlib import Path

import onnx
import torch

from .vendor.infer_pack.models_onnx import SynthesizerTrnMsNSFsidM, sr2sr
from .verify import export_graph, verify_export

SOURCE_COMMIT = "d60cd3d009b7e438f92f8624fa89b710d10c6a39"
INPUTS = ("phone", "phone_lengths", "pitch", "pitchf", "ds", "rnd", "source_noise")
DYNAMIC_AXES = {
    "phone": {1: "frames"},
    "pitch": {1: "frames"},
    "pitchf": {1: "frames"},
    "rnd": {2: "frames"},
    "source_noise": {1: "samples"},
    "audio": {2: "samples"},
}


class ConversionError(ValueError):
    """A conversion failure with a user-facing message."""


def emit(stage: str, *, progress: float = 0.0, message: str | None = None) -> None:
    event = {"stage": stage, "progress": progress}
    if message is not None:
        event["message"] = message
    print(json.dumps(event, ensure_ascii=True), flush=True)


def bounded_int(value: object, minimum: int, maximum: int) -> bool:
    return type(value) is int and minimum <= value <= maximum


def bounded_list(value: object, length: int, minimum: int, maximum: int) -> bool:
    return (isinstance(value, (list, tuple)) and len(value) == length
            and all(bounded_int(item, minimum, maximum) for item in value))


def read_config(config: object, embedding: torch.Tensor) -> tuple[list, int]:
    invalid = ConversionError("音色模型的参数配置无效或超出支持的范围")
    if not isinstance(config, (list, tuple)) or len(config) != 18:
        raise invalid
    params = list(config)
    for index, upper_bound in {0: 2048, 1: 1024, 2: 768, 3: 768, 4: 3072,
                               5: 24, 6: 24, 7: 31, 13: 1024, 16: 1024}.items():
        if not bounded_int(params[index], 1, upper_bound):
            raise invalid
    if params[2] % 2 or params[3] % params[5]:
        raise invalid
    if type(params[8]) not in (int, float) or not 0 <= params[8] <= 1:
        raise invalid
    if params[9] not in ("1", "2"):
        raise invalid
    kernel_sizes, dilations, strides, stride_kernels = params[10], params[11], params[12], params[14]
    if not isinstance(kernel_sizes, (list, tuple)) or not 1 <= len(kernel_sizes) <= 5:
        raise invalid
    if not bounded_list(kernel_sizes, len(kernel_sizes), 1, 31):
        raise invalid
    if not isinstance(dilations, (list, tuple)) or len(dilations) != len(kernel_sizes):
        raise invalid
    dilation_count = 3 if params[9] == "1" else 2
    if not all(bounded_list(item, dilation_count, 1, 63) for item in dilations):
        raise invalid
    if not isinstance(strides, (list, tuple)) or not 1 <= len(strides) <= 6:
        raise invalid
    if not bounded_list(strides, len(strides), 1, 32) or not bounded_list(
        stride_kernels, len(strides), 1, 80
    ):
        raise invalid
    if any(kernel < stride or (kernel - stride) % 2 for kernel, stride in zip(stride_kernels, strides)):
        raise invalid
    if params[13] < 2 ** len(strides):
        raise invalid
    rate = sr2sr.get(params[17]) if isinstance(params[17], str) else params[17]
    if type(rate) is not int or rate not in sr2sr.values() or math.prod(strides) != rate // 100:
        raise ConversionError("音色模型的采样率不受支持，仅支持 32k、40k 与 48k")
    if embedding.ndim != 2 or embedding.shape[1] != params[16] or not 1 <= embedding.shape[0] <= 10000:
        raise invalid
    params[15] = embedding.shape[0]
    params[17] = rate
    return params, rate


def load_voice(path: Path) -> tuple[SynthesizerTrnMsNSFsidM, dict[str, str]]:
    try:
        if path.stat().st_size > 2 * 1024**3:
            raise ConversionError("音色文件超过 2 GB 大小限制")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except ConversionError:
        raise
    except Exception as exc:
        raise ConversionError("该音色文件格式不安全或已损坏，已拒绝加载") from exc
    if not isinstance(checkpoint, dict):
        raise ConversionError("该音色文件格式不安全或已损坏，已拒绝加载")
    if "weight" not in checkpoint:
        raise ConversionError("这是训练检查点，不是可用于推理的音色文件")
    if checkpoint.get("f0", 1) in (0, "0"):
        raise ConversionError("暂不支持不带音高的音色")
    if checkpoint.get("f0", 1) not in (1, "1"):
        raise ConversionError("音色模型的音高配置无效")
    version = checkpoint.get("version", "v1")
    if version not in ("v1", "v2"):
        raise ConversionError("音色模型版本不受支持，仅支持 RVC v1 和 v2")
    weights = checkpoint["weight"]
    if not isinstance(weights, Mapping) or not all(
        isinstance(key, str) and type(value) is torch.Tensor and value.is_floating_point()
        for key, value in weights.items()
    ) or "emb_g.weight" not in weights:
        raise ConversionError("音色权重无效，缺少 RVC 推理所需的参数")
    params, sr = read_config(checkpoint.get("config"), weights["emb_g.weight"])
    try:
        voice = SynthesizerTrnMsNSFsidM(*params, is_half=False, version=version)
        del voice.enc_q
        synthesis_weights = {key: value for key, value in weights.items() if not key.startswith("enc_q.")}
        voice.load_state_dict(synthesis_weights, strict=True)
    except (TypeError, RuntimeError, ValueError, IndexError) as exc:
        raise ConversionError("音色权重与 RVC 模型参数不匹配") from exc
    voice.float().eval()
    voice.dec.remove_weight_norm()
    voice.flow.remove_weight_norm()
    return voice, {
        "mv_format": "1",
        "version": version,
        "sr": str(sr),
        "f0": "1",
        "vec_dim": "256" if version == "v1" else "768",
        "source_commit": SOURCE_COMMIT,
    }


def make_inputs(frames: int, dim: int, latent: int, sample_rate: int) -> tuple[torch.Tensor, ...]:
    rng = torch.Generator().manual_seed(1600 + frames)
    pitchf = torch.linspace(95, 350, frames)[None, :]
    pitchf[:, ::9] = 0
    minimum_mel, maximum_mel = 1127 * math.log1p(50 / 700), 1127 * math.log1p(1100 / 700)
    mel = torch.log1p(pitchf / 700) * 1127
    pitch = (((mel - minimum_mel) * 254 / (maximum_mel - minimum_mel) + 1)
             .clamp(1, 255).round().to(torch.long))
    return (
        torch.randn((1, frames, dim), generator=rng) * 0.3,
        torch.tensor([frames], dtype=torch.long),
        pitch,
        pitchf,
        torch.zeros(1, dtype=torch.long),
        torch.randn((1, latent, frames), generator=rng),
        torch.randn((1, frames * (sample_rate // 100), 1), generator=rng),
    )


def convert(path: Path, out: Path) -> None:
    emit("loading", progress=0.05)
    voice, properties = load_voice(path)
    sample_rate = int(properties["sr"])
    dim = int(properties["vec_dim"])
    lengths = (29, 91)
    examples = [make_inputs(frames, dim, voice.inter_channels, sample_rate) for frames in lengths]
    output_names = ("model.onnx", "meta.json")
    out.mkdir(parents=True, exist_ok=True)
    if any((out / name).exists() for name in output_names):
        raise ConversionError("目标目录已有转换文件，请使用空目录")
    with tempfile.TemporaryDirectory(prefix=".maskvox-voice-", dir=out) as temporary:
        staging = Path(temporary)
        graph_path = staging / "model.onnx"
        emit("exporting", progress=0.25)
        export_graph(voice, examples[0], graph_path, INPUTS, ("audio",), DYNAMIC_AXES)
        graph = onnx.load(graph_path)
        onnx.helper.set_model_props(graph, properties)
        onnx.save(graph, graph_path)
        del graph
        emit("verifying", progress=0.72)
        expected = [((1, 1, frames * (sample_rate // 100)),) for frames in lengths]
        error = verify_export(voice, graph_path, examples, ("audio",), max_abs=1e-3, expected_shapes=expected)
        print(f".pth → ONNX CPU max_abs_error={error:.6g}", file=sys.stderr, flush=True)
        meta = {
            **properties,
            "name": path.stem,
            "sr": sample_rate,
            "f0": 1,
            "vec_dim": dim,
            "source_filename": path.name,
        }
        (staging / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        published = []
        try:
            for name in output_names:
                target = out / name
                (staging / name).replace(target)
                published.append(target)
        except OSError:
            for target in published:
                target.unlink(missing_ok=True)
            raise
    emit("done", progress=1.0)


def main() -> int:
    parser = argparse.ArgumentParser(description="将 RVC 音色转换成 MaskVox ONNX 音色")
    parser.add_argument("--pth", required=True, type=Path, help="RVC 推理音色路径")
    parser.add_argument("--out", required=True, type=Path, help="转换结果目录")
    args = parser.parse_args()
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    try:
        convert(args.pth, args.out)
        return 0
    except ConversionError as exc:
        emit("error", message=str(exc))
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        emit("error", message=f"音色转换失败：{exc}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
