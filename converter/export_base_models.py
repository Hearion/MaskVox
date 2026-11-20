"""Export the pinned ContentVec and RMVPE base models from official weights."""

import argparse
import hashlib
import os
import sys
import tempfile
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import onnx
import torch

from .verify import export_graph, verify_export

DEFAULT_ASSETS = Path(__file__).resolve().parents[1] / "assets"
SAMPLE_RATE, FFT, HOP, MEL_BINS, FMIN, FMAX = 16000, 1024, 160, 128, 30, 8000
ASSET_NAMES = ("contentvec.onnx", "rmvpe.onnx", "rmvpe_mel.npz")


@dataclass(frozen=True)
class Source:
    local: str
    remote: str
    sha256: str


CONTENTVEC = "lengyue233/content-vec-best/resolve/ab04aa7067b99ee05cc82499bc64916b980a1967"
RMVPE = "lj1995/VoiceConversionWebUI/resolve/e6d0c1a17da07c33557852f9dfa2bd44cc75737d"
SOURCES = (
    Source("contentvec/config.json", f"{CONTENTVEC}/config.json",
           "2ddde063b795d38d9051a7215a092fecf4cfe148b54251e38de51d88d356898b"),
    Source("contentvec/pytorch_model.bin", f"{CONTENTVEC}/pytorch_model.bin",
           "d8dd400e054ddf4e6be75dab5a2549db748cc99e756a097c496c099f65a4854e"),
    Source("rmvpe.pt", f"{RMVPE}/rmvpe.pt",
           "6d62215f4306e3ca278246188607209f09af3dc77ed4232efdd069798c4ec193"),
)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_source(source: Source, sources_dir: Path) -> None:
    path = sources_dir / source.local
    if path.is_file() and file_hash(path) == source.sha256:
        return
    path.parent.mkdir(exist_ok=True, parents=True)
    preferred = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")
    official = "https://huggingface.co"
    failures = []
    for endpoint in dict.fromkeys((preferred, official)):
        print(f"正在从 {endpoint} 下载 {path.name}…", flush=True)
        url = f"{endpoint}/{source.remote}?download=true"
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".download", delete=False) as output:
                temporary = Path(output.name)
                digest = hashlib.sha256()
                with urlopen(Request(url, headers={"User-Agent": "MaskVox"}), timeout=30) as response:
                    for chunk in iter(lambda: response.read(1024 * 1024), b""):
                        output.write(chunk)
                        digest.update(chunk)
            if digest.hexdigest() != source.sha256:
                raise ValueError("文件 SHA-256 与固定来源不符")
            temporary.replace(path)
            return
        except (OSError, ValueError) as exc:
            failures.append(f"{endpoint}：{exc}")
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    raise ValueError(
        f"下载 {path.name} 失败（{'；'.join(failures)}）。"
        f"请从 {official}/{source.remote} 手动下载并保存为 {path}"
    )


def load_contentvec(path: Path) -> torch.nn.Module:
    from transformers import HubertModel

    class HubertModelWithFinalProj(HubertModel):
        def __init__(self, config):
            super().__init__(config)
            self.final_proj = torch.nn.Linear(config.hidden_size, config.classifier_proj_size)

    class ContentVec(torch.nn.Module):
        def __init__(self, backbone: HubertModelWithFinalProj):
            super().__init__()
            self.backbone = backbone

        def forward(self, audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            outputs = self.backbone(audio, output_hidden_states=True, return_dict=True)
            return outputs.last_hidden_state, self.backbone.final_proj(outputs.hidden_states[9])

    backbone, loading = HubertModelWithFinalProj.from_pretrained(
        path,
        local_files_only=True,
        attn_implementation="eager",
        output_loading_info=True,
    )
    if loading["missing_keys"] or loading["mismatched_keys"]:
        raise ValueError("ContentVec 权重缺少模型所需的参数")
    return ContentVec(backbone).float().eval()


def export_and_check(
    model: torch.nn.Module,
    inputs: list[tuple[torch.Tensor, ...]],
    name: str,
    input_name: str,
    output_names: tuple[str, ...],
    dynamic_axes: dict[str, dict[int, str]],
    staging: Path,
    half_staging: Path | None,
    tolerance: float,
    half_tolerance: float,
) -> None:
    print(f"正在导出 {name}…", flush=True)
    target = staging / name
    export_graph(model, inputs[0], target, (input_name,), output_names, dynamic_axes)
    error = verify_export(model, target, inputs, output_names, max_abs=tolerance)
    print(f"{name} CPU 数值校验通过，最大绝对误差 {error:.6g}。", flush=True)
    if half_staging is not None:
        from onnxconverter_common.float16 import convert_float_to_float16

        reduced = half_staging / name
        graph = convert_float_to_float16(onnx.load(target), keep_io_types=True)
        onnx.save(graph, reduced)
        del graph
        error = verify_export(model, reduced, inputs, output_names, max_abs=half_tolerance)
        print(f"{name} FP16 CPU 数值校验通过，最大绝对误差 {error:.6g}。", flush=True)


def build(sources: Path, staging: Path, half: Path | None) -> None:
    from .vendor.rmvpe import E2E, MelSpectrogram

    rng = torch.Generator().manual_seed(703)
    audio = [torch.randn((1, size), generator=rng) * 0.08 for size in (16000, 20736)]
    content = load_contentvec(sources / "contentvec")
    export_and_check(
        content, [(item,) for item in audio], "contentvec.onnx", "audio",
        ("feats768", "feats256"),
        {"audio": {1: "samples"}, "feats768": {1: "frames"}, "feats256": {1: "frames"}},
        staging, half, 1e-3, 0.12,
    )
    del content

    mel = MelSpectrogram(False, MEL_BINS, SAMPLE_RATE, FFT, HOP, FFT, FMIN, FMAX).eval()
    with torch.inference_mode():
        spectrograms = [mel(samples) for samples in audio]
    paddings = [-item.shape[-1] % 32 for item in spectrograms]
    mels = [(torch.nn.functional.pad(item, (0, padding)),)
            for item, padding in zip(spectrograms, paddings, strict=True)]
    rmvpe = E2E(4, 1, (2, 2)).eval()
    rmvpe.load_state_dict(torch.load(sources / "rmvpe.pt", map_location="cpu", weights_only=True))
    export_and_check(
        rmvpe, mels, "rmvpe.onnx", "mel", ("salience",),
        {"mel": {2: "frames"}, "salience": {1: "frames"}},
        staging, half, 1e-3, 0.04,
    )

    for directory in (staging,) if half is None else (staging, half):
        np.savez(
            directory / "rmvpe_mel.npz", mel_basis=mel.mel_basis.numpy(),
            window=mel.hann_window["0_cpu"].numpy(), sr=SAMPLE_RATE,
            n_fft=FFT, hop=HOP, n_mels=MEL_BINS, fmin=FMIN, fmax=FMAX,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="下载、导出并校验 MaskVox 基础模型")
    parser.add_argument("--out", type=Path, default=DEFAULT_ASSETS, help="基础模型输出目录")
    parser.add_argument("--sources", type=Path, help="原始权重保存目录，默认为输出目录下的 sources")
    parser.add_argument("--fp16-out", type=Path, help="另外生成待真机验证的 FP16 候选模型")
    parser.add_argument("--force", action="store_true", help="重新导出已有的基础模型")
    args = parser.parse_args()
    try:
        destination = args.out.resolve()
        half_destination = args.fp16_out.resolve() if args.fp16_out else None
        if half_destination == destination:
            raise ValueError("FP16 候选需要单独的输出目录")
        if not args.force and half_destination is None and all(
            (destination / name).is_file() for name in ASSET_NAMES
        ):
            print("基础模型已就绪。")
            return 0
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        sources = args.sources or destination / "sources"
        for spec in SOURCES:
            ensure_source(spec, sources)
        destination.mkdir(exist_ok=True, parents=True)
        with ExitStack() as stack:
            staging = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix=".maskvox-base-", dir=destination)))
            half = None
            if half_destination is not None:
                half_destination.mkdir(exist_ok=True, parents=True)
                half = Path(stack.enter_context(
                    tempfile.TemporaryDirectory(prefix=".maskvox-half-", dir=half_destination)
                ))
            build(sources, staging, half)
            for name in ASSET_NAMES:
                (staging / name).replace(destination / name)
            if half_destination is not None and half is not None:
                for name in ASSET_NAMES:
                    (half / name).replace(half_destination / name)
        print(f"基础模型可用，已保存到 {destination}。")
        return 0
    except Exception as exc:
        print(f"基础模型导出失败：{exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
