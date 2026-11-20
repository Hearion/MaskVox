"""Build a self-contained Windows x64 folder or ZIP from the main environment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import sysconfig
import tempfile
import wave
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
if not __package__:
    sys.path.insert(0, str(ROOT))

from app.cable_setup import CABLE_ARCHIVE, verify_installer_archive
from app.config import BUILTIN_VOICES_DIR, VERSION
from app.voices.library import BUILTIN_VOICE_IDS
from app.voices.package import verify_model

EMBED_NAME = "python-3.10.11-embed-amd64.zip"
EMBED_URL = f"https://www.python.org/ftp/python/3.10.11/{EMBED_NAME}"
EMBED_SHA256 = "608619f8619075629c9c69f361352a0da6ed7e62f83a0e19c63e0ea32eb7629d"
ASSET_NAMES = ("contentvec.onnx", "rmvpe.onnx", "rmvpe_mel.npz")
EXCLUDED = shutil.ignore_patterns(
    "__pycache__", "*.pyc", "*.whl", "tests", "pip", "pip-*.dist-info", "setuptools",
    "setuptools-*.dist-info", "_distutils_hack", "distutils-precedence.pth",
    "pkg_resources", "torch", "torch-*.dist-info",
)


def ignore_package_files(directory: str, names: list[str]) -> set[str]:
    ignored = set(EXCLUDED(directory, names))
    path = Path(directory)
    if path.name == "onnx" or path.parts[-2:] == ("onnx", "backend"):
        ignored.add("test")
    if path.name == "onnxruntime":
        ignored.update({"transformers", "tools"})
    return ignored


def runtime_archive() -> Path:
    cache = ROOT / ".runtime" / EMBED_NAME
    if not cache.is_file():
        cache.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=cache.parent, suffix=".download", delete=False) as temp:
            partial = Path(temp.name)
        try:
            with urlopen(EMBED_URL, timeout=60) as source, partial.open("wb") as target:
                shutil.copyfileobj(source, target)
            verify_download(partial)
            partial.replace(cache)
        finally:
            partial.unlink(missing_ok=True)
    verify_download(cache)
    return cache


def verify_download(path: Path) -> None:
    if hashlib.sha256(path.read_bytes()).hexdigest() != EMBED_SHA256:
        raise ValueError(f"Python 官方运行环境 SHA-256 校验失败：{path}")


def copy_first_run_assets(destination: Path) -> None:
    driver_dir = destination / "drivers"
    driver_dir.mkdir()
    (driver_dir / CABLE_ARCHIVE.name).write_bytes(verify_installer_archive())

    notice = BUILTIN_VOICES_DIR / "NOTICE.md"
    if not notice.is_file():
        raise ValueError("缺少内置音色的来源和许可说明。请先生成内置音色。")
    preset_dir = destination / BUILTIN_VOICES_DIR.name
    preset_dir.mkdir()
    shutil.copy2(notice, preset_dir / notice.name)
    for voice_id, gender in zip(BUILTIN_VOICE_IDS, ("male", "female"), strict=True):
        source = BUILTIN_VOICES_DIR / voice_id
        model = source / "model.onnx"
        metadata = source / "meta.json"
        sample = source / "preview.wav"
        if not all(path.is_file() for path in (model, metadata, sample)):
            raise ValueError(f"缺少内置音色或试听样本 {voice_id}。请先生成内置音色。")
        meta = json.loads(metadata.read_text(encoding="utf-8"))
        model_metadata = verify_model(model)
        if (
            meta.get("id") != voice_id or meta.get("gender") != gender
            or not meta.get("name") or not meta.get("attribution")
            or any(meta.get(key) != value for key, value in model_metadata.items())
        ):
            raise ValueError(f"内置音色元数据不完整或不匹配 {voice_id}。")
        with wave.open(str(sample), "rb") as audio:
            if not audio.getnframes() or audio.getsampwidth() != 2:
                raise ValueError(f"内置音色试听样本不可用 {voice_id}。")
        target = preset_dir / voice_id
        target.mkdir()
        for path in (model, metadata, sample):
            shutil.copy2(path, target / path.name)


def build(output: Path, assets: Path, archive: bool) -> None:
    if sys.platform != "win32" or sys.version_info[:2] != (3, 10) or struct.calcsize("P") != 8:
        raise ValueError("请在 Windows 64 位主程序环境中使用 Python 3.10 运行。")
    if sys.base_prefix == sys.prefix:
        raise ValueError("请使用 .venv\\Scripts\\python.exe 构建发行包。")
    zip_path = output.with_name(output.name + ".zip")
    if output.exists() or (archive and zip_path.exists()):
        raise ValueError("输出已存在。请指定新的 --out 路径。")
    for name in ASSET_NAMES:
        if not (assets / name).is_file():
            raise ValueError(f"缺少基础模型 {assets / name}。请先完成基础模型安装。")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".maskvox-build-", dir=output.parent) as tmp:
        staging = Path(tmp) / output.name
        runtime = staging / "runtime"
        with ZipFile(runtime_archive()) as source:
            source.extractall(runtime)
        (runtime / "python310._pth").write_text(
            "python310.zip\n.\n..\nLib\\site-packages\nimport site\n", encoding="ascii"
        )
        packages = Path(sysconfig.get_path("purelib"))
        if (packages / "torch").exists():
            raise ValueError("当前环境包含转换组件依赖，请切换到主程序 .venv。")
        shutil.copytree(packages, runtime / "Lib" / "site-packages", ignore=ignore_package_files)
        shutil.copytree(ROOT / "app", staging / "app", ignore=EXCLUDED)
        (staging / "assets").mkdir()
        for name in ASSET_NAMES:
            shutil.copy2(assets / name, staging / "assets" / name)
        copy_first_run_assets(staging / "assets")
        for name in ("start.bat", "README.md", "THIRD_PARTY.md", "LICENSE_ContentVec"):
            shutil.copy2(ROOT / name, staging / name)
        shutil.copy2(ROOT / "converter" / "vendor" / "LICENSE_RVC", staging / "LICENSE_RVC")

        check = (
            "import sys, app.main, onnxruntime, onnx, numpy, faiss, sounddevice, soxr, webview; "
            "assert 'torch' not in sys.modules; "
            "assert 'DmlExecutionProvider' in onnxruntime.get_available_providers()"
        )
        subprocess.run([str(runtime / "python.exe"), "-B", "-c", check], cwd=staging, check=True)
        total = sum(file.stat().st_size for file in staging.rglob("*") if file.is_file())
        if archive:
            zip_staging = Path(tmp) / zip_path.name
            with ZipFile(zip_staging, "w", ZIP_DEFLATED, compresslevel=4) as bundle:
                for file in sorted(staging.rglob("*")):
                    if file.is_file():
                        bundle.write(file, arcname=file.relative_to(staging.parent))
            zip_staging.replace(zip_path)
            print(f"压缩包：{zip_path}（{zip_path.stat().st_size / 1e6:.1f} MB）")
        staging.rename(output)
        print(f"便携目录：{output}（{total / 1e6:.1f} MB）")


def main() -> int:
    parser = argparse.ArgumentParser(description="构建 MaskVox 便携发行包")
    parser.add_argument("--out", type=Path, default=ROOT / "dist" / f"MaskVox-{VERSION}")
    parser.add_argument("--assets", type=Path, default=ROOT / "assets")
    parser.add_argument("--zip", action="store_true", help="另存可分发的 ZIP 压缩包")
    arguments = parser.parse_args()
    try:
        build(arguments.out.resolve(), arguments.assets.resolve(), arguments.zip)
    except (OSError, ValueError, RuntimeError, wave.Error, subprocess.CalledProcessError) as exc:
        print(f"构建失败：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
