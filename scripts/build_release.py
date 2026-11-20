"""Build a self-contained Windows x64 folder or ZIP from the main environment."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import struct
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
if not __package__:
    sys.path.insert(0, str(ROOT))

from app.cable_setup import CABLE_ARCHIVE, verify_installer_archive
from app.config import VERSION

EMBED_NAME = "python-3.10.11-embed-amd64.zip"
EMBED_URL = f"https://www.python.org/ftp/python/3.10.11/{EMBED_NAME}"
EMBED_SHA256 = "608619f8619075629c9c69f361352a0da6ed7e62f83a0e19c63e0ea32eb7629d"
EXCLUDED = shutil.ignore_patterns(
    "__pycache__", "*.pyc", "*.whl", "tests", "pip", "pip-*.dist-info", "setuptools",
    "setuptools-*.dist-info", "_distutils_hack", "distutils-precedence.pth", "pkg_resources",
)


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


def build(output: Path, archive: bool) -> None:
    if sys.platform != "win32" or sys.version_info[:2] != (3, 10) or struct.calcsize("P") != 8:
        raise ValueError("请在 Windows 64 位主程序环境中使用 Python 3.10 运行。")
    if sys.base_prefix == sys.prefix:
        raise ValueError("请使用 .venv\\Scripts\\python.exe 构建发行包。")
    zip_path = output.with_name(output.name + ".zip")
    if output.exists() or (archive and zip_path.exists()):
        raise ValueError("输出已存在。请指定新的 --out 路径。")
    driver = verify_installer_archive()

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
        shutil.copytree(packages, runtime / "Lib" / "site-packages", ignore=EXCLUDED)
        shutil.copytree(ROOT / "app", staging / "app", ignore=EXCLUDED)
        (staging / "assets" / "drivers").mkdir(parents=True)
        (staging / "assets" / "drivers" / CABLE_ARCHIVE.name).write_bytes(driver)
        for name in ("start.bat", "README.md", "THIRD_PARTY.md"):
            shutil.copy2(ROOT / name, staging / name)

        check = "import app.main, app.engine.chain, numpy, sounddevice, webview"
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
    parser.add_argument("--zip", action="store_true", help="另存可分发的 ZIP 压缩包")
    arguments = parser.parse_args()
    try:
        build(arguments.out.resolve(), arguments.zip)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"构建失败：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
