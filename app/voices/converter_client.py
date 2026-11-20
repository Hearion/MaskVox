"""Run the optional converter and consume its JSON-line progress."""

from __future__ import annotations

import json
import logging
import os
import queue
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from threading import Event, Thread

from app.config import CONVERTER_PYTHON, ROOT

from .package import VoiceError

LOGGER = logging.getLogger(__name__)


class ImportCancelled(VoiceError):
    """The user requested that a running import stop."""


class ConverterMissing(VoiceError):
    """The converter environment is absent."""


def installed() -> bool:
    environment = CONVERTER_PYTHON.parent.parent
    packages = environment / ("Lib/site-packages" if os.name == "nt" else "lib/python3.10/site-packages")
    return (
        CONVERTER_PYTHON.is_file()
        and (ROOT / "converter" / "convert_voice.py").is_file()
        and all((packages / name).is_dir() for name in ("torch", "numpy", "onnx", "onnxruntime"))
    )


def convert(
    pth: Path,
    output: Path,
    cancelled: Event,
    progress: Callable[[str, float], None],
    timeout: float = 300,
) -> None:
    if not installed():
        raise ConverterMissing("导入 .pth 需要先安装转换组件。可直接导入 .maskvox 音色包。")
    if cancelled.is_set():
        raise ImportCancelled("已取消导入。")
    args = [
        str(CONVERTER_PYTHON), "-u", "-m", "converter.convert_voice",
        "--pth", str(pth), "--out", str(output),
    ]
    with subprocess.Popen(
        args,
        cwd=ROOT,
        env={**os.environ, "PYTHONUTF8": "1"},
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
        errors="replace",
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    ) as process:
        messages: queue.Queue[str | None] = queue.Queue()

        def read_stdout() -> None:
            try:
                for line in process.stdout:
                    messages.put(line)
            finally:
                messages.put(None)

        def read_stderr() -> None:
            for line in process.stderr:
                logging.getLogger("converter").info("%s", line.rstrip())

        stdout = Thread(target=read_stdout, name="converter-stdout", daemon=True)
        stderr = Thread(target=read_stderr, name="converter-stderr", daemon=True)
        stdout.start()
        stderr.start()
        deadline = time.monotonic() + timeout
        error = ""
        completed = False
        try:
            while True:
                if cancelled.is_set():
                    raise ImportCancelled("已取消导入。")
                if time.monotonic() >= deadline:
                    raise VoiceError("转换超过 5 分钟，已停止。请检查日志或稍后重试。")
                try:
                    line = messages.get(timeout=0.15)
                except queue.Empty:
                    continue
                if line is None:
                    try:
                        code = process.wait(timeout=2)
                    except subprocess.TimeoutExpired as exc:
                        raise VoiceError("转换组件未正常退出。请查看日志。") from exc
                    break
                try:
                    report = json.loads(line)
                    stage = report["stage"]
                    if stage == "error":
                        error = str(report.get("message") or "转换失败。")
                    elif stage == "done":
                        completed = True
                    elif stage in {"loading", "exporting", "verifying"}:
                        progress(stage, min(1, max(0, float(report.get("progress", 0)))))
                except (KeyError, TypeError, ValueError):
                    LOGGER.info("转换组件输出 %s", line.rstrip())
            if code != 0 or error or not completed or not (output / "model.onnx").is_file():
                raise VoiceError(error or "转换组件未生成音色模型。请查看日志。")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            stdout.join(timeout=2)
            stderr.join(timeout=2)
