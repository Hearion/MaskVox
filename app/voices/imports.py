"""Serial background imports for voice packages and community checkpoints."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from threading import Event, Lock, Thread
from uuid import uuid4

from . import converter_client as converter
from .library import VoiceLibrary
from .package import MAX_BYTES, VoiceError, match_index, unpack_package, unpack_rvc

LOGGER = logging.getLogger(__name__)
VALID_SUFFIXES = {".pth", ".zip", ".maskvox", ".index"}


class Imports:
    def __init__(self, library: VoiceLibrary, emit: Callable, on_imported: Callable):
        self._library = library
        self._emit = emit
        self._on_imported = on_imported
        self._lock = Lock()
        self._job_id: str | None = None
        self._aborted: Event | None = None
        self._thread: Thread | None = None
        self._closed = False

    def start(self, paths: list[str]) -> str:
        if not isinstance(paths, list) or not 0 < len(paths) <= 64:
            raise VoiceError("请选择 1～64 个音色文件。")
        files = []
        for path in paths:
            candidate = Path(path)
            if candidate.suffix.lower() not in VALID_SUFFIXES:
                raise VoiceError(f"不支持该文件类型：{candidate.name}")
            if not candidate.is_file() or not 0 < candidate.stat().st_size <= MAX_BYTES:
                raise VoiceError(f"文件无法读取、为空或大于 2 GB：{candidate.name}")
            files.append(candidate.resolve())
        files = list(dict.fromkeys(files))
        if all(path.suffix.lower() == ".index" for path in files):
            raise VoiceError("请同时选择对应的 .pth 音色文件。")
        with self._lock:
            if self._closed:
                raise VoiceError("MaskVox 正在退出。")
            if self._job_id is not None:
                raise VoiceError("已有导入任务，请等当前任务完成或取消。")
            self._job_id, self._aborted = uuid4().hex[:12], Event()
            self._thread = Thread(
                target=self._run, args=(self._job_id, files, self._aborted), name="voice-import", daemon=True
            )
            self._thread.start()
            return self._job_id

    def cancel(self, job_id: str) -> None:
        with self._lock:
            if job_id == self._job_id and self._aborted is not None:
                self._aborted.set()

    def close(self) -> None:
        with self._lock:
            self._closed = True
            thread = self._thread
            if self._aborted is not None:
                self._aborted.set()
        if thread is not None:
            thread.join(timeout=4)

    @staticmethod
    def _check(aborted: Event) -> None:
        if aborted.is_set():
            raise converter.ImportCancelled("已取消导入。")

    def _run(self, job_id: str, paths: list[Path], aborted: Event) -> None:
        work = [path for path in paths if path.suffix.lower() != ".index"]
        indexes = [(path.name, path) for path in paths if path.suffix.lower() == ".index"]
        last_id = None
        current = work[0].name
        error: VoiceError | None = None
        successes = 0

        def progress(stage: str, value: float, **fields) -> None:
            self._emit("import_progress", {
                "job_id": job_id, "file": current, "stage": stage, "progress": value, **fields,
            })

        def check() -> None:
            self._check(aborted)

        def convert_one(name: str, checkpoint: Path, index: Path | None) -> None:
            nonlocal last_id, current, successes
            current = name
            self._check(aborted)
            progress("loading", 0.05)
            with self._library.stage() as tmp:
                target = Path(tmp)
                converter.convert(checkpoint, target, aborted, progress)
                self._check(aborted)
                meta = self._library.add(target, Path(name).stem, name, index)
            last_id = meta["id"]
            successes += 1
            self._emit("voices_changed", {})

        def failed(exc: Exception) -> None:
            nonlocal error
            if isinstance(exc, VoiceError):
                error = exc
                LOGGER.warning("音色导入失败 %s: %s", current, exc)
            else:
                LOGGER.exception("音色导入失败 %s", current)
                error = VoiceError("读取或写入音色失败。请检查文件、磁盘空间和日志。")
            if len(work) > 1:
                self._emit("warning", {"code": "voice_import_error", "message": f"{current}：{error}"})

        try:
            for path in work:
                self._check(aborted)
                current = path.name
                try:
                    suffix = path.suffix.lower()
                    if suffix == ".maskvox":
                        progress("verifying", 0.35)
                        with self._library.stage() as tmp:
                            target = Path(tmp)
                            verified = unpack_package(path, target, check)
                            self._check(aborted)
                            meta = self._library.add(target, verified["name"], path.name, verified=verified)
                        last_id = meta["id"]
                        successes += 1
                        self._emit("voices_changed", {})
                    elif not converter.installed():
                        raise converter.ConverterMissing(
                            "导入 .pth 或 RVC 压缩包需要安装转换组件。可以直接导入 .maskvox 音色包。"
                        )
                    elif suffix == ".zip":
                        progress("loading", 0.02)
                        with self._library.stage() as tmp:
                            checkpoints, bundled = unpack_rvc(path, Path(tmp), check)
                            for name, file in checkpoints:
                                try:
                                    convert_one(name, file, match_index(name, bundled))
                                except converter.ImportCancelled:
                                    raise
                                except Exception as exc:
                                    failed(exc)
                    else:
                        candidates = indexes + [
                            (item.name, item) for item in path.parent.glob("*.index") if item.is_file()
                        ]
                        index = indexes[0][1] if len(work) == len(indexes) == 1 else match_index(path.name, candidates)
                        convert_one(path.name, path, index)
                except converter.ImportCancelled:
                    raise
                except Exception as exc:
                    failed(exc)
            self._check(aborted)
            if last_id is not None:
                self._on_imported(last_id)
                terminal = "done", 1, {"voice_id": last_id, "message": (
                    f"已导入 {successes} 个音色" if error is None
                    else f"已导入 {successes} 个音色。部分文件失败：{error}"
                )}
            else:
                code = "converter_missing" if isinstance(error, converter.ConverterMissing) else "import_failed"
                terminal = "error", 0, {"message": str(error or "没有可导入的音色。"), "code": code}
        except converter.ImportCancelled:
            terminal = "cancelled", 0, {"message": "已取消。此前完成导入的音色已保留。"}
        except Exception:
            LOGGER.exception("音色导入任务异常退出")
            terminal = "error", 0, {"message": "导入任务发生错误。请查看日志。"}
        finally:
            with self._lock:
                self._job_id, self._aborted, self._thread = None, None, None
        stage, amount, details = terminal
        progress(stage, amount, **details)
