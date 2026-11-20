"""On-disk voice catalog with staged imports and atomic metadata writes."""

from __future__ import annotations

import json
import logging
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from uuid import uuid4

from .package import CHAR_LIMIT, MAX_BYTES, VoiceError, verify_model, write_package

LOGGER = logging.getLogger(__name__)
ID_PATTERN = re.compile(r"[0-9a-f]{8}")


def clean_name(name: str) -> str:
    if not isinstance(name, str):
        raise VoiceError("请输入有效的音色名称。")
    name = " ".join(name.split())
    if not name or len(name) > CHAR_LIMIT or any(ord(c) < 32 for c in name):
        raise VoiceError("音色名称长度应为 1～80 个字符。")
    return name


class VoiceLibrary:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()

    def stage(self) -> tempfile.TemporaryDirectory:
        return tempfile.TemporaryDirectory(prefix=".import-", dir=self.root)

    def directory(self, voice_id: str) -> Path:
        if not isinstance(voice_id, str) or not ID_PATTERN.fullmatch(voice_id):
            raise VoiceError("找不到指定的音色。")
        path = self.root / voice_id
        if not path.is_dir() or path.resolve().parent != self.root.resolve():
            raise VoiceError("该音色已被删除或移动，请重新导入。")
        return path

    def get(self, voice_id: str) -> dict:
        with self._lock:
            folder = self.directory(voice_id)
            try:
                meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
                if not isinstance(meta, dict) or meta.get("id") != voice_id:
                    raise ValueError
                return meta
            except (OSError, UnicodeError, ValueError) as exc:
                raise VoiceError("该音色的元数据缺失或已损坏。") from exc

    def list(self) -> list[dict]:
        with self._lock:
            voices = []
            for path in self.root.iterdir():
                if path.is_dir() and ID_PATTERN.fullmatch(path.name):
                    try:
                        voices.append(self.get(path.name))
                    except VoiceError:
                        LOGGER.warning("跳过元数据缺失的音色 %s", path.name)
            return sorted(voices, key=lambda meta: str(meta.get("imported_at", "")), reverse=True)

    def _name(self, desired: str, ignore_id: str | None = None) -> str:
        names = {str(voice.get("name", "")).casefold() for voice in self.list() if voice["id"] != ignore_id}
        name = desired
        count = 2
        while name.casefold() in names:
            suffix = f" ({count})"
            name = desired[: CHAR_LIMIT - len(suffix)].rstrip() + suffix
            count += 1
        return name

    @staticmethod
    def _save(folder: Path, meta: dict) -> None:
        temp = folder / ".meta.json.tmp"
        temp.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temp.replace(folder / "meta.json")

    def add(self, staging: Path, name: str, source: str, index: Path | None = None, verified: dict | None = None) -> dict:
        model = staging / "model.onnx"
        info = verify_model(model) if verified is None else verified
        if index is not None:
            if not 0 < index.stat().st_size or model.stat().st_size + index.stat().st_size >= MAX_BYTES - 65_536:
                raise VoiceError("检索文件为空，或音色包总大小超过 2 GB。")
            shutil.copyfile(index, staging / "model.index")
        has_index = (staging / "model.index").is_file()
        with self._lock:
            new_id = uuid4().hex[:8]
            while (self.root / new_id).exists():
                new_id = uuid4().hex[:8]
            safe_name = clean_name(name.strip()[:CHAR_LIMIT] or "新音色")
            meta = {
                "id": new_id,
                "name": self._name(safe_name),
                "version": info["version"],
                "sr": info["sr"],
                "vec_dim": info["vec_dim"],
                "f0": info["f0"],
                "has_index": has_index,
                "imported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "source_filename": source,
            }
            self._save(staging, meta)
            staging.rename(self.root / new_id)
            return meta

    def rename(self, voice_id: str, name: str) -> dict:
        name = clean_name(name)
        with self._lock:
            meta = self.get(voice_id)
            meta["name"] = self._name(name, voice_id)
            self._save(self.directory(voice_id), meta)
            return meta

    def delete(self, voice_id: str) -> None:
        with self._lock, self.stage() as temp:
            path = self.directory(voice_id)
            path.rename(Path(temp) / voice_id)

    def export(self, voice_id: str, destination: Path) -> None:
        with self._lock:
            folder = self.directory(voice_id)
            if not (folder / "model.onnx").is_file():
                raise VoiceError("该音色的模型文件已被移动或删除，请重新导入。")
            if destination.resolve().is_relative_to(self.root.resolve()):
                raise VoiceError("请选择音色库以外的保存目录。")
            write_package(folder, destination)
