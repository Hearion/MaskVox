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
BUILTIN_VOICE_IDS = ("b0010001", "b0010002")


def clean_name(name: str) -> str:
    if not isinstance(name, str):
        raise VoiceError("请输入有效的音色名称。")
    name = " ".join(name.split())
    if not name or len(name) > CHAR_LIMIT or any(ord(c) < 32 for c in name):
        raise VoiceError("音色名称长度应为 1～80 个字符。")
    return name


class VoiceLibrary:
    def __init__(self, root: Path, builtin_root: Path | None = None) -> None:
        self.root = root
        self._builtin_root = builtin_root
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()

    def stage(self) -> tempfile.TemporaryDirectory:
        return tempfile.TemporaryDirectory(prefix=".import-", dir=self.root)

    def _locate(self, voice_id: str) -> tuple[Path, bool]:
        if not isinstance(voice_id, str) or not ID_PATTERN.fullmatch(voice_id):
            raise VoiceError("找不到指定的音色。")
        path = self.root / voice_id
        if path.is_dir() and path.resolve().parent == self.root.resolve():
            return path, False
        if self._builtin_root is not None and voice_id in BUILTIN_VOICE_IDS:
            path = self._builtin_root / voice_id
            if path.is_dir() and path.resolve().parent == self._builtin_root.resolve():
                return path, True
        raise VoiceError("该音色已被删除或移动，请重新导入。")

    def directory(self, voice_id: str) -> Path:
        return self._locate(voice_id)[0]

    def preview(self, voice_id: str) -> Path:
        folder, builtin = self._locate(voice_id)
        sample = folder / "preview.wav"
        if not builtin or not sample.is_file() or sample.resolve().parent != folder.resolve():
            raise VoiceError("当前音色没有内置试听样本。")
        return sample

    def get(self, voice_id: str) -> dict:
        with self._lock:
            folder, builtin = self._locate(voice_id)
            try:
                meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
                if not isinstance(meta, dict) or meta.get("id") != voice_id:
                    raise ValueError
                if builtin:
                    if meta.get("gender") not in {"male", "female"} or not (folder / "model.onnx").is_file():
                        raise ValueError
                    return {
                        **meta, "builtin": True, "exportable": bool(meta.get("attribution")),
                        "preview_ready": (folder / "preview.wav").is_file(),
                    }
                return {**meta, "builtin": False}
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
            voices.sort(key=lambda meta: str(meta.get("imported_at", "")), reverse=True)
            if self._builtin_root is not None:
                occupied_ids = {voice["id"] for voice in voices}
                for voice_id in BUILTIN_VOICE_IDS:
                    if voice_id in occupied_ids or not (self._builtin_root / voice_id).is_dir():
                        continue
                    try:
                        voices.append(self.get(voice_id))
                    except VoiceError:
                        LOGGER.warning("跳过文件缺失的内置音色 %s", voice_id)
            return voices

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
            while (self.root / new_id).exists() or new_id in BUILTIN_VOICE_IDS:
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
            if "attribution" in info:
                meta["attribution"] = info["attribution"]
            self._save(staging, meta)
            staging.rename(self.root / new_id)
            return {**meta, "builtin": False}

    def rename(self, voice_id: str, name: str) -> dict:
        name = clean_name(name)
        with self._lock:
            meta = self.get(voice_id)
            if meta["builtin"]:
                raise VoiceError("内置音色的名称固定。")
            meta["name"] = self._name(name, voice_id)
            del meta["builtin"]
            self._save(self.directory(voice_id), meta)
            return meta

    def delete(self, voice_id: str) -> None:
        with self._lock:
            path, builtin = self._locate(voice_id)
            if builtin:
                raise VoiceError("内置音色会始终保留在音色库。")
            with self.stage() as temp:
                path.rename(Path(temp) / voice_id)

    def export(self, voice_id: str, destination: Path) -> None:
        with self._lock:
            if self.get(voice_id).get("exportable") is False:
                raise VoiceError("当前音色缺少导出所需的署名信息。")
            folder = self.directory(voice_id)
            if not (folder / "model.onnx").is_file():
                raise VoiceError("该音色的模型文件已被移动或删除，请重新导入。")
            dest = destination.resolve()
            if dest.is_relative_to(self.root.resolve()) or (
                self._builtin_root is not None and dest.is_relative_to(self._builtin_root.resolve())
            ):
                raise VoiceError("请选择音色库以外的保存目录。")
            write_package(folder, destination)
