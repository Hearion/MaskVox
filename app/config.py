"""Shared paths and per-user settings."""

from __future__ import annotations

import json
import logging
import math
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from threading import RLock
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("MASKVOX_DATA_DIR", ROOT)).expanduser().resolve()
ASSETS_DIR = ROOT / "assets"
BUILTIN_VOICES_DIR = ASSETS_DIR / "builtin-voices"
VOICES_DIR = DATA_DIR / "voices"
LOGS_DIR = DATA_DIR / "logs"
SETTINGS_PATH = DATA_DIR / "settings.json"
CONVERTER_DIR = ROOT / "converter"
CONVERTER_PYTHON = CONVERTER_DIR / ".venv" / (
    "Scripts/python.exe" if os.name == "nt" else "bin/python"
)
UI_PATH = ROOT / "app" / "ui" / "index.html"
BASE_MODELS = ("contentvec.onnx", "rmvpe.onnx", "rmvpe_mel.npz")
VERSION = "1.1.0"
VB_CABLE_URL = "https://vb-audio.com/Cable/"
VB_CABLE_SUPPORT_URL = "https://shop.vb-audio.com/en/win-apps/11-vb-cable.html"
WEBVIEW_URL = "https://developer.microsoft.com/microsoft-edge/webview2/"

DEFAULT_SETTINGS: dict[str, Any] = {
    "input_id": None,
    "output_id": None,
    "monitor_id": None,
    "input_name": None,
    "output_name": None,
    "output_preferred": False,
    "monitor_name": None,
    "selected_voice_id": None,
    "pitch": 0,
    "index_rate": 0.5,
    "gate_db": -45,
    "latency_mode": "balanced",
    "gpu_index": 0,
    "monitor": False,
    "monitor_volume": 0.6,
    "onboarding_dismissed": False,
}
LOGGER = logging.getLogger(__name__)


def _valid_setting(key: str, value: Any) -> bool:
    if key == "gpu_index":
        return type(value) is int and -1 <= value <= 64
    if key == "latency_mode":
        return value in ("low", "balanced", "high")
    if key == "selected_voice_id" or key.endswith("_name"):
        return value is None or isinstance(value, str)
    if key.endswith("_id"):
        return value is None or (type(value) is int and value >= 0)
    if type(DEFAULT_SETTINGS[key]) is bool:
        return type(value) is bool
    limits = {"pitch": (-12, 12), "index_rate": (0, 1), "gate_db": (-60, -20), "monitor_volume": (0, 1)}
    minimum, maximum = limits[key]
    return type(value) in (int, float) and minimum <= value <= maximum and math.isfinite(value)


def base_models_ready() -> bool:
    return all((ASSETS_DIR / name).is_file() for name in BASE_MODELS)


def setup_logging() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        LOGS_DIR / "maskvox.log", maxBytes=3_000_000, backupCount=3, encoding="utf-8"
    )
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)


class Settings:
    def __init__(self, path: Path = SETTINGS_PATH) -> None:
        self._path = path
        self._lock = RLock()
        self._data = DEFAULT_SETTINGS.copy()
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                raise ValueError("Settings must be an object")
            self._data.update({
                key: loaded[key] for key in self._data if key in loaded and _valid_setting(key, loaded[key])
            })
        except FileNotFoundError:
            pass
        except (OSError, ValueError):
            LOGGER.exception("读取设置失败，已使用默认值")

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self._data.copy()

    def update(self, **changes: Any) -> dict[str, Any]:
        with self._lock:
            data = {**self._data, **changes}
            if data != self._data:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                temp = self._path.with_suffix(".json.tmp")
                temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                temp.replace(self._path)
                self._data = data
            return self._data.copy()
