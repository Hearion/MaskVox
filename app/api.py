"""Small, validated Python API for the WebView2 interface."""

from __future__ import annotations

import logging
import math
import os
import webbrowser
from functools import wraps
from pathlib import Path
from threading import RLock
from typing import Any

from . import cable_setup, config
from .audio.devices import list_devices, refresh_devices
from .engine.ort_session import list_adapters
from .engine.realtime import RealtimeEngine
from .events import Events
from .voices.converter_client import installed as converter_installed
from .voices.imports import Imports
from .voices.library import VoiceLibrary
from .voices.package import VoiceError

LOGGER = logging.getLogger(__name__)
WINDOW_FILTERS = ("MaskVox 音色 (*.pth;*.index;*.zip;*.maskvox)",)
DEVICE_SETTINGS = {"input_id": "inputs", "output_id": "outputs", "monitor_id": "outputs"}
NUMERIC_PARAMETERS = {
    "pitch": (-12, 12, 1),
    "index_rate": (0, 1, 0.05),
    "gate_db": (-60, -20, 1),
    "monitor_volume": (0, 1, 0.01),
}


def response(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        try:
            return {"ok": True, "data": method(self, *args, **kwargs)}
        except (ValueError, VoiceError) as exc:
            LOGGER.info("%s: %s", method.__name__, exc)
            return {"ok": False, "error": str(exc)}
        except Exception:
            LOGGER.exception("%s 操作失败", method.__name__)
            return {"ok": False, "error": "操作失败。请查看日志。"}
    return wrapped


class Api:
    def __init__(self, events: Events) -> None:
        self._window: Any = None
        self._events = events
        self._settings = config.Settings()
        self._library = VoiceLibrary(config.VOICES_DIR, config.BUILTIN_VOICES_DIR)
        self._engine = RealtimeEngine(events.emit)
        self._lock = RLock()
        self._imports = Imports(self._library, events.emit, self._select_imported)
        self._closing = False
        try:
            self._devices = list_devices()
        except Exception:
            LOGGER.exception("初始化音频设备列表失败")
            self._devices = {
                "inputs": [], "outputs": [], "default_input_id": None, "default_output_id": None,
                "cable_output_id": None, "has_vb_cable": False,
            }
            self._events.emit("warning", {"code": "device_list_failed", "message": "无法读取音频设备，请尝试刷新。"})
        self._adapters = list_adapters()
        self._reconcile_devices()
        self._reconcile_voice()

    def _reconcile_voice(self) -> None:
        selected = self._settings.snapshot()["selected_voice_id"]
        if selected:
            try:
                self._library.get(selected)
                if (self._library.directory(selected) / "model.onnx").is_file():
                    return
            except VoiceError:
                LOGGER.info("上次选择的音色已不可用，尝试使用其它音色")
        voices = self._library.list()
        self._settings.update(selected_voice_id=voices[0]["id"] if voices else None)

    def _reconcile_devices(self) -> None:
        settings = self._settings.snapshot()
        defaults = {
            "input_id": self._devices.get("default_input_id"),
            "output_id": self._devices.get("cable_output_id"),
            "monitor_id": self._devices.get("default_output_id"),
        }
        if defaults["output_id"] is None:
            headset = next(
                (item["id"] for item in self._devices["outputs"]
                 if any(name in item["name"].casefold() for name in ("headphone", "headset", "耳机", "耳機"))),
                None,
            )
            defaults["output_id"] = headset if headset is not None else self._devices.get("default_output_id")
        changes = {}
        for key, group in DEVICE_SETTINGS.items():
            name_key = key.replace("_id", "_name")
            items = self._devices[group]
            old_id = settings[key]
            old_name = settings[name_key]
            item = next(
                (device for device in items if device["id"] == old_id and device["name"] == old_name),
                None,
            )
            if item is None and old_name:
                item = next((device for device in items if device["name"] == old_name), None)
            if item is None and old_name is None:
                item = next((device for device in items if device["id"] == old_id), None)
            if item is None or (key == "output_id" and not settings["output_preferred"]):
                item = next((device for device in items if device["id"] == defaults[key]), None)
            if key == "monitor_id" and (item is None or item["id"] == changes.get("output_id")):
                item = next(
                    (device for device in items if device["id"] != changes.get("output_id")
                     and device["id"] != self._devices.get("cable_output_id")),
                    None,
                )
            changes[key] = item["id"] if item is not None else None
            changes[name_key] = item["name"] if item is not None else None
        if changes["monitor_id"] is None:
            changes["monitor"] = False
        self._settings.update(**changes)

    def _start_locked(self, settings: dict) -> None:
        if self._closing:
            raise VoiceError("MaskVox 正在退出。")
        voice_id = settings["selected_voice_id"]
        if not voice_id:
            raise VoiceError("请先导入并选择一个音色。")
        voice = self._library.get(voice_id)
        folder = self._library.directory(voice_id)
        if not (folder / "model.onnx").is_file():
            raise VoiceError("当前音色的模型文件已被删除或移动，请重新导入。")
        if not config.base_models_ready():
            raise VoiceError("缺少基础模型。请运行 install_converter.bat 安装，或使用完整的便携版。")
        if settings["input_id"] is None or settings["output_id"] is None:
            raise VoiceError("请选择麦克风和输出设备，然后重试。")
        devices = {key: settings[key] for key in DEVICE_SETTINGS}
        self._engine.start(
            assets_dir=config.ASSETS_DIR, voice_dir=folder, voice=voice,
            devices=devices, params=settings,
        )

    def _replace_settings(self, restart: bool = False, **changes: Any) -> dict:
        before = self._settings.snapshot()
        if all(before.get(key) == value for key, value in changes.items()):
            return before
        active = self._engine.state in {"loading", "running"}
        if restart and active:
            self._engine.stop()
            try:
                updated = self._settings.update(**changes)
                self._start_locked(updated)
            except Exception:
                try:
                    self._settings.update(**{key: before[key] for key in changes})
                    self._start_locked(before)
                except Exception:
                    LOGGER.exception("恢复之前的变声设置失败")
                raise
            return updated
        updated = self._settings.update(**changes)
        if active:
            self._engine.update_params(updated)
        return updated

    def _select_imported(self, voice_id: str) -> None:
        with self._lock:
            if self._closing:
                return
            try:
                self._replace_settings(restart=True, selected_voice_id=voice_id)
            except Exception as exc:
                LOGGER.exception("新音色已入库，切换引擎失败")
                message = str(exc) if isinstance(exc, VoiceError) else "新音色已保存，请重新启动变声。"
                self._events.emit("warning", {"code": "voice_restart_failed", "message": message})
            self._events.emit("voices_changed", {})

    @response
    def get_state(self) -> dict:
        with self._lock:
            return {
                "settings": self._settings.snapshot(),
                "voices": self._library.list(),
                "devices": self._devices,
                "engine": {
                    "state": self._engine.state, "message": self._engine.message,
                    "backend": self._engine.backend,
                },
                "has_vb_cable": self._devices["has_vb_cable"],
                "vb_cable_installer_ready": cable_setup.is_installer_available(),
                "base_models_ready": config.base_models_ready(),
                "converter_installed": converter_installed(),
                "gpu_adapters": self._adapters,
                "logs_path": str(config.LOGS_DIR),
                "version": config.VERSION,
            }

    @response
    def list_devices(self) -> dict:
        with self._lock:
            if self._engine.state in {"loading", "running"}:
                raise VoiceError("请先停止变声，再刷新音频设备。")
            self._devices = refresh_devices()
            self._reconcile_devices()
            return self._devices

    @response
    def set_devices(self, selection: dict) -> dict:
        with self._lock:
            if not isinstance(selection, dict) or not selection.keys() <= DEVICE_SETTINGS.keys():
                raise VoiceError("无效的设备设置。")
            changes = {}
            settings = self._settings.snapshot()
            for key, device_id in selection.items():
                item = None
                if device_id is not None:
                    if type(device_id) is not int:
                        raise VoiceError("请选择有效设备。")
                    item = next((device for device in self._devices[DEVICE_SETTINGS[key]] if device["id"] == device_id), None)
                    if item is None:
                        raise VoiceError("该音频设备不可用，请刷新设备列表。")
                changes[key] = device_id
                changes[key.replace("_id", "_name")] = item["name"] if item else None
                if key == "output_id":
                    changes["output_preferred"] = device_id is not None
            output = changes.get("output_id", settings["output_id"])
            monitor = changes.get("monitor_id", settings["monitor_id"])
            if settings["monitor"] and (monitor is None or monitor == output):
                raise VoiceError("自己监听需要选择与主输出不同的设备。")
            if self._engine.state in {"loading", "running"} and any(
                changes.get(key, settings[key]) is None for key in ("input_id", "output_id")
            ):
                raise VoiceError("变声运行时需要保留麦克风和主输出设备。")
            return self._replace_settings(restart=True, **changes)

    @response
    def set_param(self, name: str, value: Any) -> dict:
        with self._lock:
            if name in NUMERIC_PARAMETERS:
                low, high, step = NUMERIC_PARAMETERS[name]
                if type(value) not in {int, float} or not math.isfinite(value) or not low <= value <= high:
                    raise VoiceError("调节值超出范围。")
                value = round(low + round((value - low) / step) * step, 2)
            elif name == "latency_mode":
                if value not in ("low", "balanced", "high"):
                    raise VoiceError("请选择一个延迟档位。")
            elif name == "gpu_index":
                if type(value) is not int or value not in {-1, 0, *(adapter["id"] for adapter in self._adapters)}:
                    raise VoiceError("请选择可用的显卡或 CPU。")
            elif name == "monitor":
                if type(value) is not bool:
                    raise VoiceError("监听开关的值无效。")
                settings = self._settings.snapshot()
                if value and (settings["monitor_id"] is None or settings["monitor_id"] == settings["output_id"]):
                    raise VoiceError("请先选择与主输出不同的监听设备。")
            else:
                raise VoiceError("不支持该设置项。")
            return self._replace_settings(restart=name in {"latency_mode", "gpu_index"}, **{name: value})

    @response
    def dismiss_onboarding(self) -> dict:
        return self._settings.update(onboarding_dismissed=True)

    @response
    def list_voices(self) -> list[dict]:
        return self._library.list()

    @response
    def select_voice(self, voice_id: str) -> dict:
        with self._lock:
            self._library.get(voice_id)
            if not (self._library.directory(voice_id) / "model.onnx").is_file():
                raise VoiceError("该音色的模型文件已被删除或移动，请重新导入。")
            return self._replace_settings(restart=True, selected_voice_id=voice_id)

    @response
    def rename_voice(self, voice_id: str, name: str) -> dict:
        voice = self._library.rename(voice_id, name)
        self._events.emit("voices_changed", {})
        return voice

    @response
    def delete_voice(self, voice_id: str) -> dict:
        with self._lock:
            if self._library.get(voice_id)["builtin"]:
                raise VoiceError("内置音色会始终保留在音色库。")
            selected = self._settings.snapshot()["selected_voice_id"] == voice_id
            if selected:
                self._engine.stop()
            self._library.delete(voice_id)
            if selected:
                remaining = self._library.list()
                self._settings.update(selected_voice_id=remaining[0]["id"] if remaining else None)
            self._events.emit("voices_changed", {})
            return self._settings.snapshot()

    @response
    def pick_and_import_voices(self) -> dict | None:
        import webview

        paths = self._window.create_file_dialog(webview.OPEN_DIALOG, allow_multiple=True, file_types=WINDOW_FILTERS)
        return {"job_id": self._imports.start(list(paths))} if paths else None

    @response
    def import_voice_paths(self, paths: list[str]) -> dict:
        return {"job_id": self._imports.start(paths)}

    @response
    def cancel_import(self, job_id: str) -> None:
        self._imports.cancel(job_id)

    @response
    def export_voice(self, voice_id: str) -> str | None:
        import webview

        voice = self._library.get(voice_id)
        name = "".join("_" if character in '\\/:*?"<>|' else character for character in voice["name"])
        result = self._window.create_file_dialog(
            webview.SAVE_DIALOG, save_filename=f"{name}.maskvox", file_types=("MaskVox 音色包 (*.maskvox)",)
        )
        if not result:
            return None
        path = Path(result if isinstance(result, str) else result[0])
        if path.suffix.lower() != ".maskvox":
            path = path.with_suffix(".maskvox")
        self._library.export(voice_id, path)
        return str(path)

    @response
    def start(self) -> None:
        with self._lock:
            if self._engine.state not in {"loading", "running"}:
                self._start_locked(self._settings.snapshot())

    @response
    def stop(self) -> None:
        with self._lock:
            self._engine.stop()

    @response
    def preview_voice(self) -> dict:
        with self._lock:
            if self._engine.state in {"loading", "running"}:
                raise VoiceError("请先停止变声，再试听音色样本。")
            voice_id = self._settings.snapshot()["selected_voice_id"]
            if voice_id is None:
                raise VoiceError("请选择一个内置音色。")
            return {"url": self._library.preview(voice_id).resolve().as_uri()}

    @response
    def install_vb_cable(self) -> None:
        with self._lock:
            if self._closing or self._engine.state in {"loading", "running"}:
                raise VoiceError("请先停止变声，再安装 VB-Cable。")
            if self._devices["has_vb_cable"]:
                raise VoiceError("系统已检测到 VB-Cable。")
        try:
            cable_setup.launch_installer()
        except RuntimeError as exc:
            raise VoiceError(str(exc)) from exc

    @response
    def open_url(self, url: str) -> None:
        if url not in {config.VB_CABLE_URL, config.VB_CABLE_SUPPORT_URL, config.WEBVIEW_URL}:
            raise VoiceError("该链接不可用。")
        webbrowser.open(url)

    @response
    def open_logs(self) -> None:
        os.startfile(config.LOGS_DIR)

    @response
    def open_install_guide(self) -> None:
        # Show the bundled Markdown in a browser if no Markdown editor is associated.
        readme = config.ROOT / "README.md"
        try:
            os.startfile(readme)
        except OSError:
            webbrowser.open(readme.as_uri())

    def _on_drop(self, event: dict) -> None:
        items = event.get("dataTransfer", {}).get("files", [])
        paths = [item["pywebviewFullPath"] for item in items if item.get("pywebviewFullPath")]
        if not paths:
            self._events.emit("warning", {"code": "drop_unavailable", "message": "无法读取拖入的路径，请点击「导入音色」。"})
            return
        result = self.import_voice_paths(paths)
        if not result["ok"]:
            self._events.emit("warning", {"code": "drop_failed", "message": result["error"]})

    def _close(self) -> None:
        with self._lock:
            self._closing = True
            self._events.close()
        self._imports.close()
        self._engine.stop()
