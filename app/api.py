"""Small, validated Python API for the WebView2 interface."""

from __future__ import annotations

import logging
import math
import os
import webbrowser
from functools import wraps
from threading import RLock
from typing import Any

from . import cable_setup, config
from .audio.devices import list_devices, refresh_devices
from .engine.presets import ADJUSTABLE, BY_ID, PRESETS, chain_params
from .engine.realtime import RealtimeEngine
from .events import Events

LOGGER = logging.getLogger(__name__)
DEVICE_SETTINGS = {"input_id": "inputs", "output_id": "outputs", "monitor_id": "outputs"}
NUMERIC_PARAMETERS = {**ADJUSTABLE, "gate_db": (-60, -20, 1), "monitor_volume": (0, 1, 0.01)}
ACTIVE = {"loading", "running"}


class ApiError(ValueError):
    """A failure whose message is shown to the user."""


def response(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        try:
            return {"ok": True, "data": method(self, *args, **kwargs)}
        except ValueError as exc:
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
        self._engine = RealtimeEngine(events.emit)
        self._lock = RLock()
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
        self._reconcile_devices()

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

    def _params(self, settings: dict) -> dict:
        return {**chain_params(settings), "monitor": settings["monitor"], "monitor_volume": settings["monitor_volume"]}

    def _start_locked(self, settings: dict) -> None:
        if self._closing:
            raise ApiError("MaskVox 正在退出。")
        if settings["input_id"] is None or settings["output_id"] is None:
            raise ApiError("请选择麦克风和输出设备，然后重试。")
        devices = {key: settings[key] for key in DEVICE_SETTINGS}
        self._engine.start(devices=devices, params=self._params(settings))

    def _replace_settings(self, restart: bool = False, **changes: Any) -> dict:
        before = self._settings.snapshot()
        if all(before.get(key) == value for key, value in changes.items()):
            return before
        active = self._engine.state in ACTIVE
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
            self._engine.update_params(self._params(updated))
        return updated

    @response
    def get_state(self) -> dict:
        with self._lock:
            return {
                "settings": self._settings.snapshot(),
                "presets": [
                    {key: preset[key] for key in ("id", "name", "hint", "icon", "hue")} for preset in PRESETS
                ],
                "devices": self._devices,
                "engine": {"state": self._engine.state, "message": self._engine.message},
                "has_vb_cable": self._devices["has_vb_cable"],
                "vb_cable_installer_ready": cable_setup.is_installer_available(),
                "logs_path": str(config.LOGS_DIR),
                "version": config.VERSION,
            }

    @response
    def list_devices(self) -> dict:
        with self._lock:
            if self._engine.state in ACTIVE:
                raise ApiError("请先停止变声，再刷新音频设备。")
            self._devices = refresh_devices()
            self._reconcile_devices()
            return {"devices": self._devices, "settings": self._settings.snapshot()}

    @response
    def set_devices(self, selection: dict) -> dict:
        with self._lock:
            if not isinstance(selection, dict) or not selection.keys() <= DEVICE_SETTINGS.keys():
                raise ApiError("无效的设备设置。")
            changes = {}
            settings = self._settings.snapshot()
            for key, device_id in selection.items():
                item = None
                if device_id is not None:
                    if type(device_id) is not int:
                        raise ApiError("请选择有效设备。")
                    item = next((device for device in self._devices[DEVICE_SETTINGS[key]] if device["id"] == device_id), None)
                    if item is None:
                        raise ApiError("该音频设备不可用，请刷新设备列表。")
                changes[key] = device_id
                changes[key.replace("_id", "_name")] = item["name"] if item else None
                if key == "output_id":
                    changes["output_preferred"] = device_id is not None
            output = changes.get("output_id", settings["output_id"])
            monitor = changes.get("monitor_id", settings["monitor_id"])
            if settings["monitor"] and (monitor is None or monitor == output):
                raise ApiError("自己监听需要选择与主输出不同的设备。")
            if self._engine.state in ACTIVE and any(
                changes.get(key, settings[key]) is None for key in ("input_id", "output_id")
            ):
                raise ApiError("变声运行时需要保留麦克风和主输出设备。")
            return self._replace_settings(restart=True, **changes)

    @response
    def select_preset(self, preset_id: str) -> dict:
        """Switch effect and load its own slider values."""
        with self._lock:
            if preset_id not in BY_ID:
                raise ApiError("没有这个音效。")
            values = {key: BY_ID[preset_id]["params"][key] for key in ADJUSTABLE}
            return self._replace_settings(preset=preset_id, **values)

    @response
    def set_param(self, name: str, value: Any) -> dict:
        with self._lock:
            if name in NUMERIC_PARAMETERS:
                low, high, step = NUMERIC_PARAMETERS[name]
                if type(value) not in {int, float} or not math.isfinite(value) or not low <= value <= high:
                    raise ApiError("调节值超出范围。")
                value = round(low + round((value - low) / step) * step, 2)
            elif name == "monitor":
                if type(value) is not bool:
                    raise ApiError("监听开关的值无效。")
                settings = self._settings.snapshot()
                if value and (settings["monitor_id"] is None or settings["monitor_id"] == settings["output_id"]):
                    raise ApiError("请先选择与主输出不同的监听设备。")
            else:
                raise ApiError("不支持该设置项。")
            return self._replace_settings(**{name: value})

    @response
    def dismiss_onboarding(self) -> dict:
        return self._settings.update(onboarding_dismissed=True)

    @response
    def start(self) -> None:
        with self._lock:
            if self._engine.state not in ACTIVE:
                self._start_locked(self._settings.snapshot())

    @response
    def stop(self) -> None:
        with self._lock:
            self._engine.stop()

    @response
    def install_vb_cable(self) -> None:
        with self._lock:
            if self._closing or self._engine.state in ACTIVE:
                raise ApiError("请先停止变声，再安装 VB-Cable。")
            if self._devices["has_vb_cable"]:
                raise ApiError("系统已检测到 VB-Cable。")
        try:
            cable_setup.launch_installer()
        except RuntimeError as exc:
            raise ApiError(str(exc)) from exc

    @response
    def open_url(self, url: str) -> None:
        if url not in {config.VB_CABLE_URL, config.VB_CABLE_SUPPORT_URL, config.WEBVIEW_URL}:
            raise ApiError("该链接不可用。")
        webbrowser.open(url)

    @response
    def open_logs(self) -> None:
        os.startfile(config.LOGS_DIR)

    def _close(self) -> None:
        with self._lock:
            self._closing = True
            self._events.close()
        self._engine.stop()
