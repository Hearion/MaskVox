from __future__ import annotations

import ctypes
from contextlib import ExitStack, contextmanager

import numpy as np
import sounddevice as sd

from .ringbuffer import RingBuffer

SAMPLE_RATE = 48000
# Extra queued output that absorbs scheduling and inference jitter.
JITTER_MS = 30


@contextmanager
def audio_apartment():
    """WASAPI calls made by a Python worker require COM on that worker."""
    ole32 = ctypes.WinDLL("ole32")
    result = ole32.CoInitializeEx(None, 2)  # COINIT_APARTMENTTHREADED
    changed_mode = ctypes.c_long(0x80010106).value
    if result < 0 and result != changed_mode:
        raise RuntimeError(f"Windows 音频初始化失败（0x{result & 0xFFFFFFFF:08X}）")
    try:
        yield
    finally:
        if result >= 0:
            ole32.CoUninitialize()


@contextmanager
def timer_resolution():
    """Give the worker's short waits millisecond accuracy on Windows."""
    winmm = ctypes.WinDLL("winmm")
    raised = winmm.timeBeginPeriod(1) == 0
    try:
        yield
    finally:
        if raised:
            winmm.timeEndPeriod(1)


class AudioStreams:
    """Own one input and separate output clocks, with copy-only callbacks."""

    def __init__(self, devices: dict, block_size: int, startup_delay_ms: float = 0) -> None:
        self._devices = devices
        self._limit = block_size * 2
        self._silence = np.zeros(block_size, dtype=np.float32)
        prefill = block_size + round(SAMPLE_RATE * (startup_delay_ms + JITTER_MS) / 1000)
        self._startup_silence = np.zeros(min(self._limit, prefill), dtype=np.float32)
        self.input = RingBuffer(block_size * 4)
        self.output = RingBuffer(block_size * 4)
        self.monitor = RingBuffer(block_size * 4)
        self.input_overflows = 0
        self.underruns = 0
        self.monitor_underruns = 0
        self.monitor_enabled = False
        self.device_latency_ms = 0.0
        self._main_stack = ExitStack()
        self._monitor_stack = ExitStack()
        self._input_stream = None
        self._output_stream = None
        self._monitor_stream = None

    def _capture(self, data, frames, time_info, status) -> None:
        accepted = self.input.write(data[:, 0])
        if accepted < frames or status.input_overflow:
            self.input_overflows += 1

    def _render(self, data, frames, time_info, status) -> None:
        self.output.trim(self._limit)
        filled = self.output.read_into(data[:, 0])
        if filled < frames or status.output_underflow:
            self.underruns += 1
        if data.shape[1] > 1:
            data[:, 1] = data[:, 0]

    def _render_monitor(self, data, frames, time_info, status) -> None:
        self.monitor.trim(self._limit)
        filled = self.monitor.read_into(data[:, 0])
        if not self.monitor_enabled:
            data.fill(0)
            return
        if filled < frames or status.output_underflow:
            self.monitor_underruns += 1
        if data.shape[1] > 1:
            data[:, 1] = data[:, 0]

    def _open(self, stack: ExitStack, device: int | None, *, capture: bool, callback, label: str):
        stream_cls = sd.InputStream if capture else sd.OutputStream
        settings = sd.WasapiSettings(auto_convert=True)
        if device is None:
            raise RuntimeError(f"请先选择{label}")
        try:
            info = sd.query_devices(device, "input" if capture else "output")
            if sd.query_hostapis(info["hostapi"])["name"] != "Windows WASAPI":
                raise ValueError("设备已变化，请刷新设备列表后重新选择")
            channels_key = "max_input_channels" if capture else "max_output_channels"
            for channels in range(1, min(info[channels_key], 2) + 1):
                stream = None
                try:
                    stream = stream_cls(
                        device=device,
                        channels=channels,
                        samplerate=SAMPLE_RATE,
                        dtype="float32",
                        blocksize=0,
                        latency="low",
                        extra_settings=settings,
                        callback=callback,
                    )
                    stream.start()
                except sd.PortAudioError:
                    if stream is not None:
                        stream.close()
                    if channels < min(info[channels_key], 2):
                        continue
                    raise
                stack.callback(stream.close)
                return stream
        except (sd.PortAudioError, ValueError) as exc:
            raise RuntimeError(f"无法打开{label}。检查设备是否已拔出或被其他应用独占。{exc}") from exc
        raise RuntimeError(f"无法打开{label}。设备没有可用的声道。")

    def start(self) -> None:
        self.output.write(self._startup_silence)
        self._input_stream = self._open(
            self._main_stack, self._devices.get("input_id"), capture=True,
            callback=self._capture, label="麦克风",
        )
        self._output_stream = self._open(
            self._main_stack, self._devices.get("output_id"), capture=False,
            callback=self._render, label="输出设备",
        )
        self.device_latency_ms = (self._input_stream.latency + self._output_stream.latency) * 1000

    def set_monitor(self, enabled: bool) -> None:
        if self.monitor_enabled == enabled:
            return
        if not enabled:
            self.monitor_enabled = False
            self._monitor_stack.close()
            self._monitor_stream = None
            self.monitor.trim(0)
            return

        if self._devices.get("monitor_id") == self._devices.get("output_id"):
            raise RuntimeError("监听设备和主输出必须使用不同的设备")
        self.monitor.write(self._silence)
        self.monitor_enabled = True
        try:
            self._monitor_stream = self._open(
                self._monitor_stack, self._devices.get("monitor_id"), capture=False,
                callback=self._render_monitor, label="监听设备",
            )
        except Exception:
            self.set_monitor(False)
            raise

    def check_active(self) -> None:
        if not self._input_stream.active or not self._output_stream.active:
            raise RuntimeError("麦克风或输出设备已断开，请刷新设备列表后重新启动")

    def monitor_active(self) -> bool:
        return self._monitor_stream is not None and self._monitor_stream.active

    def close(self) -> None:
        try:
            self._monitor_stack.close()
        finally:
            self._main_stack.close()
