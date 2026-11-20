from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable

import numpy as np

from app.audio.stream import JITTER_MS, SAMPLE_RATE, AudioStreams, audio_apartment, timer_resolution

from .chain import BLOCK, VoiceChain

logger = logging.getLogger(__name__)
_EMIT_INTERVAL = 0.1
_BLOCK_MS = BLOCK / SAMPLE_RATE * 1000

Emit = Callable[[str, dict], None]


def _level(samples: np.ndarray) -> float:
    power = float(np.mean(samples * samples))
    return 10 * math.log10(max(power, 1e-9))


class RealtimeEngine:
    """Run the voice chain between the microphone and the output devices."""

    def __init__(self, emit: Emit) -> None:
        self._emit = emit
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._cancelled = threading.Event()
        self._params: dict = {}
        self.state = "idle"
        self.message = ""

    def _status(self, state: str, message: str = "") -> None:
        self.state = state
        self.message = message
        self._emit("status", {"state": state, "message": message})

    def start(self, *, devices: dict, params: dict, delay_ms: float = JITTER_MS) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("引擎正在运行或停止，请稍后重试")
            self.update_params(params)
            self._cancelled.clear()
            self._status("loading", "正在连接音频设备…")
            self._thread = threading.Thread(
                target=self._run, args=(dict(devices), delay_ms), name="maskvox-audio", daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        with self._lock:
            thread = self._thread
            self._cancelled.set()
            if thread is not None and thread.is_alive():
                thread.join(timeout=5)
                if thread.is_alive():
                    raise RuntimeError("正在等待音频设备关闭，请稍后重试")
            self._thread = None
            if self.state != "idle":
                self._status("idle")

    def update_params(self, params: dict) -> None:
        # Publish a new object. The worker reads it once for each block.
        self._params = dict(params)

    def _run(self, devices: dict, delay_ms: float) -> None:
        error = ""
        try:
            with audio_apartment(), timer_resolution():
                streams = AudioStreams(devices, BLOCK, delay_ms)
                try:
                    streams.start()
                    self._status("running", "正在变声")
                    self._loop(streams, VoiceChain(), delay_ms)
                finally:
                    streams.close()
        except Exception as exc:
            if not self._cancelled.is_set():
                logger.exception("Realtime audio worker stopped")
                error = f"变声已停止。{exc}"
        finally:
            self._status("error" if error else "idle", error)

    def _loop(self, streams: AudioStreams, chain: VoiceChain, delay_ms: float) -> None:
        block = np.empty(BLOCK, dtype=np.float32)
        monitoring = False
        process_ms = 0.0
        in_db = out_db = -90.0
        slow_blocks = 0
        warned = False
        next_emit = next_device_check = time.perf_counter()
        while not self._cancelled.is_set():
            params = self._params
            monitor = bool(params.get("monitor", False))
            if monitor != monitoring:
                monitoring = monitor
                self._set_monitor(streams, monitor)

            if streams.input.available >= BLOCK:
                streams.input.trim(streams.limit)
                streams.input.read_into(block)
                in_db = _level(block)
                started = time.perf_counter()
                output = chain.process(block, params)
                process_ms = (time.perf_counter() - started) * 1000
                streams.output.write(output)
                if streams.monitor_enabled:
                    streams.monitor.write(output * float(params.get("monitor_volume", 0.6)))
                out_db = _level(output)
                slow_blocks = slow_blocks + 1 if process_ms > _BLOCK_MS else max(0, slow_blocks - 1)
                if slow_blocks >= 10 and not warned:
                    warned = True
                    self._emit("warning", {"code": "performance", "message": "电脑负载过高，声音可能断续"})
                if slow_blocks == 0:
                    warned = False
            else:
                self._cancelled.wait(0.002)

            now = time.perf_counter()
            if now >= next_device_check:
                streams.check_active()
                if streams.monitor_enabled and not streams.monitor_active():
                    self._set_monitor(streams, False)
                    self._emit("warning", {"code": "monitor_unavailable", "message": "监听设备已断开，主输出继续运行"})
                next_device_check = now + 0.5
            if now >= next_emit:
                self._emit("meter", {
                    "in_db": round(in_db, 1), "out_db": round(out_db, 1),
                    "process_ms": round(process_ms, 2),
                    "latency_ms": round(_BLOCK_MS + delay_ms + chain.delay_ms + streams.device_latency_ms),
                    "underruns": streams.underruns,
                })
                next_emit = now + _EMIT_INTERVAL

    def _set_monitor(self, streams: AudioStreams, enabled: bool) -> None:
        try:
            streams.set_monitor(enabled)
        except Exception as exc:
            logger.warning("Monitor could not be configured", exc_info=True)
            self._emit("warning", {"code": "monitor_unavailable", "message": f"监听设备无法使用，主输出继续运行。{exc}"})
