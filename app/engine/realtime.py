from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from app.audio.stream import AudioStreams, SAMPLE_RATE, audio_apartment

from .content_encoder import ContentEncoder
from .f0_rmvpe import HOP, RMVPE_MULTIPLE, RMVPE, f0_to_coarse
from .index import VoiceIndex
from .ort_session import DML, available_providers
from .passthrough import Passthrough
from .resample import InputResampler, OutputResampler
from .rvc_onnx import RvcModel
from .sola import Sola

logger = logging.getLogger(__name__)
LATENCY_FRAMES = {"low": 12, "balanced": 20, "high": 35}
_IO_HOP = SAMPLE_RATE // 100
_HISTORY_FRAMES = 100
_CROSSFADE_FRAMES = 5
_OVERLAP_FRAMES = 4
_SEARCH_FRAMES = 1
_CONTEXT_FRAMES = _HISTORY_FRAMES + _CROSSFADE_FRAMES + _SEARCH_FRAMES
_LOOKAHEAD_FRAMES = _CROSSFADE_FRAMES - _OVERLAP_FRAMES
_EMIT_INTERVAL = 0.1

Emit = Callable[[str, dict], None]


def apply_gate(block: np.ndarray, threshold: float) -> np.ndarray:
    if threshold <= -60:
        return block
    frames = block.reshape(-1, _IO_HOP)
    mean_power = np.mean(frames * frames, axis=1)
    return np.where((mean_power >= 10 ** (threshold / 10))[:, None], frames, 0).reshape(-1)


def _level(samples: np.ndarray) -> float:
    power = float(np.mean(samples * samples))
    return 10 * math.log10(max(power, 1e-9))


class RvcPipeline:
    """Fixed-input RVC blocks shared by the sound card and WAV conversion."""

    def __init__(
        self, assets_dir: Path, voice_dir: Path, voice: dict, block_frames: int,
        gpu_index: int, emit: Emit,
    ) -> None:
        self.block_frames = block_frames
        self.block_size = block_frames * _IO_HOP
        window_frames = block_frames + _CONTEXT_FRAMES
        self._audio = np.zeros(window_frames * HOP, dtype=np.float32)
        self._f0 = np.zeros(window_frames, dtype=np.float32)
        pitch_frames = RMVPE_MULTIPLE * math.ceil((block_frames + 5) / RMVPE_MULTIPLE)
        self._pitch_samples = (pitch_frames - 1) * HOP
        self._encoder = ContentEncoder(assets_dir / "contentvec.onnx", gpu_index, len(self._audio), int(voice["vec_dim"]))
        self._rmvpe = RMVPE(assets_dir, gpu_index, pitch_frames)
        self._voice = RvcModel(voice_dir, voice, gpu_index, window_frames)
        self._input_resampler = InputResampler()
        self._output_resampler = OutputResampler(self._voice.sample_rate)
        self._sola = Sola(self.block_size, _OVERLAP_FRAMES * _IO_HOP, _SEARCH_FRAMES * _IO_HOP)
        self._emit = emit
        self._index = None
        if (voice_dir / "model.index").is_file():
            try:
                self._index = VoiceIndex(voice_dir / "model.index", self._voice.dimension)
            except Exception as exc:
                logger.warning("Voice index could not be loaded", exc_info=True)
                self._emit("warning", {"code": "index_invalid", "message": f"无法加载音色索引，已关闭音色贴合。{exc}"})
        elif voice.get("has_index"):
            self._emit("warning", {"code": "index_missing", "message": "找不到音色索引文件，已关闭音色贴合"})

        sessions = (self._encoder.session, self._rmvpe.session, self._voice.session)
        self.backend = "DirectML" if all(DML in session.get_providers() for session in sessions) else "CPU"
        # Sample age includes the input conversion slack, overlap and lookahead.
        self.delay_ms = self._input_resampler.delay_ms + _CROSSFADE_FRAMES * 10
        self.startup_delay_ms = 0.0

    def _roll_audio(self, block: np.ndarray, threshold: float) -> None:
        new_audio = self._input_resampler.process(apply_gate(block, threshold))
        shift = len(new_audio)
        self._audio[:-shift] = self._audio[shift:]
        self._audio[-shift:] = new_audio

    def _features(self, index_rate: float) -> np.ndarray:
        features = self._encoder.encode(self._audio)
        if self._index is not None and index_rate > 0:
            first = _HISTORY_FRAMES // 2
            try:
                features[first:] = self._index.blend(features[first:], index_rate)
            except Exception as exc:
                self._index = None
                logger.warning("Voice index search failed", exc_info=True)
                self._emit("warning", {"code": "index_invalid", "message": f"音色检索失败，已关闭音色贴合。{exc}"})
        frame_count = len(self._f0)
        frames = np.repeat(features, 2, axis=0)
        if len(frames) < frame_count:
            raise ValueError("内容模型输出帧数不足，请重新安装基础模型")
        return frames[:frame_count]

    def _pitches(self, semitones: float) -> np.ndarray:
        estimate = self._rmvpe.estimate(self._audio[-self._pitch_samples :])
        self._f0[: -self.block_frames] = self._f0[self.block_frames :]
        # The first three estimates have no full STFT history. The last frame
        # depends on reflection at the block edge. RVC keeps the interior.
        stable = estimate[3:-1]
        self._f0[-len(stable) :] = stable
        return self._f0 * np.float32(np.exp2(semitones / 12))

    def process(self, block: np.ndarray, params: dict) -> np.ndarray:
        if block.shape != (self.block_size,):
            raise ValueError("输入音频块长度与延迟档位不匹配")
        self._roll_audio(block, float(params.get("gate_db", -45)))
        features = self._features(float(params.get("index_rate", 0.5)))
        pitches = self._pitches(float(params.get("pitch", 0)))
        generated = self._voice.synthesize(features, pitches, f0_to_coarse(pitches))

        # Extra synthesis on both sides gives the rate converter room for its
        # filter before cutting the exact window expected by SOLA.
        stride = self._voice.stride
        tail_frames = self.block_frames + _OVERLAP_FRAMES + _SEARCH_FRAMES
        filter_history = 2
        resampled = self._output_resampler.process(
            generated[-(tail_frames + _LOOKAHEAD_FRAMES + filter_history) * stride :],
        )
        window = resampled[filter_history * _IO_HOP : -_LOOKAHEAD_FRAMES * _IO_HOP]
        if not np.isfinite(window).all():
            raise ValueError("音色模型生成了无效音频，请尝试重新转换音色或切换到 CPU")
        return np.clip(self._sola.join(window), -1, 1)

    def warmup(self, cancelled: threading.Event, params: dict) -> None:
        silence = np.zeros(self.block_size, dtype=np.float32)
        for _ in range(2):
            if cancelled.is_set():
                break
            started = time.perf_counter()
            self.process(silence, params)
            self.startup_delay_ms = (time.perf_counter() - started) * 1000
        self._input_resampler = InputResampler()
        self._sola = Sola(self.block_size, _OVERLAP_FRAMES * _IO_HOP, _SEARCH_FRAMES * _IO_HOP)
        self._audio.fill(0)
        self._f0.fill(0)


class RealtimeEngine:
    def __init__(self, emit: Emit) -> None:
        self._emit = emit
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._cancelled = threading.Event()
        self._params: dict = {}
        self.state = "idle"
        self.backend: str | None = None
        self.message = ""

    def _status(self, state: str, message: str = "") -> None:
        self.state = state
        self.message = message
        self._emit("status", {"state": state, "message": message, "backend": self.backend})

    def start(
        self, *, assets_dir: Path, voice_dir: Path | None, voice: dict | None,
        devices: dict, params: dict, passthrough: bool = False,
    ) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("引擎正在运行或停止，请稍后重试")
            if not passthrough and (voice_dir is None or voice is None):
                raise ValueError("请先导入并选择一个音色")
            block_frames = LATENCY_FRAMES[params.get("latency_mode", "balanced")]
            self.update_params(params)
            self._cancelled.clear()
            self.backend = None
            self._status("loading", "正在连接音频设备…" if passthrough else "正在加载音色…")
            self._thread = threading.Thread(
                target=self._run, args=(assets_dir, voice_dir, voice, dict(devices), block_frames, passthrough),
                name="maskvox-inference", daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        with self._lock:
            thread = self._thread
            self._cancelled.set()
            if thread is not None and thread.is_alive():
                thread.join(timeout=15)
                if thread.is_alive():
                    raise RuntimeError("正在等待模型退出，请稍后重试")
            self._thread = None
            self.backend = None
            if self.state != "idle":
                self._status("idle")

    def update_params(self, params: dict) -> None:
        # Publish a new object. The worker reads it once for each block.
        self._params = dict(params)

    def _make_pipeline(self, assets_dir, voice_dir, voice, block_frames) -> RvcPipeline:
        gpu_index = int(self._params.get("gpu_index", 0))
        warn = gpu_index >= 0
        if warn and DML in available_providers():
            try:
                pipeline = RvcPipeline(assets_dir, voice_dir, voice, block_frames, gpu_index, self._emit)
                pipeline.warmup(self._cancelled, self._params)
                if pipeline.backend == "DirectML":
                    return pipeline
                # Python ORT may fall back on its own during session creation.
                del pipeline
            except Exception:
                if self._cancelled.is_set():
                    raise
                logger.exception("DirectML warm-up failed, retrying on CPU")

        pipeline = RvcPipeline(assets_dir, voice_dir, voice, block_frames, -1, self._emit)
        pipeline.warmup(self._cancelled, self._params)
        if warn:
            self._emit("warning", {"code": "cpu_fallback", "message": "未启用显卡加速，已使用 CPU 推理"})
        return pipeline

    def _run(self, assets_dir, voice_dir, voice, devices, block_frames, passthrough) -> None:
        error = ""
        try:
            pipeline = Passthrough() if passthrough else self._make_pipeline(assets_dir, voice_dir, voice, block_frames)
            if self._cancelled.is_set():
                return
            with audio_apartment():
                streams = AudioStreams(devices, block_frames * _IO_HOP, pipeline.startup_delay_ms)
                try:
                    streams.start()
                    self.backend = pipeline.backend
                    self._status("running", "音频直通" if passthrough else "正在变声")
                    self._loop(streams, pipeline, block_frames * _IO_HOP)
                finally:
                    streams.close()
        except Exception as exc:
            if not self._cancelled.is_set():
                logger.exception("Realtime audio worker stopped")
                error = f"变声已停止。{exc}"
        finally:
            self.backend = None
            self._status("error" if error else "idle", error)

    def _loop(self, streams: AudioStreams, pipeline: RvcPipeline | Passthrough, block_size: int) -> None:
        block = np.empty(block_size, dtype=np.float32)
        block_ms = block_size / SAMPLE_RATE * 1000
        last_monitor = False
        infer_ms = 0.0
        in_db = out_db = -90.0
        slow_blocks = 0
        warned = False
        next_emit = next_device_check = time.perf_counter()
        while not self._cancelled.is_set():
            now = time.perf_counter()
            params = self._params
            monitor = bool(params.get("monitor", False))
            if monitor != last_monitor:
                last_monitor = monitor
                self._set_monitor(streams, monitor)

            if streams.input.available >= block_size:
                streams.input.trim(block_size * 2)
                streams.input.read_into(block)
                in_db = _level(block)
                now = time.perf_counter()
                output = pipeline.process(block, params)
                infer_ms = (time.perf_counter() - now) * 1000
                streams.output.write(output)
                if streams.monitor_enabled:
                    streams.monitor.write(output * float(params.get("monitor_volume", 0.7)))
                out_db = _level(output)
                slow_blocks = slow_blocks + 1 if infer_ms > block_ms else max(0, slow_blocks - 1)
                if slow_blocks >= 5 and not warned:
                    warned = True
                    self._emit("warning", {
                        "code": "performance",
                        "message": "性能不足，推理耗时持续超过音频块时长，请切换到更高延迟档位",
                    })
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
                    "infer_ms": round(infer_ms, 1),
                    "latency_ms": round(
                        block_ms + max(infer_ms, pipeline.startup_delay_ms) + 10
                        + streams.device_latency_ms + pipeline.delay_ms,
                    ),
                    "underruns": streams.underruns,
                    "input_overflows": streams.input_overflows,
                    "monitor_underruns": streams.monitor_underruns,
                })
                next_emit = now + _EMIT_INTERVAL

    def _set_monitor(self, streams: AudioStreams, enabled: bool) -> None:
        try:
            streams.set_monitor(enabled)
        except Exception as exc:
            logger.warning("Monitor could not be configured", exc_info=True)
            self._emit("warning", {"code": "monitor_unavailable", "message": f"监听设备无法使用，主输出继续运行。{exc}"})
