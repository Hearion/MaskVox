"""Convert a PCM WAV through MaskVox's real-time RVC blocks."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import tempfile
import wave
from pathlib import Path
from threading import Event

import numpy as np
import soxr

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.audio.stream import SAMPLE_RATE
from app.config import ASSETS_DIR, VOICES_DIR, setup_logging
from app.engine.ort_session import DML, available_providers
from app.engine.realtime import LATENCY_FRAMES, RvcPipeline

logger = logging.getLogger(__name__)


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as source:
        rate = source.getframerate()
        channels = source.getnchannels()
        frames = source.readframes(source.getnframes())
        width = source.getsampwidth()
    if width == 1:
        samples = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128) / 128
    elif width in (2, 4):
        samples = np.frombuffer(frames, dtype=f"<i{width}").astype(np.float32) / np.float32(2 ** (8 * width - 1))
    elif width == 3:
        packed = np.frombuffer(frames, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        signed = ((packed[:, 2] << 24) | (packed[:, 1] << 16) | (packed[:, 0] << 8)) >> 8
        samples = signed.astype(np.float32) / 8388608
    else:
        raise ValueError("仅支持 8、16、24 或 32 位 PCM WAV 文件")
    if samples.size == 0:
        raise ValueError("WAV 文件没有音频数据")
    mono = samples.reshape(-1, channels).mean(axis=1, dtype=np.float32)
    return soxr.resample(mono, rate, SAMPLE_RATE) if rate != SAMPLE_RATE else mono


def write_wav(path: Path, samples: np.ndarray) -> None:
    temp = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".wav", delete=False) as target:
            temp = Path(target.name)
            with wave.open(target, "wb") as destination:
                destination.setnchannels(1)
                destination.setsampwidth(2)
                destination.setframerate(SAMPLE_RATE)
                pcm = np.rint(samples.clip(-1, 1) * 32767).astype("<i2")
                destination.writeframes(pcm.tobytes())
        os.replace(temp, path)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def convert_samples(pipeline: RvcPipeline, samples: np.ndarray, params: dict) -> np.ndarray:
    block_size = pipeline.block_size
    delay = round(pipeline.delay_ms * SAMPLE_RATE / 1000)
    padded_size = ((len(samples) + delay + block_size - 1) // block_size) * block_size
    block = np.zeros(block_size, dtype=np.float32)
    output = np.empty(padded_size, dtype=np.float32)
    for offset in range(0, padded_size, block_size):
        new_audio = samples[offset : offset + block_size]
        block[: len(new_audio)] = new_audio
        block[len(new_audio) :] = 0
        output[offset : offset + block_size] = pipeline.process(block, params)
    return output[delay : delay + len(samples)]


def _show(event: str, payload: dict) -> None:
    if event == "warning":
        print(payload["message"], file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description="用 MaskVox 把 PCM WAV 转换为另一种音色")
    parser.add_argument("--voice", required=True, help="音色库 ID")
    parser.add_argument("--in", dest="input", required=True, type=Path, help="输入 PCM WAV")
    parser.add_argument("--out", dest="output", required=True, type=Path, help="输出 48 kHz WAV")
    parser.add_argument("--pitch", type=int, choices=range(-12, 13), default=0, metavar="半音")
    parser.add_argument("--index-rate", type=float, default=0.5, metavar="比率")
    parser.add_argument("--gate-db", type=float, default=-45, metavar="dB")
    parser.add_argument("--latency", choices=LATENCY_FRAMES, default="balanced")
    parser.add_argument("--gpu", type=int, default=0, metavar="显卡编号", help="使用 -1 选择 CPU")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-f0-9]{8}", args.voice):
        parser.error("音色库 ID 应为 8 位十六进制字符")
    if args.input.resolve() == args.output.resolve():
        parser.error("请为输出选择不同的文件名")
    if not 0 <= args.index_rate <= 1 or not -60 <= args.gate_db <= -20 or args.gpu < -1:
        parser.error("请检查检索比例、噪声门或显卡编号的取值范围")
    try:
        setup_logging()
        voice_dir = VOICES_DIR / args.voice
        voice = json.loads((voice_dir / "meta.json").read_text(encoding="utf-8"))
        samples = read_wav(args.input)
        params = {"pitch": args.pitch, "index_rate": args.index_rate, "gate_db": args.gate_db}
        gpu = args.gpu if DML in available_providers() else -1
        try:
            pipeline = RvcPipeline(ASSETS_DIR, voice_dir, voice, LATENCY_FRAMES[args.latency], gpu, _show)
            pipeline.warmup(Event(), params)
        except Exception:
            if gpu < 0:
                raise
            logger.exception("DirectML offline warm-up failed, retrying on CPU")
            pipeline = RvcPipeline(ASSETS_DIR, voice_dir, voice, LATENCY_FRAMES[args.latency], -1, _show)
            pipeline.warmup(Event(), params)
        if args.gpu >= 0 and pipeline.backend == "CPU":
            _show("warning", {"message": "未启用显卡加速，已使用 CPU 推理"})
        result = convert_samples(pipeline, samples, params)
        write_wav(args.output, result)
        print(f"已保存 {args.output}（{len(result) / SAMPLE_RATE:.2f} 秒，{pipeline.backend}）")
    except (ValueError, OSError, RuntimeError, wave.Error) as exc:
        logger.exception("WAV conversion failed")
        print(f"离线变声失败。{exc}", file=sys.stderr)
        return 1
    except Exception:
        logger.exception("WAV conversion failed")
        print("离线变声失败，请查看 MaskVox 日志。", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
