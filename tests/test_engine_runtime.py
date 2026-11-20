from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from threading import Event

import numpy as np
import onnx
from onnx import TensorProto as P
from onnx import helper as h
from onnx.numpy_helper import from_array

from app.engine.f0_rmvpe import decode, f0_to_coarse
from app.engine.index import VoiceIndex
from app.engine.realtime import LATENCY_FRAMES, RvcPipeline, apply_gate
from scripts.offline_convert import convert_samples, read_wav, write_wav


def _save(path, nodes, inputs, outputs, initializers, metadata=None) -> None:
    graph = h.make_graph(nodes, path.stem, inputs, outputs, initializers)
    model = h.make_model(graph, ir_version=9, opset_imports=[h.make_opsetid("", 17)])
    if metadata:
        h.set_model_props(model, metadata)
    onnx.checker.check_model(model)
    onnx.save(model, path)


def _int_vector(name, values):
    return from_array(np.array(values, dtype=np.int64), name)


def _content(path) -> None:
    _save(
        path,
        [
            h.make_node("Slice", ["audio", "start", "end", "dim1", "hop"], ["downsampled"]),
            h.make_node("Unsqueeze", ["downsampled", "dim2"], ["channel"]),
            h.make_node("Tile", ["channel", "repeat256"], ["feats256"]),
            h.make_node("Tile", ["channel", "repeat768"], ["feats768"]),
        ],
        [h.make_tensor_value_info("audio", P.FLOAT, [1, "samples"])],
        [h.make_tensor_value_info(f"feats{d}", P.FLOAT, [1, "feat_frames", d]) for d in (256, 768)],
        [_int_vector(key, value) for key, value in {
            "start": [0], "end": [-399], "dim1": [1], "dim2": [2], "hop": [320],
            "repeat256": [1, 1, 256], "repeat768": [1, 1, 768],
        }.items()],
    )


def _rmvpe(assets: Path) -> None:
    salience = np.zeros((1, 1, 360), dtype=np.float32)
    salience[0, 0, 104] = 0.7
    _save(
        assets / "rmvpe.onnx",
        [
            h.make_node("Shape", ["mel"], ["mel_shape"]),
            h.make_node("Gather", ["mel_shape", "time_axis"], ["frame_count"]),
            h.make_node("Concat", ["batch", "frame_count", "bins"], ["out_shape"], axis=0),
            h.make_node("Expand", ["template", "out_shape"], ["salience"]),
        ],
        [h.make_tensor_value_info("mel", P.FLOAT, [1, 128, "frames"])],
        [h.make_tensor_value_info("salience", P.FLOAT, [1, "frames", 360])],
        [from_array(salience, "template"), _int_vector("time_axis", [2]),
         _int_vector("batch", [1]), _int_vector("bins", [360])],
    )
    np.savez(
        assets / "rmvpe_mel.npz", mel_basis=np.full((128, 513), 0.008, dtype=np.float32),
        window=np.hanning(1025)[:-1].astype(np.float32), sr=16000, n_fft=1024,
        hop=160, n_mels=128, fmin=30, fmax=8000,
    )


def _voice(path: Path, rate: int, dimension: int, has_source_noise: bool) -> dict:
    version = "v1" if dimension == 256 else "v2"
    inputs = [
        h.make_tensor_value_info("phone", P.FLOAT, [1, "frames", dimension]),
        h.make_tensor_value_info("phone_lengths", P.INT64, [1]),
        h.make_tensor_value_info("pitch", P.INT64, [1, "frames"]),
        h.make_tensor_value_info("pitchf", P.FLOAT, [1, "frames"]),
        h.make_tensor_value_info("ds", P.INT64, [1]),
        h.make_tensor_value_info("rnd", P.FLOAT, [1, 192, "frames"]),
    ]
    if has_source_noise:
        inputs.append(h.make_tensor_value_info("source_noise", P.FLOAT, [1, "samples", 1]))
    stride = rate // 100
    params = {"version": version, "sr": rate, "f0": 1, "vec_dim": dimension, "mv_format": 1}
    _save(
        path,
        [
            h.make_node("Gather", ["phone", "channel"], ["values"], axis=2),
            h.make_node("Tile", ["values", "repeats"], ["samples_per_frame"]),
            h.make_node("Reshape", ["samples_per_frame", "audio_shape"], ["audio"]),
        ],
        inputs, [h.make_tensor_value_info("audio", P.FLOAT, [1, 1, "samples"])],
        [_int_vector("channel", [0]), _int_vector("repeats", [1, 1, stride]),
         _int_vector("audio_shape", [1, 1, -1])],
        {key: str(value) for key, value in params.items()},
    )
    return params


class EngineRuntimeTests(unittest.TestCase):
    def test_coarse_pitch_is_quantized_and_clipped(self) -> None:
        values = f0_to_coarse(np.array([-1, 0, 50, 440, 1100, 9000], dtype=np.float32))
        np.testing.assert_array_equal(values, [1, 1, 1, 122, 255, 255])

    def test_rmvpe_centroid_handles_edges_and_quiet(self) -> None:
        salience = np.zeros((4, 360), dtype=np.float32)
        salience[0, :2] = [0.8, 0.2]
        salience[1, -1] = 1
        salience[2, 48] = 0.02
        actual = decode(salience)
        expected = 10 * np.exp2((np.array([0.2, 359]) * 20 + 1997.3794084376191) / 1200)
        np.testing.assert_allclose(actual[:2], expected, rtol=1e-6)
        np.testing.assert_array_equal(actual[2:], [0, 0])

    def test_noise_gate_works_on_single_ten_ms_frame(self) -> None:
        samples = np.repeat(np.array([0.001, 0.2], dtype=np.float32), 480)
        np.testing.assert_array_equal(apply_gate(samples, -45), np.repeat(np.array([0, 0.2], dtype=np.float32), 480))
        np.testing.assert_array_equal(apply_gate(samples, -60), samples)

    def test_faiss_top_matches_and_ignores_dimension_mismatch(self) -> None:
        import faiss

        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / "中文索引.index"
            features = np.zeros((2, 256), dtype=np.float32)
            features[1].fill(0.25)
            database = faiss.IndexFlatL2(256)
            database.add(features)
            filename.write_bytes(bytes(faiss.serialize_index(database)))
            index = VoiceIndex(filename, 256)
            np.testing.assert_array_equal(index.blend(features, 1.0), features)
            with self.assertRaisesRegex(ValueError, "索引维度"):
                VoiceIndex(filename, 768)

            # RVC training normally writes a self-contained IVF Flat index.
            train = np.random.default_rng(0).standard_normal((128, 256), dtype=np.float32)
            database = faiss.IndexIVFFlat(faiss.IndexFlatL2(256), 256, 2)
            database.train(train)
            database.add(train)
            filename.write_bytes(bytes(faiss.serialize_index(database)))
            index = VoiceIndex(filename, 256)
            np.testing.assert_allclose(index.blend(train[:2], 1.0), train[:2], atol=1e-6)

    def test_full_fake_onnx_pipeline_for_each_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            assets = Path(directory)
            _content(assets / "contentvec.onnx")
            _rmvpe(assets)
            for (mode, frames), rate, dimension in zip(LATENCY_FRAMES.items(), (32000, 40000, 48000), (256, 768, 768)):
                with self.subTest(mode=mode):
                    voice = _voice(assets / "model.onnx", rate, dimension, has_source_noise=rate != 32000)
                    pipeline = RvcPipeline(assets, assets, voice, frames, -1, lambda *_: None)
                    params = {"gate_db": -60, "pitch": 5, "index_rate": 0}
                    pipeline.warmup(Event(), params)
                    positions = np.arange(pipeline.block_size * 5 + 113)
                    source = (0.25 * np.sin(positions * (2 * np.pi * 2 / 48000))).astype(np.float32)
                    source[-1920:] *= np.linspace(1, 0, 1920, dtype=np.float32)
                    output = convert_samples(pipeline, source, params)
                    self.assertEqual(output.shape, source.shape)
                    self.assertTrue(np.isfinite(output).all())
                    self.assertGreater(np.max(np.abs(output)), 0.15)
                    # These dummy graphs hold each feature frame flat. Allow
                    # their normal staircase steps and check the splice itself.
                    self.assertLess(np.max(np.abs(np.diff(output))), 0.12)
                    boundaries = np.arange(1, 6) * pipeline.block_size - round(pipeline.delay_ms * 48)
                    self.assertLess(np.max(np.abs(output[boundaries] - output[boundaries - 1])), 0.065)
                    wave_path = assets / "合成样本.wav"
                    write_wav(wave_path, output)
                    np.testing.assert_allclose(read_wav(wave_path), output, atol=5e-5)


if __name__ == "__main__":
    unittest.main()
