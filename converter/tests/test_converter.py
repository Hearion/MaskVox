import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import onnx
import torch

from app.engine.f0_rmvpe import MelSpectrogram as NumpyMel
from converter.convert_voice import ConversionError, convert, load_voice
from converter.vendor.infer_pack.models import SynthesizerTrnMs256NSFsid, SynthesizerTrnMs768NSFsid
from converter.vendor.rmvpe import MelSpectrogram as TorchMel


def synthetic_checkpoint(version: str, sr: int) -> dict:
    upsample = {
        32000: ([10, 4, 2, 2, 2], [20, 8, 4, 4, 4]),
        40000: ([10, 10, 2, 2], [16, 16, 4, 4]),
        48000: ([12, 5, 2, 2, 2], [24, 9, 4, 4, 4]),
    }
    strides, kernels = upsample[sr]
    config = [33, 32, 16, 16, 32, 2, 2, 3, 0.0, "1", [3], [[1, 3, 5]],
              strides, 64, kernels, 1, 8, sr]
    definition = SynthesizerTrnMs256NSFsid if version == "v1" else SynthesizerTrnMs768NSFsid
    voice = definition(*config, is_half=False)
    weights = {key: value.half() for key, value in voice.state_dict().items()
               if not key.startswith("enc_q.")}
    checkpoint = {"f0": 1, "config": config, "weight": weights}
    if version == "v2":
        checkpoint["version"] = version
    return checkpoint


class ConverterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(4)
        torch.manual_seed(251)

    def test_dynamic_conversion_of_synthetic_v1_and_v2_at_all_rates(self) -> None:
        with tempfile.TemporaryDirectory(prefix="MaskVox 测试 ") as temporary:
            workspace = Path(temporary)
            for version, sr in (("v1", 32000), ("v2", 40000), ("v2", 48000)):
                with self.subTest(version=version, sample_rate=sr):
                    checkpoint = workspace / "合成测试.pth"
                    out = workspace / f"{version}-{sr}"
                    torch.save(synthetic_checkpoint(version, sr), checkpoint)
                    stream = io.StringIO()
                    with redirect_stdout(stream):
                        convert(checkpoint, out)
                    stages = [json.loads(line)["stage"] for line in stream.getvalue().splitlines()]
                    self.assertEqual(stages, ["loading", "exporting", "verifying", "done"])
                    graph = onnx.load(out / "model.onnx", load_external_data=False)
                    properties = {item.key: item.value for item in graph.metadata_props}
                    self.assertEqual(properties["mv_format"], "1")
                    self.assertEqual(properties["version"], version)
                    self.assertEqual(int(properties["sr"]), sr)
                    self.assertEqual(json.loads((out / "meta.json").read_text(encoding="utf-8"))["sr"], sr)

    def test_rejected_checkpoints_report_distinct_reasons(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "test.pth"
            path.write_bytes(b"invalid checkpoint")
            with self.assertRaisesRegex(ConversionError, "不安全或已损坏"):
                load_voice(path)
            torch.save({"iteration": 10, "model": {}}, path)
            with self.assertRaisesRegex(ConversionError, "训练检查点"):
                load_voice(path)
            torch.save({"f0": 0, "weight": {}}, path)
            with self.assertRaisesRegex(ConversionError, "暂不支持不带音高"):
                load_voice(path)

    def test_numpy_mel_matches_vendor_reference(self) -> None:
        audio = np.random.default_rng(26).normal(0, 0.08, 16320).astype(np.float32)
        reference = TorchMel(False, 128, 16000, 1024, 160, 1024, 30, 8000).eval()
        with torch.inference_mode():
            expected = reference(torch.from_numpy(audio[None, :])).numpy()[0]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "rmvpe_mel.npz"
            np.savez(path, mel_basis=reference.mel_basis.numpy(), window=reference.hann_window["0_cpu"].numpy(),
                     sr=16000, hop=160, n_fft=1024, n_mels=128, fmin=30, fmax=8000)
            actual = NumpyMel(path)(audio)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=3e-5)
