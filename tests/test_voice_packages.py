"""Archive and library regressions for local and shared voice packages."""

import json
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

import onnx
from onnx import TensorProto, helper
from onnx.external_data_helper import set_external_data

from app.voices.library import VoiceLibrary
from app.voices.package import VoiceError, match_index, unpack_package, unpack_rvc, verify_model


def write_model(path: Path, external: bool = False) -> None:
    inputs = [
        helper.make_tensor_value_info("phone", TensorProto.FLOAT, [1, "frames", 256]),
        helper.make_tensor_value_info("phone_lengths", TensorProto.INT64, [1]),
        helper.make_tensor_value_info("pitch", TensorProto.INT64, [1, "frames"]),
        helper.make_tensor_value_info("pitchf", TensorProto.FLOAT, [1, "frames"]),
        helper.make_tensor_value_info("ds", TensorProto.INT64, [1]),
        helper.make_tensor_value_info("rnd", TensorProto.FLOAT, [1, 192, "frames"]),
        helper.make_tensor_value_info("source_noise", TensorProto.FLOAT, [1, "samples", 1]),
    ]
    value = helper.make_tensor("wave", TensorProto.FLOAT, [1, 1, 4], b"\0" * 16, raw=True)
    if external:
        set_external_data(value, "../secrets.dat")
    node = helper.make_node("Constant", [], ["audio"], value=value)
    output = helper.make_tensor_value_info("audio", TensorProto.FLOAT, [1, 1, 4])
    model = helper.make_model(helper.make_graph([node], "package_test", inputs, [output]))
    helper.set_model_props(model, {"mv_format": "1", "version": "v1", "sr": "32000", "vec_dim": "256", "f0": "1"})
    path.write_bytes(model.SerializeToString())


class PackageTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def test_catalog_round_trip(self):
        library = VoiceLibrary(self.root / "音色库")
        with library.stage() as staging:
            folder = Path(staging)
            write_model(folder / "model.onnx")
            voice = library.add(folder, "测试音色", "test.pth")
        voice = library.rename(voice["id"], "更新后的音色")
        path = self.root / "分享.maskvox"
        library.export(voice["id"], path)
        with library.stage() as staging:
            folder = Path(staging)
            meta = unpack_package(path, folder)
            copy = library.add(folder, meta["name"], path.name, verified=meta)
        self.assertEqual(voice["name"], "更新后的音色")
        self.assertEqual(copy["name"], "更新后的音色 (2)")
        library.delete(voice["id"])
        self.assertEqual(library.list(), [copy])

    def test_traversal_rejected_for_both_archives(self):
        for name in ("../out.pth", "safe/../../out.pth", "C:/out.pth", "safe/file.pth"):
            with self.subTest(name=name):
                path = self.root / "bad.zip"
                with ZipFile(path, "w") as archive:
                    archive.writestr(name, b"model")
                if name == "safe/file.pth":
                    path.write_bytes(path.read_bytes().replace(b"safe/file.pth", b"safe\\file.pth"))
                with self.assertRaisesRegex(VoiceError, "不安全"):
                    unpack_rvc(path, self.root)
                with self.assertRaisesRegex(VoiceError, "不安全"):
                    unpack_package(path, self.root)

    def test_rejects_external_data_in_node_attribute(self):
        path = self.root / "model.onnx"
        write_model(path, external=True)
        with self.assertRaisesRegex(VoiceError, "外部数据"):
            verify_model(path)

    def test_package_metadata_comes_from_graph(self):
        model = self.root / "source.onnx"
        write_model(model)
        path = self.root / "source.maskvox"
        with ZipFile(path, "w") as archive:
            archive.write(model, "model.onnx")
            archive.writestr("meta.json", json.dumps({"name": "Test", "sr": 999, "f0": False}))
        meta = unpack_package(path, self.root)
        self.assertEqual(meta["sr"], 32000)
        self.assertEqual(meta["f0"], 1)

    def test_rvc_index_pairing(self):
        added = Path("added")
        trained = Path("trained")
        options = [
            ("trained_IVF128_Flat_nprobe_1_azure_v2.index", trained),
            ("added_IVF128_Flat_nprobe_1_azure_v2.index", added),
            ("added_IVF128_Flat_nprobe_1_unrelated_v2.index", Path("unrelated")),
        ]
        self.assertEqual(match_index("azure.pth", options), added)
        self.assertIsNone(match_index("vermilion.pth", options))
