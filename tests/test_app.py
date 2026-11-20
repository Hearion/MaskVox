"""The interface API: settings, effect selection and engine restarts."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from threading import RLock
from unittest.mock import Mock, patch

from app.api import Api
from app.config import Settings


def _api(root: str, engine_state: str = "idle") -> Api:
    api = Api.__new__(Api)
    api._lock = RLock()
    api._closing = False
    api._settings = Settings(Path(root) / "settings.json")
    api._engine = Mock(state=engine_state)
    api._devices = {"has_vb_cable": False}
    return api


class AppTests(unittest.TestCase):
    def test_main_needs_no_machine_learning_packages(self):
        result = subprocess.run(
            [
                sys.executable, "-c",
                "import sys, app.main, app.api; "
                "assert not set(('torch', 'onnxruntime', 'faiss', 'scipy', 'soxr')) & sys.modules.keys()",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_choosing_an_effect_loads_its_sliders_and_updates_a_running_engine(self):
        with tempfile.TemporaryDirectory() as root:
            api = _api(root, "running")
            settings = api.select_preset("giant")["data"]
            self.assertEqual((settings["preset"], settings["pitch"], settings["formant"], settings["reverb"]),
                             ("giant", -9, -5, 0.15))
            params = api._engine.update_params.call_args.args[0]
            self.assertEqual((params["pitch"], params["mode"], params["monitor"]), (-9, "normal", False))
            api._engine.stop.assert_not_called()
            api._settings.update(input_id=1, output_id=2)

            # A slider keeps the effect and changes one value.
            settings = api.set_param("pitch", 3.2)["data"]
            self.assertEqual((settings["preset"], settings["pitch"]), ("giant", 3.0))
            self.assertFalse(api.select_preset("nope")["ok"])

            # The delay reopens the audio streams with the new amount queued.
            self.assertEqual(api.set_param("delay_ms", 204)["data"]["delay_ms"], 200)
            api._engine.stop.assert_called_once_with()
            self.assertEqual(api._engine.start.call_args.kwargs["delay_ms"], 200)
            self.assertFalse(api.set_param("delay_ms", 5)["ok"])
            self.assertFalse(api.set_param("pitch", 40)["ok"])
            self.assertFalse(api.set_param("mode", "robot")["ok"])

    def test_settings_survive_a_restart_and_ignore_old_keys(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "settings.json"
            path.write_text('{"preset": "robot", "pitch": 2, "selected_voice_id": "abcd1234", "echo": 9}', encoding="utf-8")
            settings = Settings(path).snapshot()
            self.assertEqual((settings["preset"], settings["pitch"], settings["echo"]), ("robot", 2, 0))
            self.assertNotIn("selected_voice_id", settings)

    def test_rejected_restart_restores_previous_settings(self):
        with tempfile.TemporaryDirectory() as root:
            api = _api(root, "running")
            api._start_locked = Mock(side_effect=[ValueError("设备不可用"), None])
            with self.assertRaisesRegex(ValueError, "设备不可用"):
                api._replace_settings(restart=True, input_id=3)
            self.assertIsNone(api._settings.snapshot()["input_id"])
            self.assertIsNone(api._start_locked.call_args.args[0]["input_id"])

    def test_driver_launch_requires_an_idle_engine(self):
        with tempfile.TemporaryDirectory() as root:
            api = _api(root, "running")
            with patch("app.api.cable_setup.launch_installer") as launch:
                self.assertFalse(api.install_vb_cable()["ok"])
                launch.assert_not_called()
                api._engine.state = "idle"
                self.assertTrue(api.install_vb_cable()["ok"])
                launch.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
