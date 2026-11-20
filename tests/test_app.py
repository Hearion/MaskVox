"""The desktop entry point must remain independent of the offline converter."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from threading import RLock
from unittest.mock import Mock, patch

from app.api import Api
from app.config import Settings


class AppTests(unittest.TestCase):
    def test_main_does_not_load_torch(self):
        result = subprocess.run(
            [
                sys.executable, "-c",
                "import sys, app.main, app.api; "
                "assert not set(('torch', 'torchaudio', 'transformers', 'librosa', 'scipy')) & sys.modules.keys()",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_rejected_restart_restores_previous_settings(self):
        with tempfile.TemporaryDirectory() as root:
            api = Api.__new__(Api)
            api._settings = Settings(Path(root) / "settings.json")
            api._engine = Mock(state="running")
            api._start_locked = Mock(side_effect=[ValueError("设备不可用"), None])
            with self.assertRaisesRegex(ValueError, "设备不可用"):
                api._replace_settings(restart=True, latency_mode="low")
            self.assertEqual(api._settings.snapshot()["latency_mode"], "balanced")
            self.assertEqual(api._start_locked.call_args.args[0]["latency_mode"], "balanced")

    def test_voice_reconciliation_preserves_existing_choice(self):
        with tempfile.TemporaryDirectory() as root:
            api = Api.__new__(Api)
            api._settings = Settings(Path(root) / "settings.json")
            api._library = Mock()
            api._library.list.return_value = [{"id": "b0010001"}]
            api._reconcile_voice()
            self.assertEqual(api._settings.snapshot()["selected_voice_id"], "b0010001")
            api._settings.update(selected_voice_id="abcdef12")
            (Path(root) / "model.onnx").touch()
            api._library.directory.return_value = Path(root)
            api._reconcile_voice()
            self.assertEqual(api._settings.snapshot()["selected_voice_id"], "abcdef12")

    def test_driver_launch_requires_an_idle_engine_and_stays_in_app(self):
        api = Api.__new__(Api)
        api._lock = RLock()
        api._closing = False
        api._engine = Mock(state="running")
        api._devices = {"has_vb_cable": False}
        with patch("app.api.cable_setup.launch_installer") as launch:
            self.assertFalse(api.install_vb_cable()["ok"])
            launch.assert_not_called()
            api._engine.state = "idle"
            self.assertTrue(api.install_vb_cable()["ok"])
            launch.assert_called_once_with()
