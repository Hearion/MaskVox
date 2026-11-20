"""The desktop entry point must remain independent of the offline converter."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

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
