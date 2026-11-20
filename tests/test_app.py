"""The desktop entry point must remain independent of the offline converter."""

import subprocess
import sys
import unittest


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
