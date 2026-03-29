from __future__ import annotations

import importlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import screenlot.runtime as runtime


class RuntimePathTests(unittest.TestCase):
    def test_invalid_data_env_falls_back_to_packaged_demo(self) -> None:
        with mock.patch.dict(os.environ, {"SCREENLOT_DATA_DIR": "/tmp/does-not-exist-screenlot"}, clear=False):
            reloaded = importlib.reload(runtime)
            self.assertEqual(reloaded.resolve_default_data_dir(), reloaded.PACKAGED_DATA_DIR)

        importlib.reload(runtime)

    def test_existing_data_env_is_respected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            with mock.patch.dict(os.environ, {"SCREENLOT_DATA_DIR": str(temp_path)}, clear=False):
                reloaded = importlib.reload(runtime)
                self.assertEqual(reloaded.resolve_default_data_dir(), temp_path)

        importlib.reload(runtime)


if __name__ == "__main__":
    unittest.main()
