from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from api_key import load_api_key


class ApiKeyTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "key.txt"
        environment = patch.dict("os.environ", {"DASHSCOPE_API_KEY": ""})
        environment.start()
        self.addCleanup(environment.stop)

    def test_reads_windows_bom_and_trims_whitespace(self) -> None:
        self.path.write_text("  test-file-key\n", encoding="utf-8-sig")
        self.assertEqual(load_api_key(self.path), "test-file-key")

    def test_environment_overrides_unreadable_file(self) -> None:
        self.path.write_bytes(b"\xff")
        with patch.dict("os.environ", {"DASHSCOPE_API_KEY": " test-env-key "}):
            self.assertEqual(load_api_key(self.path), "test-env-key")

    def test_missing_or_empty_file_explains_configuration(self) -> None:
        with self.assertRaisesRegex(ValueError, "dashscope_api_key.txt"):
            load_api_key(self.path)
        self.path.write_text("\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "DASHSCOPE_API_KEY"):
            load_api_key(self.path)

    def test_invalid_encoding_reports_actionable_error(self) -> None:
        self.path.write_bytes(b"\xff")
        with self.assertRaisesRegex(ValueError, "UTF-8"):
            load_api_key(self.path)

    def test_multiline_error_does_not_echo_key(self) -> None:
        self.path.write_text("test-secret-key\nextra-line", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "一行") as raised:
            load_api_key(self.path)
        self.assertNotIn("test-secret-key", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
