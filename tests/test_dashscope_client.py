from __future__ import annotations

import unittest
from unittest.mock import patch

from dashscope_client import call_responses


class DashScopeClientTest(unittest.TestCase):
    @patch("dashscope_client.urlopen", side_effect=TimeoutError("timed out"))
    def test_timeout_is_reported_as_runtime_error(self, _mock_urlopen) -> None:
        with self.assertRaisesRegex(RuntimeError, "超过 3 秒"):
            call_responses(
                api_base_url="https://example.com",
                api_key="test-key",
                model="test-model",
                instructions="system",
                input_text="user",
                timeout_seconds=3,
            )


if __name__ == "__main__":
    unittest.main()
