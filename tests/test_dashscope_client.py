from __future__ import annotations

import io
import json
import unittest
from http.client import RemoteDisconnected
from urllib.error import HTTPError
from unittest.mock import patch

from dashscope_client import RequestPacer, call_recommendation, extract_output_text


def _stream(chunks: list[dict], *, done: bool = True) -> io.BytesIO:
    text = ': keepalive\n\n' + ''.join(
        'data: ' + json.dumps(chunk, ensure_ascii=False) + '\n\n' for chunk in chunks
    )
    return io.BytesIO((text + ('data: [DONE]\n\n' if done else '')).encode('utf-8'))


class DashScopeClientTest(unittest.TestCase):
    def test_429_retry_after_same_model_and_shared_pacing(self):
        error = HTTPError("https://example.com", 429, "limited", {"Retry-After": "7"}, io.BytesIO(b"limited"))
        chunks = [{"choices": [{"delta": {"content": "{}"}, "finish_reason": "stop"}]}]
        with patch("dashscope_client.urlopen", side_effect=[error, _stream(chunks)]) as http, \
             patch("dashscope_client.time.sleep") as sleep, patch("dashscope_client.RequestPacer") as pacer:
            call_recommendation(api_base_url="https://example.com", api_key="test-key",
                                model="qwen3.8-max", instructions="JSON", input_text="customer", before_request=pacer)
        self.assertEqual(http.call_count, 2)
        self.assertIs(http.call_args_list[0].args[0], http.call_args_list[1].args[0])
        self.assertEqual(pacer.call_count, 2)
        sleep.assert_called_once_with(7.0)

    def test_retry_limit_and_no_retry_for_auth_failure(self):
        for code, attempts in ((429, 3), (401, 1)):
            errors = [HTTPError("https://example.com", code, "error", {}, io.BytesIO(b"error")) for _ in range(attempts)]
            with self.subTest(code=code), patch("dashscope_client.urlopen", side_effect=errors) as http, \
                 patch("dashscope_client.time.sleep") as sleep:
                with self.assertRaisesRegex(RuntimeError, f"HTTP {code}"):
                    call_recommendation(api_base_url="https://example.com", api_key="test-key",
                                        model="qwen3.8-max", instructions="JSON", input_text="customer")
                self.assertEqual(http.call_count, attempts)
                self.assertEqual(sleep.call_count, attempts-1)

    def test_pacer_spaces_request_starts(self):
        now = [10.0]
        def sleep(delay):
            now[0] += delay
        with patch("dashscope_client.time.monotonic", side_effect=lambda: now[0]), \
             patch("dashscope_client.time.sleep", side_effect=sleep):
            pacer = RequestPacer(0.25)
            starts = []
            for _ in range(4):
                pacer()
                starts.append(now[0])
        self.assertEqual(starts, [10, 10.25, 10.5, 10.75])

    @patch("dashscope_client.urlopen", side_effect=TimeoutError("timed out"))
    def test_timeout_is_reported_as_runtime_error(self, _mock_urlopen) -> None:
        with self.assertRaisesRegex(RuntimeError, "超过 3 秒"):
            call_recommendation(
                api_base_url="https://example.com",
                api_key="test-key",
                model="test-model",
                instructions="system",
                input_text="user",
                timeout_seconds=3,
            )

    def test_json_request_and_normalized_response(self) -> None:
        body = json.dumps({"reasons": ['包含"引号"']}, ensure_ascii=False)
        chunks = [
            {"id": "test-response", "model": "qwen3.8-max", "choices": [{"delta": {"reasoning_content": "不应进入最终输出"}}]},
            {"choices": [{"delta": {"content": body[:15]}}]},
            {"choices": [{"delta": {"content": body[15:]}, "finish_reason": "stop"}]},
            {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}},
        ]
        with patch("dashscope_client.urlopen", return_value=_stream(chunks)) as http:
            response = call_recommendation(
                api_base_url="https://example.com/", api_key="test-key", model="qwen3.8-max",
                instructions="返回 JSON", input_text="客户信息", timeout_seconds=300,
            )
        request = http.call_args.args[0]
        self.assertEqual(request.full_url, "https://example.com/compatible-mode/v1/chat/completions")
        payload = json.loads(request.data)
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertEqual(payload["thinking_budget"], 4096)
        self.assertTrue(payload["enable_thinking"])
        self.assertTrue(payload["stream"])
        self.assertEqual(payload["stream_options"], {"include_usage": True})
        self.assertEqual([m["role"] for m in payload["messages"]], ["system", "user"])
        self.assertEqual(response["usage"]["input_tokens"], 10)
        self.assertEqual(response["usage"]["output_tokens"], 20)
        self.assertEqual(json.loads(extract_output_text(response))["reasons"], ['包含"引号"'])

    def test_rejects_truncated_or_empty_completion(self) -> None:
        for choices in (
            [],
            [{"finish_reason": "length", "delta": {"content": '{"partial":'}}],
            [{"finish_reason": "stop", "delta": {"content": ""}}],
        ):
            with self.subTest(choices=choices), patch("dashscope_client.urlopen", return_value=_stream([{"choices": choices}])):
                with self.assertRaises(RuntimeError):
                    call_recommendation(
                        api_base_url="https://example.com", api_key="test-key", model="qwen3.8-max",
                        instructions="返回 JSON", input_text="客户信息",
                    )

    def test_rejects_stream_without_done_marker(self) -> None:
        chunks = [{"choices": [{"finish_reason": "stop", "delta": {"content": "{}"}}]}]
        with patch("dashscope_client.urlopen", return_value=_stream(chunks, done=False)):
            with self.assertRaisesRegex(RuntimeError, "DONE"):
                call_recommendation(
                    api_base_url="https://example.com", api_key="test-key", model="qwen3.8-max",
                    instructions="返回 JSON", input_text="客户信息",
                )

    def test_remote_disconnect_is_reported_as_runtime_error(self) -> None:
        with patch("dashscope_client.urlopen", side_effect=RemoteDisconnected()):
            with self.assertRaisesRegex(RuntimeError, "连接中断"):
                call_recommendation(
                    api_base_url="https://example.com", api_key="test-key", model="qwen3.8-max",
                    instructions="返回 JSON", input_text="客户信息",
                )


if __name__ == "__main__":
    unittest.main()
