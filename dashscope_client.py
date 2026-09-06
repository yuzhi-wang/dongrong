"""DashScope Chat Completions JSON 模式客户端，统一返回内部响应结构。"""

from __future__ import annotations

import json
import math
import random
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from threading import Lock
from typing import Callable
from http.client import HTTPException
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

MAX_RATE_LIMIT_RETRIES = 2


class RequestPacer:
    """在工作线程之间平滑分配请求起始时间，重试也走同一入口。"""

    def __init__(self, interval: float):
        self.interval = interval
        self._lock = Lock()
        self._next_start = 0.0

    def __call__(self):
        with self._lock:
            time.sleep(max(0.0, self._next_start - time.monotonic()))
            self._next_start = time.monotonic() + self.interval


def _retry_delay(header: str | None, attempt: int) -> float:
    delay = 2 ** (attempt + 1) + random.uniform(0, 1)
    if header:
        try:
            retry_after = float(header)
        except ValueError:
            try:
                retry_after = (parsedate_to_datetime(header) - datetime.now(timezone.utc)).total_seconds()
            except (ValueError, TypeError, OverflowError):
                retry_after = 0
        if math.isfinite(retry_after):
            delay = max(delay, retry_after)
    return delay


def _read_chat_stream(response) -> dict:
    """收集 SSE 正文与用量；忽略思考内容，拒绝不完整的数据流。"""
    content: list[str] = []
    completion: dict = {"choices": []}
    finish_reason = None
    completed = False
    for line in response:
        line = line.decode("utf-8").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            completed = True
            break
        chunk = json.loads(data)
        if chunk.get("error"):
            raise RuntimeError("DashScope 流式接口返回错误，未获得完整推荐")
        for field in ("id", "model", "usage"):
            if chunk.get(field):
                completion[field] = chunk[field]
        for choice in chunk.get("choices") or []:
            if choice.get("index", 0) != 0:
                continue
            text = (choice.get("delta") or {}).get("content")
            if text:
                content.append(text)
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]
    if not completed:
        raise RuntimeError("DashScope 流式响应中断，未收到 [DONE]")
    completion["choices"] = [{
        "finish_reason": finish_reason,
        "message": {"content": "".join(content)},
    }]
    return completion


def call_recommendation(
    *,
    api_base_url: str,
    api_key: str,
    model: str,
    instructions: str,
    input_text: str,
    timeout_seconds: int = 300,
    before_request: Callable[[], None] | None = None,
) -> dict:
    """流式 JSON 调用；仅 HTTP 429 最多重试两次，始终使用同一模型。"""
    url = f"{api_base_url.rstrip('/')}/compatible-mode/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": instructions},
            {"role": "user", "content": input_text},
        ],
        "stream": True,
        "stream_options": {"include_usage": True},
        "store": False,
        "response_format": {"type": "json_object"},
        "enable_thinking": True,
        "thinking_budget": 4096,
        "temperature": 0.1,
    }
    request = Request(
        url=url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
            if before_request is not None:
                before_request()
            try:
                with urlopen(request, timeout=timeout_seconds) as response:
                    completion = _read_chat_stream(response)
                break
            except HTTPError as exc:
                if exc.code != 429 or attempt == MAX_RATE_LIMIT_RETRIES:
                    raise
                delay = _retry_delay((exc.headers or {}).get("Retry-After"), attempt)
                exc.close()
                time.sleep(delay)
    except HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")[:2000]
        raise RuntimeError(f"DashScope 接口返回 HTTP {exc.code}：{error_body}") from exc
    except TimeoutError as exc:
        raise RuntimeError(
            f"等待 DashScope 响应超过 {timeout_seconds} 秒"
        ) from exc
    except URLError as exc:
        raise RuntimeError(f"无法连接 DashScope 接口：{exc.reason}") from exc
    except (HTTPException, ConnectionError) as exc:
        raise RuntimeError("DashScope 连接中断，未获得完整响应，请稍后重试") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("DashScope 接口返回的内容不是有效 JSON") from exc

    choices = completion.get("choices") or []
    if not choices:
        raise RuntimeError("模型响应中没有 choices")
    choice = choices[0]
    if choice.get("finish_reason") != "stop":
        raise RuntimeError(f"模型输出未正常完成：finish_reason={choice.get('finish_reason')}")
    output_text = (choice.get("message") or {}).get("content")
    if not isinstance(output_text, str) or not output_text.strip():
        raise RuntimeError("模型响应中没有有效 content")
    usage = completion.get("usage") or {}
    # 保持主入口、回测报告及现有解析器使用的内部结构一致。
    return {
        "id": completion.get("id"),
        "model": completion.get("model"),
        "status": "completed",
        "api_format": "chat_completions_json",
        "output": [{
            "type": "message",
            "content": [{"type": "output_text", "text": output_text}],
        }],
        "usage": {
            "input_tokens": usage.get("prompt_tokens"),
            "output_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "input_tokens_details": usage.get("prompt_tokens_details") or {},
            "output_tokens_details": usage.get("completion_tokens_details") or {},
        },
    }


def extract_output_text(response: dict) -> str:
    """从内部统一响应结构提取最终文本。"""
    texts: list[str] = []
    for item in response.get("output") or []:
        if item.get("type") != "message":
            continue
        for content in item.get("content") or []:
            if content.get("type") == "output_text" and content.get("text"):
                texts.append(content["text"])
    if not texts:
        raise RuntimeError("模型响应中没有 output_text")
    return "\n".join(texts).strip()
