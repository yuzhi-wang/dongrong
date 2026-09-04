"""DashScope Responses API 的最小 HTTP 客户端。"""

from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def call_responses(
    *,
    api_base_url: str,
    api_key: str,
    model: str,
    instructions: str,
    input_text: str,
    timeout_seconds: int = 180,
) -> dict:
    """发起一次非流式、无工具且不持久化的模型调用。"""
    url = f"{api_base_url.rstrip('/')}/compatible-mode/v1/responses"
    payload = {
        "model": model,
        "instructions": instructions,
        "input": input_text,
        "stream": False,
        "store": False,
        "tool_choice": "none",
        "reasoning": {"effort": "high"},
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
        with urlopen(request, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")[:2000]
        raise RuntimeError(f"DashScope 接口返回 HTTP {exc.code}：{error_body}") from exc
    except TimeoutError as exc:
        raise RuntimeError(
            f"等待 DashScope 响应超过 {timeout_seconds} 秒"
        ) from exc
    except URLError as exc:
        raise RuntimeError(f"无法连接 DashScope 接口：{exc.reason}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("DashScope 接口返回的内容不是有效 JSON") from exc


def extract_output_text(response: dict) -> str:
    """从非流式 Responses API 响应中提取最终文本。"""
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
