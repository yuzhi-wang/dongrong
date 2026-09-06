"""读取环境变量或随项目交付的明文 DashScope Key。"""

from __future__ import annotations

import os
from pathlib import Path


KEY_PATH = Path(__file__).resolve().parent / "dashscope_api_key.txt"


def load_api_key(path: Path = KEY_PATH) -> str:
    """环境变量优先；文件中只放 Key，兼容 Windows UTF-8 BOM。"""
    key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if key:
        return key
    try:
        key = path.read_text(encoding="utf-8-sig").strip()
    except FileNotFoundError:
        key = ""
    except (OSError, UnicodeError) as exc:
        raise ValueError("无法读取 dashscope_api_key.txt，请检查权限及 UTF-8 编码") from exc
    if not key:
        raise ValueError(
            "请在项目根目录的 dashscope_api_key.txt 中填入 API Key，"
            "或设置环境变量 DASHSCOPE_API_KEY"
        )
    if len(key.splitlines()) != 1:
        raise ValueError("dashscope_api_key.txt 中只能填写一行 API Key，不要添加说明文字")
    return key
