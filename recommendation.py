"""推荐流程共用的配置、数据读取、提示词与结果校验。"""

from __future__ import annotations

import json
import copy
import os
from pathlib import Path



ROOT_DIR = Path(__file__).resolve().parent
PRODUCTS_PATH = ROOT_DIR / "data" / "loan_products.json"
CUSTOMER_PATH = ROOT_DIR / "data" / "test_customer.json"
SYSTEM_PROMPT_PATH = ROOT_DIR / "prompts" / "loan_recommendation_system.md"
USER_PROMPT_PATH = ROOT_DIR / "prompts" / "loan_recommendation_user.md"

API_BASE_URL = os.getenv("DASHSCOPE_API_BASE_URL", "https://dashscope.aliyuncs.com")
MODEL = os.getenv("DASHSCOPE_MODEL", "qwen3.8-max")
TIMEOUT_SECONDS = 300

RANKING_PRODUCT_FIELDS = (
    "产品唯一ID",
    "银行名称",
    "产品名称",
    "产品类型",
    "最高额度（元）",
    "最长期限（月）",
    "支持的还款方式",
    "最低年利率",
    "最高年利率",
    "允许的身份",
    "允许的客群（满足其一）",
    "单位性质要求",
    "现单位在职时长要求",
    "公积金要求",
    "社保要求",
    "打卡工资（或者个税）要求",
    "经营流水要求（企业主身份客群）",
    "资产要求（允许客群）",
    "资金用途限制",
    "备注",
)


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取有效 JSON：{path.name}") from exc


def validate_inputs() -> None:
    if not API_BASE_URL.startswith(("http://", "https://")):
        raise ValueError("DASHSCOPE_API_BASE_URL 必须是 HTTP(S) 地址")

    missing_files = [
        path.relative_to(ROOT_DIR).as_posix()
        for path in (
            PRODUCTS_PATH,
            CUSTOMER_PATH,
            SYSTEM_PROMPT_PATH,
            USER_PROMPT_PATH,
        )
        if not path.is_file()
    ]
    if missing_files:
        raise ValueError(f"缺少本地输入文件：{'、'.join(missing_files)}")


def build_user_prompt(hard_filter_result: dict, customer: dict) -> str:
    template = USER_PROMPT_PATH.read_text(encoding="utf-8")
    candidate_catalog = {
        "product_count": hard_filter_result["candidate_count"],
        "products": [
            {
                field: product.get(field)
                for field in RANKING_PRODUCT_FIELDS
            }
            for product in hard_filter_result["candidates"]
        ],
    }
    filter_summary = {
        "total_product_count": hard_filter_result["total_product_count"],
        "candidate_count": hard_filter_result["candidate_count"],
        "excluded_count": hard_filter_result["excluded_count"],
        "excluded_product_ids": [
            product["product_id"]
            for product in hard_filter_result["excluded_products"]
        ],
    }
    replacements = {
        "{{LOCAL_FILTER_SUMMARY_JSON}}": filter_summary,
        "{{PRODUCTS_JSON}}": candidate_catalog,
        "{{CUSTOMER_JSON}}": customer,
    }
    for placeholder, value in replacements.items():
        template = template.replace(
            placeholder,
            json.dumps(value, ensure_ascii=False, indent=2),
        )
    template = template.replace("{{CANDIDATE_COUNT}}", str(hard_filter_result["candidate_count"]))
    return template


def parse_recommendation(text: str, expected_product_ids: set[str]) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        cleaned = cleaned.removeprefix("```json").removeprefix("```")
        cleaned = cleaned.removesuffix("```").strip()

    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"模型没有返回合法 JSON：{text}") from exc

    if not isinstance(result, dict):
        raise RuntimeError("模型推荐结果必须是 JSON 对象")
    top3 = result.get("top3")
    if not isinstance(top3, list):
        raise RuntimeError(f"top3 必须是数组，实际类型：{type(top3).__name__}")
    expected_top_count = min(3, len(expected_product_ids))
    if len(top3) != expected_top_count:
        raise RuntimeError(f"Top3 数量必须为 {expected_top_count}，实际为 {len(top3)}")
    if any(not isinstance(item, dict) for item in top3):
        raise RuntimeError("top3 数组元素必须是 JSON 对象")
    ids = [item.get("product_id") for item in top3]
    if any(not isinstance(product_id, str) for product_id in ids):
        raise RuntimeError("top3 的 product_id 必须是字符串")
    if len(set(ids)) != len(ids):
        raise RuntimeError("Top3 含重复产品")
    if not set(ids).issubset(expected_product_ids):
        raise RuntimeError("Top3 包含候选集合之外的产品")
    if [item.get("rank") for item in top3] != list(range(1, expected_top_count + 1)):
        raise RuntimeError("Top3 的 rank 必须从 1 开始连续排列")
    scores = [item.get("match_score") for item in top3]
    if any(not isinstance(score, (int, float)) or isinstance(score, bool) or not 0 <= score <= 100 for score in scores):
        raise RuntimeError("Top3 的 match_score 必须是 0 到 100 的数值")
    if scores != sorted(scores, reverse=True):
        raise RuntimeError("Top3 未按 match_score 从高到低排列")
    return result


def preserve_model_recommendation(recommendation: dict, products_by_id: dict) -> dict:
    """仅保留推荐协议字段，保持 Top3 的顺序、分数与理由并规范名称。"""
    result = {key: copy.deepcopy(recommendation[key]) for key in ("customer_id", "top3", "disclaimer") if key in recommendation}
    detail_fields = {
        "rank", "product_id", "product_name", "bank_name", "match_score", "confidence",
        "amount_range", "annual_rate_range", "term_months_max", "reasons", "missing_information",
    }
    for row in result["top3"]:
        product = products_by_id[row["product_id"]]
        for key in set(row) - detail_fields:
            row.pop(key)
        row.update({"product_name": product.get("产品名称"), "bank_name": product.get("银行名称")})
    return result
