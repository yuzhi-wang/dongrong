"""贷款产品推荐入口：本地硬过滤后，单次调用 Qwen 完成候选排序。"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dashscope_client import call_responses, extract_output_text
from hard_filter import filter_products


ROOT_DIR = Path(__file__).resolve().parent
PRODUCTS_PATH = ROOT_DIR / "data" / "loan_products.json"
CUSTOMER_PATH = ROOT_DIR / "data" / "test_customer.json"
SYSTEM_PROMPT_PATH = ROOT_DIR / "prompts" / "loan_recommendation_system.md"
USER_PROMPT_PATH = ROOT_DIR / "prompts" / "loan_recommendation_user.md"

API_BASE_URL = os.getenv("DASHSCOPE_API_BASE_URL", "https://dashscope.aliyuncs.com")
API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
MODEL = os.getenv("DASHSCOPE_MODEL", "qwen3.7-max")
TIMEOUT_SECONDS = 180

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
    if not API_KEY:
        raise ValueError("请先设置环境变量 DASHSCOPE_API_KEY")
    if not API_BASE_URL.startswith(("http://", "https://")):
        raise ValueError("DASHSCOPE_API_BASE_URL 必须是 HTTP(S) 地址")

    missing_files = [
        path.relative_to(ROOT_DIR).as_posix()
        for path in (PRODUCTS_PATH, CUSTOMER_PATH, SYSTEM_PROMPT_PATH, USER_PROMPT_PATH)
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

    expected_count = len(expected_product_ids)
    rankings = result.get("rankings") or []
    ranked_ids = [item.get("product_id") for item in rankings]
    ranks = [item.get("rank") for item in rankings]
    if result.get("ranked_product_count") != expected_count:
        raise RuntimeError(
            f"模型未完成全量候选排序：期望 {expected_count}，"
            f"实际 {result.get('ranked_product_count')}"
        )
    if set(ranked_ids) != expected_product_ids or len(ranked_ids) != expected_count:
        raise RuntimeError("rankings 未完整覆盖全部唯一候选产品 ID")
    if ranks != list(range(1, expected_count + 1)):
        raise RuntimeError("rankings 的 rank 必须从 1 开始连续排列")
    if any(
        not isinstance(item.get("match_score"), (int, float))
        or isinstance(item.get("match_score"), bool)
        or not 0 <= item["match_score"] <= 100
        for item in rankings
    ):
        raise RuntimeError("rankings 的 match_score 必须是 0 到 100 的数值")
    ranking_scores = [item["match_score"] for item in rankings]
    if ranking_scores != sorted(ranking_scores, reverse=True):
        raise RuntimeError("rankings 未按 match_score 从高到低排序")

    top3 = result.get("top3") or []
    expected_top_count = min(3, expected_count)
    top3_ids = [item.get("product_id") for item in top3]
    if len(top3) != expected_top_count:
        raise RuntimeError(f"Top3 数量必须为 {expected_top_count}")
    if top3_ids != ranked_ids[:expected_top_count]:
        raise RuntimeError("Top3 必须与 rankings 的前三名完全一致")
    if [item.get("rank") for item in top3] != list(range(1, expected_top_count + 1)):
        raise RuntimeError("Top3 的 rank 必须从 1 开始连续排列")
    return result


def main() -> int:
    try:
        validate_inputs()
        catalog = load_json(PRODUCTS_PATH)
        customer = load_json(CUSTOMER_PATH)
        hard_filter_result = filter_products(catalog, customer)
        candidates = hard_filter_result["candidates"]
        if not candidates:
            raise ValueError("本地硬规则过滤后没有剩余候选产品")

        response = call_responses(
            api_base_url=API_BASE_URL,
            api_key=API_KEY,
            model=MODEL,
            instructions=SYSTEM_PROMPT_PATH.read_text(encoding="utf-8"),
            input_text=build_user_prompt(hard_filter_result, customer),
            timeout_seconds=TIMEOUT_SECONDS,
        )
        recommendation = parse_recommendation(
            extract_output_text(response),
            expected_product_ids={product["产品唯一ID"] for product in candidates},
        )
    except (ValueError, RuntimeError) as exc:
        print(f"执行失败：{exc}", file=sys.stderr)
        return 1

    print("\n===== 本地硬规则过滤 =====")
    print(json.dumps({
        "total_product_count": hard_filter_result["total_product_count"],
        "candidate_count": hard_filter_result["candidate_count"],
        "excluded_count": hard_filter_result["excluded_count"],
        "excluded_products": hard_filter_result["excluded_products"],
    }, ensure_ascii=False, indent=2))

    print("\n===== Qwen 候选产品推荐结果 =====")
    print(json.dumps(recommendation, ensure_ascii=False, indent=2))

    usage = response.get("usage") or {}
    print("\n===== 调用信息 =====")
    print(f"response_id: {response.get('id', '')}")
    print(f"model: {response.get('model', '')}")
    print(f"status: {response.get('status', '')}")
    print(f"input_tokens: {usage.get('input_tokens', '')}")
    print(f"output_tokens: {usage.get('output_tokens', '')}")
    print(f"total_tokens: {usage.get('total_tokens', '')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
