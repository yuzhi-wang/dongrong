"""使用历史产品频率对 Qwen 的完整候选排名做轻量、确定性重排。"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


def load_history_prior(path: Path) -> dict[str, Any]:
    """读取并校验历史产品频率配置。"""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取有效历史先验 JSON：{path.name}") from exc

    sample_count = document.get("sample_count")
    weight = document.get("default_history_weight")
    products = document.get("products")
    if not isinstance(sample_count, int) or sample_count <= 0:
        raise ValueError("历史先验 sample_count 必须是正整数")
    if (
        not isinstance(weight, (int, float))
        or isinstance(weight, bool)
        or not 0 <= weight <= 1
    ):
        raise ValueError("历史先验 default_history_weight 必须在 0 到 1 之间")
    if not isinstance(products, list) or not products:
        raise ValueError("历史先验缺少 products 数组")

    seen_ids: set[str] = set()
    count_sum = 0
    for product in products:
        product_id = product.get("product_id")
        count = product.get("count")
        if not product_id or product_id in seen_ids:
            raise ValueError("历史先验产品 ID 为空或重复")
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError(f"历史先验产品 {product_id} 的 count 无效")
        seen_ids.add(product_id)
        count_sum += count
    if count_sum != sample_count:
        raise ValueError(
            f"历史先验样本数不一致：声明 {sample_count}，产品计数合计 {count_sum}"
        )
    return document


def _history_counts(
    prior: dict[str, Any],
    *,
    exclude_product_id: str | None,
) -> tuple[dict[str, int], int]:
    counts = {
        product["product_id"]: product["count"]
        for product in prior["products"]
    }
    sample_count = prior["sample_count"]
    if exclude_product_id is not None:
        count = counts.get(exclude_product_id, 0)
        if count <= 0:
            raise ValueError(f"无法从历史先验中留一扣除产品：{exclude_product_id}")
        counts[exclude_product_id] = count - 1
        sample_count -= 1
    return counts, sample_count


def _amount_text(product: dict[str, Any]) -> str | None:
    maximum = product.get("最高额度（元）")
    return f"最高{maximum:g}元" if isinstance(maximum, (int, float)) else None


def _rate_text(product: dict[str, Any]) -> str | None:
    minimum = product.get("最低年利率")
    maximum = product.get("最高年利率")
    if isinstance(minimum, (int, float)) and isinstance(maximum, (int, float)):
        return f"{minimum:.2%} - {maximum:.2%}"
    if isinstance(minimum, (int, float)):
        return f"最低{minimum:.2%}，最高年利率未知"
    if isinstance(maximum, (int, float)):
        return f"最高{maximum:.2%}，最低年利率未知"
    return None


def rerank_recommendation(
    recommendation: dict[str, Any],
    prior: dict[str, Any],
    products_by_id: dict[str, dict[str, Any]],
    *,
    history_weight: float | None = None,
    exclude_product_id: str | None = None,
) -> dict[str, Any]:
    """融合模型名次和历史频率；只重排现有候选，绝不恢复已排除产品。"""
    weight = (
        prior["default_history_weight"]
        if history_weight is None
        else history_weight
    )
    if (
        not isinstance(weight, (int, float))
        or isinstance(weight, bool)
        or not 0 <= weight <= 1
    ):
        raise ValueError("history_weight 必须在 0 到 1 之间")

    original_rankings = recommendation.get("rankings") or []
    if not original_rankings:
        raise ValueError("模型推荐结果缺少 rankings")
    candidate_ids = [item.get("product_id") for item in original_rankings]
    if any(not product_id for product_id in candidate_ids):
        raise ValueError("模型 rankings 中存在空产品 ID")
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("模型 rankings 中存在重复产品 ID")
    missing_products = [
        product_id for product_id in candidate_ids if product_id not in products_by_id
    ]
    if missing_products:
        raise ValueError(f"产品目录缺少候选：{'、'.join(missing_products)}")

    counts, history_sample_count = _history_counts(
        prior,
        exclude_product_id=exclude_product_id,
    )
    max_history_count = max(counts.values(), default=0)
    candidate_count = len(original_rankings)
    scored: list[dict[str, Any]] = []
    for item in original_rankings:
        product_id = item["product_id"]
        model_rank = item["rank"]
        model_rank_score = (
            (candidate_count - model_rank) / (candidate_count - 1)
            if candidate_count > 1
            else 1.0
        )
        history_count = counts.get(product_id, 0)
        history_prior_score = (
            history_count / max_history_count if max_history_count else 0.0
        )
        final_score = (
            (1 - weight) * model_rank_score
            + weight * history_prior_score
        )
        scored.append({
            "product_id": product_id,
            "model_rank": model_rank,
            "model_match_score": item["match_score"],
            "model_rank_score": model_rank_score,
            "history_count": history_count,
            "history_prior_score": history_prior_score,
            "final_score": final_score,
        })

    scored.sort(
        key=lambda item: (
            -item["final_score"],
            item["model_rank"],
            item["product_id"],
        )
    )
    final_rankings: list[dict[str, Any]] = []
    for rank, item in enumerate(scored, start=1):
        final_rankings.append({
            "rank": rank,
            "product_id": item["product_id"],
            "match_score": round(item["final_score"] * 100, 2),
            "model_rank": item["model_rank"],
            "model_match_score": item["model_match_score"],
            "model_rank_score": round(item["model_rank_score"], 6),
            "history_count": item["history_count"],
            "history_prior_score": round(item["history_prior_score"], 6),
        })

    original_top3 = {
        item["product_id"]: item for item in recommendation.get("top3") or []
    }
    final_top3: list[dict[str, Any]] = []
    for ranking in final_rankings[: min(3, candidate_count)]:
        product_id = ranking["product_id"]
        product = products_by_id[product_id]
        detail = copy.deepcopy(original_top3.get(product_id) or {
            "product_id": product_id,
            "bank_name": product.get("银行名称"),
            "product_name": product.get("产品名称"),
            "confidence": "medium",
            "amount_range": _amount_text(product),
            "annual_rate_range": _rate_text(product),
            "term_months_max": product.get("最长期限（月）"),
            "reasons": [
                f"模型完整候选排序为第{ranking['model_rank']}名，结合历史放款先验后进入最终Top3。"
            ],
            "missing_information": [],
        })
        detail.update({
            "rank": ranking["rank"],
            "match_score": ranking["match_score"],
            "model_rank": ranking["model_rank"],
            "model_match_score": ranking["model_match_score"],
            "history_count": ranking["history_count"],
            "history_prior_score": ranking["history_prior_score"],
        })
        final_top3.append(detail)

    result = copy.deepcopy(recommendation)
    result["ranking_method"] = "qwen_rank_plus_global_history_prior"
    result["history_rerank"] = {
        "history_weight": weight,
        "model_weight": 1 - weight,
        "history_sample_count": history_sample_count,
        "leave_one_out": exclude_product_id is not None,
        "model_top3_product_ids": [
            item.get("product_id") for item in recommendation.get("top3") or []
        ],
        "formula": "final=(1-history_weight)*normalized_model_rank+history_weight*normalized_history_count",
    }
    result["top3"] = final_top3
    result["rankings"] = final_rankings
    return result
