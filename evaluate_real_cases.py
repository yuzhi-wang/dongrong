"""用历史真实用例评估“本地硬过滤 + Qwen + 历史先验重排”流程。

默认从可正常验证组中按真实产品分层抽取 12 条，每款产品至少一条。
模型调用前只使用客户 ``input``；调用完成后才用 ``validation_target``
计算重排前后 Top1、Recall@3 和硬规则误删率。历史先验重排在模型
返回后执行，并从当前真实产品的历史计数中扣除1条，避免直接标签泄漏。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from dashscope_client import call_responses, extract_output_text
from hard_filter import filter_products
from history_rerank import load_history_prior, rerank_recommendation
from main import (
    API_BASE_URL,
    API_KEY,
    HISTORY_PRIOR_PATH,
    MODEL,
    PRODUCTS_PATH,
    SYSTEM_PROMPT_PATH,
    TIMEOUT_SECONDS,
    build_user_prompt,
    load_json,
    parse_recommendation,
)


ROOT_DIR = Path(__file__).resolve().parent
VALIDATABLE_CASES_PATH = (
    ROOT_DIR / "data" / "validation_groups" / "customers_validatable.json"
)
OUTPUT_DIR = ROOT_DIR / "outputs" / "validation_runs"
DEFAULT_SAMPLE_SIZE = 12
DEFAULT_SEED = 20260903


def load_cases(path: Path = VALIDATABLE_CASES_PATH) -> list[dict[str, Any]]:
    """读取并检查可验证用例文件的最小结构。"""
    document = load_json(path)
    cases = document.get("customers")
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"{path.name} 中没有可用的 customers 数组")

    seen_ids: set[str] = set()
    for case in cases:
        customer_id = case.get("customer_id")
        target = case.get("validation_target")
        if not customer_id or not isinstance(case.get("input"), dict):
            raise ValueError("用例缺少 customer_id 或 input")
        if not isinstance(target, dict) or not target.get("product_id"):
            raise ValueError(f"用例 {customer_id} 缺少 validation_target.product_id")
        if customer_id in seen_ids:
            raise ValueError(f"客户 ID 重复：{customer_id}")
        seen_ids.add(customer_id)
    return cases


def stratified_sample(
    cases: list[dict[str, Any]],
    sample_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    """按真实产品轮询抽样，优先让每个产品都进入样本。"""
    if sample_size <= 0:
        raise ValueError("sample_size 必须大于 0")
    if sample_size >= len(cases):
        return list(cases)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        grouped[case["validation_target"]["product_id"]].append(case)

    rng = random.Random(seed)
    for group in grouped.values():
        rng.shuffle(group)

    # 产品顺序固定后，相同 seed 的抽样结果可复现。
    product_ids = sorted(grouped)
    selected: list[dict[str, Any]] = []
    depth = 0
    while len(selected) < sample_size:
        added = False
        for product_id in product_ids:
            group = grouped[product_id]
            if depth < len(group):
                selected.append(group[depth])
                added = True
                if len(selected) == sample_size:
                    break
        if not added:
            break
        depth += 1
    return selected


def choose_cases(
    cases: list[dict[str, Any]],
    *,
    sample_size: int,
    seed: int,
    run_all: bool,
    customer_ids: list[str],
) -> list[dict[str, Any]]:
    """按显式客户、全量或分层抽样三种模式选择用例。"""
    if customer_ids:
        by_id = {case["customer_id"]: case for case in cases}
        missing = [customer_id for customer_id in customer_ids if customer_id not in by_id]
        if missing:
            raise ValueError(f"找不到客户：{'、'.join(missing)}")
        return [by_id[customer_id] for customer_id in customer_ids]
    if run_all:
        return list(cases)
    return stratified_sample(cases, sample_size, seed)


def _safe_rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总硬规则、模型内和端到端指标。"""
    total = len(results)
    retained = sum(bool(item.get("target_retained")) for item in results)
    false_negatives = total - retained
    completed = [item for item in results if item.get("status") == "completed"]
    failed = [item for item in results if item.get("status") == "failed"]
    not_exercised = [item for item in results if item.get("status") == "dry_run"]
    model_exercised = bool(completed or failed)
    top1_hits = sum(bool(item.get("top1_hit")) for item in completed)
    top3_hits = sum(bool(item.get("top3_hit")) for item in completed)
    pre_rerank_top1_hits = sum(
        bool(item.get("pre_rerank_top1_hit", item.get("top1_hit")))
        for item in completed
    )
    pre_rerank_top3_hits = sum(
        bool(item.get("pre_rerank_top3_hit", item.get("top3_hit")))
        for item in completed
    )

    usage_fields = ("input_tokens", "output_tokens", "total_tokens")
    total_usage = {
        field: sum(
            value
            for item in completed
            if isinstance((value := item.get("response", {}).get("usage", {}).get(field)), int)
        )
        for field in usage_fields
    }

    product_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in results:
        product_groups[item["validation_target"]["product_id"]].append(item)

    by_product: list[dict[str, Any]] = []
    for product_id in sorted(product_groups):
        group = product_groups[product_id]
        group_completed = [item for item in group if item.get("status") == "completed"]
        group_failed = [item for item in group if item.get("status") == "failed"]
        group_model_exercised = bool(group_completed or group_failed)
        group_top1 = sum(bool(item.get("top1_hit")) for item in group_completed)
        group_top3 = sum(bool(item.get("top3_hit")) for item in group_completed)
        group_pre_rerank_top1 = sum(
            bool(item.get("pre_rerank_top1_hit", item.get("top1_hit")))
            for item in group_completed
        )
        group_pre_rerank_top3 = sum(
            bool(item.get("pre_rerank_top3_hit", item.get("top3_hit")))
            for item in group_completed
        )
        by_product.append({
            "product_id": product_id,
            "product_name": group[0]["validation_target"].get("product_name"),
            "case_count": len(group),
            "target_retained_count": sum(
                bool(item.get("target_retained")) for item in group
            ),
            "completed_count": len(group_completed),
            "pre_rerank_top1_hits": group_pre_rerank_top1,
            "pre_rerank_top3_hits": group_pre_rerank_top3,
            "top1_hits": group_top1,
            "top3_hits": group_top3,
            "end_to_end_top1_accuracy": (
                _safe_rate(group_top1, len(group)) if group_model_exercised else None
            ),
            "end_to_end_recall_at_3": (
                _safe_rate(group_top3, len(group)) if group_model_exercised else None
            ),
        })

    return {
        "case_count": total,
        "hard_filter": {
            "target_retained_count": retained,
            "false_negative_count": false_negatives,
            "target_recall": _safe_rate(retained, total),
        },
        "model": {
            "completed_count": len(completed),
            "failed_count": len(failed),
            "not_exercised_count": len(not_exercised),
            "pre_rerank_top1_hits": pre_rerank_top1_hits,
            "pre_rerank_top3_hits": pre_rerank_top3_hits,
            "pre_rerank_top1_accuracy_on_completed": _safe_rate(
                pre_rerank_top1_hits,
                len(completed),
            ),
            "pre_rerank_recall_at_3_on_completed": _safe_rate(
                pre_rerank_top3_hits,
                len(completed),
            ),
            "top1_hits": top1_hits,
            "top3_hits": top3_hits,
            "top1_accuracy_on_completed": _safe_rate(top1_hits, len(completed)),
            "recall_at_3_on_completed": _safe_rate(top3_hits, len(completed)),
        },
        "end_to_end": {
            "top1_accuracy": _safe_rate(top1_hits, total) if model_exercised else None,
            "recall_at_3": _safe_rate(top3_hits, total) if model_exercised else None,
        },
        "usage": total_usage,
        "by_product": by_product,
    }


def _default_output_path(dry_run: bool) -> Path:
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    suffix = "dry_run" if dry_run else "real"
    return OUTPUT_DIR / f"validatable_{suffix}_{timestamp}.json"


def save_report(report: dict[str, Any], path: Path) -> None:
    """覆盖写入当前报告；每处理一条保存一次，避免中断后结果丢失。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _customer_payload(case: dict[str, Any]) -> dict[str, Any]:
    # 明确只复制贷前 input；validation_target 不进入过滤器或模型提示词。
    return {"customer_id": case["customer_id"], **case["input"]}


def evaluate_case(
    case: dict[str, Any],
    catalog: dict[str, Any],
    system_prompt: str,
    history_prior: dict[str, Any],
    *,
    dry_run: bool,
) -> dict[str, Any]:
    """评估一个客户；真实标签只在模型返回后用于比对。"""
    customer_id = case["customer_id"]
    customer = _customer_payload(case)
    hard_filter_result = filter_products(catalog, customer)
    candidate_ids = {
        product["产品唯一ID"] for product in hard_filter_result["candidates"]
    }
    target_product_id = case["validation_target"]["product_id"]
    target_retained = target_product_id in candidate_ids
    target_exclusion = next(
        (
            item
            for item in hard_filter_result["excluded_products"]
            if item["product_id"] == target_product_id
        ),
        None,
    )
    result: dict[str, Any] = {
        "customer_id": customer_id,
        "validation_target": case["validation_target"],
        "status": "dry_run" if dry_run else "pending",
        "target_retained": target_retained,
        "hard_filter": {
            "candidate_count": hard_filter_result["candidate_count"],
            "excluded_count": hard_filter_result["excluded_count"],
            "target_exclusion": target_exclusion,
        },
    }

    if not target_retained:
        result["status"] = "hard_filter_false_negative"
        return result
    if dry_run:
        return result
    if not candidate_ids:
        result["status"] = "failed"
        result["error"] = "本地硬规则过滤后没有剩余候选产品"
        return result

    started_at = time.perf_counter()
    try:
        response = call_responses(
            api_base_url=API_BASE_URL,
            api_key=API_KEY,
            model=MODEL,
            instructions=system_prompt,
            input_text=build_user_prompt(hard_filter_result, customer),
            timeout_seconds=TIMEOUT_SECONDS,
        )
        model_recommendation = parse_recommendation(
            extract_output_text(response),
            expected_product_ids=candidate_ids,
        )
        if model_recommendation.get("customer_id") != customer_id:
            raise RuntimeError(
                "模型返回的 customer_id 与当前测试客户不一致："
                f"{model_recommendation.get('customer_id')!r}"
            )
        pre_rerank_top3_ids = [
            item.get("product_id")
            for item in model_recommendation.get("top3", [])
        ]
        recommendation = rerank_recommendation(
            model_recommendation,
            history_prior,
            {
                product["产品唯一ID"]: product
                for product in hard_filter_result["candidates"]
            },
            exclude_product_id=target_product_id,
        )
        top3_ids = [item.get("product_id") for item in recommendation.get("top3", [])]
        if len(top3_ids) != len(set(top3_ids)):
            raise RuntimeError("模型返回的 Top3 含重复产品")
    except (ValueError, RuntimeError) as exc:
        result["status"] = "failed"
        result["error"] = str(exc)
        result["elapsed_seconds"] = round(time.perf_counter() - started_at, 3)
        return result

    result.update({
        "status": "completed",
        "elapsed_seconds": round(time.perf_counter() - started_at, 3),
        "pre_rerank_top1_hit": (
            bool(pre_rerank_top3_ids)
            and pre_rerank_top3_ids[0] == target_product_id
        ),
        "pre_rerank_top3_hit": target_product_id in pre_rerank_top3_ids,
        "top1_hit": bool(top3_ids) and top3_ids[0] == target_product_id,
        "top3_hit": target_product_id in top3_ids,
        "target_rank": (
            top3_ids.index(target_product_id) + 1
            if target_product_id in top3_ids
            else None
        ),
        "response": {
            "id": response.get("id"),
            "model": response.get("model"),
            "status": response.get("status"),
            "usage": response.get("usage") or {},
        },
        "recommendation": recommendation,
    })
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help=f"分层样本数（默认 {DEFAULT_SAMPLE_SIZE}）",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="抽样种子")
    parser.add_argument("--all", action="store_true", help="测试全部可验证用例")
    parser.add_argument(
        "--customer-id",
        action="append",
        default=[],
        help="只测试指定客户，可重复传入",
    )
    parser.add_argument("--dry-run", action="store_true", help="只运行抽样和硬过滤")
    parser.add_argument("--output", type=Path, help="指定结果 JSON 路径")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if not args.dry_run and not API_KEY:
            raise ValueError("请先设置环境变量 DASHSCOPE_API_KEY")
        if not API_BASE_URL.startswith(("http://", "https://")):
            raise ValueError("DASHSCOPE_API_BASE_URL 必须是 HTTP(S) 地址")

        cases = load_cases()
        selected = choose_cases(
            cases,
            sample_size=args.sample_size,
            seed=args.seed,
            run_all=args.all,
            customer_ids=args.customer_id,
        )
        catalog = load_json(PRODUCTS_PATH)
        history_prior = load_history_prior(HISTORY_PRIOR_PATH)
        system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    except (OSError, ValueError) as exc:
        print(f"测试准备失败：{exc}", file=sys.stderr)
        return 1

    output_path = (args.output or _default_output_path(args.dry_run)).resolve()
    report: dict[str, Any] = {
        "run": {
            "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "mode": "dry_run" if args.dry_run else "real",
            "source_group": "validatable",
            "selection": (
                "explicit_customer_ids"
                if args.customer_id
                else "all"
                if args.all
                else "stratified_by_target_product"
            ),
            "sample_size": len(selected),
            "seed": args.seed,
            "model": None if args.dry_run else MODEL,
            "api_base_url": None if args.dry_run else API_BASE_URL,
            "history_rerank": {
                "enabled": True,
                "weight": history_prior["default_history_weight"],
                "source_version": history_prior.get("version"),
                "validation_mode": "leave_one_out",
            },
        },
        "summary": {},
        "results": [],
    }

    print(f"将处理 {len(selected)} 条用例，结果写入：{output_path}")
    for index, case in enumerate(selected, start=1):
        target = case["validation_target"]
        print(
            f"[{index}/{len(selected)}] {case['customer_id']} -> "
            f"{target['product_name']} ({target['product_id']})",
            flush=True,
        )
        result = evaluate_case(
            case,
            catalog,
            system_prompt,
            history_prior,
            dry_run=args.dry_run,
        )
        report["results"].append(result)
        report["summary"] = summarize(report["results"])
        save_report(report, output_path)
        if result["status"] == "completed":
            print(
                f"  完成：Top1={'是' if result['top1_hit'] else '否'}，"
                f"Top3={'是' if result['top3_hit'] else '否'}，"
                f"耗时={result['elapsed_seconds']}s",
                flush=True,
            )
        elif result["status"] == "hard_filter_false_negative":
            print("  停止：真实产品被硬规则排除", flush=True)
        elif result["status"] == "failed":
            print(f"  失败：{result['error']}", flush=True)

    report["run"]["finished_at"] = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )
    report["summary"] = summarize(report["results"])
    save_report(report, output_path)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))

    # 批量测试允许个别业务失败并完整保存报告；仅准备失败时返回非零。
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
