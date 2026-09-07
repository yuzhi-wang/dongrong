"""用真实用例评估本地硬过滤与 Qwen Top3 推荐。

默认从可正常验证组中按真实产品分层抽取 12 条，每款产品至少一条。
模型调用前只使用客户 ``input``；调用完成后才用 ``validation_target``
模型只返回 Top3，计算 Top1、Recall@3 和硬规则误删率。
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from api_key import load_api_key
from dashscope_client import MAX_RATE_LIMIT_RETRIES, RequestPacer, call_recommendation, extract_output_text
from report_comparison import (
    DEFAULT_PROMPT_BASELINE_PATH, compare_prompt_reports, compare_reports,
    print_comparison, validate_report,
)
from hard_filter import filter_products
from html_report import save_html_report
from recommendation import (
    API_BASE_URL,
    MODEL,
    PRODUCTS_PATH,
    SYSTEM_PROMPT_PATH,
    USER_PROMPT_PATH,
    TIMEOUT_SECONDS,
    build_user_prompt,
    load_json,
    parse_recommendation,
    preserve_model_recommendation,
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
        customer_ids = list(dict.fromkeys(customer_ids))
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
        by_product.append({
            "product_id": product_id,
            "product_name": group[0]["validation_target"].get("product_name"),
            "case_count": len(group),
            "target_retained_count": sum(
                bool(item.get("target_retained")) for item in group
            ),
            "completed_count": len(group_completed),
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
    # Windows 文件扫描或索引可能短暂占用目标文件；保留原子替换并有限重试。
    for attempt in range(5):
        try:
            temporary_path.replace(path)
            break
        except PermissionError as exc:
            if getattr(exc, "winerror", None) not in {5, 32, 33} or attempt == 4:
                raise
            time.sleep(0.1 * (attempt + 1))


def _customer_payload(case: dict[str, Any]) -> dict[str, Any]:
    # 明确只复制贷前 input；validation_target 不进入过滤器或模型提示词。
    return {"customer_id": case["customer_id"], **case["input"]}


def evaluate_case(
    case: dict[str, Any],
    catalog: dict[str, Any],
    system_prompt: str,
    *,
    dry_run: bool,
    api_key: str | None = None,
    before_request=None,
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
        response = call_recommendation(
            api_base_url=API_BASE_URL,
            api_key=api_key or load_api_key(),
            model=MODEL,
            instructions=system_prompt,
            input_text=build_user_prompt(hard_filter_result, customer),
            timeout_seconds=TIMEOUT_SECONDS,
            before_request=before_request,
        )
        result["_raw_response"] = response
        result["response"] = {key: response.get(key) for key in ("id", "model", "status", "api_format", "usage")}
        model_recommendation = parse_recommendation(
            extract_output_text(response),
            expected_product_ids=candidate_ids,
        )
        if model_recommendation.get("customer_id") != customer_id:
            raise RuntimeError(
                "模型返回的 customer_id 与当前测试客户不一致："
                f"{model_recommendation.get('customer_id')!r}"
            )
        products_by_id = {product["产品唯一ID"]: product for product in hard_filter_result["candidates"]}
        recommendation = preserve_model_recommendation(model_recommendation, products_by_id)
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
            "api_format": response.get("api_format"),
            "usage": response.get("usage") or {},
        },
        "recommendation": recommendation,
    })
    return result


def parse_args(argv: list[str] | None = None, *, default_all=False,
               default_workers=1, default_baseline=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="默认并行回测全部可验证客户，并比较历史全量基线。" if default_all else __doc__
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--sample-size",
        type=int,
        help="只回测指定数量的分层样本" if default_all else f"分层样本数（默认 {DEFAULT_SAMPLE_SIZE}）",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="抽样种子")
    selection.add_argument("--all", action="store_true", help="测试全部可验证用例")
    selection.add_argument(
        "--customer-id",
        action="append",
        default=[],
        help="只测试指定客户，可重复传入",
    )
    parser.add_argument("--dry-run", action="store_true", help="只运行抽样和硬过滤")
    parser.add_argument("--output", type=Path, help="指定结果 JSON 路径")
    parser.add_argument("--workers", type=int, default=default_workers, help="并行工作线程数")
    parser.add_argument("--request-interval", type=float, default=0.25, help="全局请求启动间隔，单位秒，默认 0.25")
    comparison = parser.add_mutually_exclusive_group()
    comparison.add_argument("--baseline", type=Path, default=default_baseline, help="历史基线报告路径")
    comparison.add_argument("--no-compare", action="store_true", help="不生成历史比较报告")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers 必须大于 0")
    if not math.isfinite(args.request_interval) or args.request_interval < 0:
        parser.error("--request-interval 必须是非负有限数值")
    if args.sample_size is not None and args.sample_size < 1:
        parser.error("--sample-size 必须大于 0")
    if default_all and args.sample_size is None and not args.customer_id:
        args.all = True
    args.sample_size = args.sample_size or DEFAULT_SAMPLE_SIZE
    if args.no_compare:
        args.baseline = None
    return args


def main(argv: list[str] | None = None, *, default_all=False,
         default_workers=1, default_baseline=None) -> int:
    args = parse_args(argv, default_all=default_all, default_workers=default_workers,
                      default_baseline=default_baseline)
    try:
        api_key = None if args.dry_run else load_api_key()
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
        customer_document = load_json(VALIDATABLE_CASES_PATH)
        system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
        baseline = load_json(args.baseline) if args.baseline else None
        prompt_baseline = (
            load_json(DEFAULT_PROMPT_BASELINE_PATH)
            if baseline is not None and DEFAULT_PROMPT_BASELINE_PATH.is_file() else None
        )
        if prompt_baseline is not None:
            validate_report(prompt_baseline)
        if baseline is not None:
            validate_report(baseline)
            baseline_check = compare_reports(baseline, baseline)
            if baseline_check["status"] != "comparable":
                raise ValueError("基线不可用：" + "；".join(baseline_check["reasons"]))
        output_path = (args.output or _default_output_path(args.dry_run)).resolve()
        if output_path.suffix.lower() != ".json":
            raise ValueError("--output 请指定 .json 文件；同名 .html 报告会自动生成")
        comparison_path = output_path.with_name(output_path.stem + "_comparison.json")
        html_path = output_path.with_suffix(".html")
        raw_response_dir = output_path.with_name(output_path.stem + "_raw")
        if args.baseline and args.baseline.resolve() in {output_path, comparison_path, html_path}:
            raise ValueError("输出路径不能覆盖历史基线")
        if DEFAULT_PROMPT_BASELINE_PATH.resolve() in {output_path, comparison_path, html_path}:
            raise ValueError("输出路径不能覆盖提示词实验基线")
    except (OSError, ValueError) as exc:
        print(f"测试准备失败：{exc}", file=sys.stderr)
        return 1

    report: dict[str, Any] = {
        "run": {
            "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "mode": "dry_run" if args.dry_run else "real",
            "source_group": "validatable",
            "data_sources": {
                "customers_path": str(VALIDATABLE_CASES_PATH),
                "customers_origin": customer_document.get("source"),
                "available_customer_count": len(cases),
                "products_path": str(PRODUCTS_PATH),
                "products_origin": catalog.get("source"),
                "product_count": len(catalog.get("products", [])),
                "system_prompt_path": str(SYSTEM_PROMPT_PATH),
                "user_prompt_path": str(USER_PROMPT_PATH),
            },
            "selection": (
                "explicit_customer_ids"
                if args.customer_id
                else "all"
                if args.all
                else "stratified_by_target_product"
            ),
            "sample_size": len(selected),
            "seed": args.seed,
            "workers": args.workers,
            "request_interval_seconds": args.request_interval,
            "max_rate_limit_retries": MAX_RATE_LIMIT_RETRIES,
            "model": None if args.dry_run else MODEL,
            "output_contract": "top3_only",
            "raw_response_directory": str(raw_response_dir) if not args.dry_run else None,
            "api_format": None if args.dry_run else "chat_completions_json",
            "thinking_budget": None if args.dry_run else 4096,
            "timeout_seconds": None if args.dry_run else TIMEOUT_SECONDS,
            "api_base_url": None if args.dry_run else API_BASE_URL,
        },
        "summary": {},
        "results": [],
    }

    print(f"将处理 {len(selected)} 条用例，模型 {'不调用（离线）' if args.dry_run else MODEL}，"
          f"并行数 {args.workers}，结果写入：{output_path}", flush=True)
    print("模型仅返回 Top3，按原始推荐统计命中率。", flush=True)
    report["summary"] = summarize([])
    save_report(report, output_path)
    pacer = RequestPacer(args.request_interval)
    completed_results = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                evaluate_case, case, catalog, system_prompt,
                dry_run=args.dry_run, api_key=api_key, before_request=pacer,
            ): index
            for index, case in enumerate(selected)
        }
        for future in as_completed(futures):
            index = futures[future]
            result = future.result()
            raw_response = result.pop("_raw_response", None)
            if raw_response is not None:
                raw_path = raw_response_dir / f"{index + 1:03d}.json"
                save_report({"customer_id": result["customer_id"], "response": raw_response}, raw_path)
                result["response"]["raw_path"] = str(raw_path)
            completed_results[index] = result
            # 只有主线程写报告，结果始终按输入顺序排列。
            report["results"] = [completed_results[i] for i in sorted(completed_results)]
            report["summary"] = summarize(report["results"])
            save_report(report, output_path)
            detail = result.get("error", "")
            if result["status"] == "completed":
                detail = (f"Top1={'是' if result['top1_hit'] else '否'}，"
                          f"Top3={'是' if result['top3_hit'] else '否'}，"
                          f"耗时={result['elapsed_seconds']}s")
            print(f"[{len(completed_results)}/{len(selected)}] {result['customer_id']} "
                  f"{result['status']} {detail}", flush=True)

    report["run"]["finished_at"] = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )
    report["summary"] = summarize(report["results"])
    save_report(report, output_path)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))

    comparison = None
    if baseline is not None:
        comparison = compare_reports(baseline, report)
        if prompt_baseline is not None:
            comparison["prompt_comparison"] = compare_prompt_reports(prompt_baseline, report)
            comparison["prompt_comparison"]["baseline_path"] = str(DEFAULT_PROMPT_BASELINE_PATH)
        comparison["baseline_path"] = str(args.baseline.resolve())
        comparison["current_path"] = str(output_path)
        save_report(comparison, comparison_path)
        print_comparison(comparison)
        print(f"比较报告：{comparison_path}")

    save_html_report(report, html_path, result_path=output_path,
                     cases=selected, catalog=catalog, comparison=comparison)
    print(f"HTML 报告：{html_path}")

    # 批量测试允许个别业务失败并完整保存报告；仅准备失败时返回非零。
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
