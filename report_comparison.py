"""按相同客户集合比较全量回测，原回测 JSON 不添加对比字段。"""

from __future__ import annotations

from pathlib import Path


DEFAULT_BASELINE_PATH = (
    Path(__file__).resolve().parent / "outputs" / "validation_runs"
    / "validatable_b_all_93_with_history_rerank.json"
)
DEFAULT_PROMPT_BASELINE_PATH = DEFAULT_BASELINE_PATH.with_name("validatable_real_20260906_185128.json")


def _cohort(report: dict) -> dict:
    rows = report.get("results") or []
    cohort = {r["customer_id"]: r["validation_target"]["product_id"] for r in rows}
    if len(cohort) != len(rows) or not rows:
        raise ValueError("对比报告的客户集合为空或存在重复 ID")
    return cohort


def _model_hit(row: dict, metric: str) -> bool:
    """旧报告读取原始模型命中标记；新报告只保留 top1_hit/top3_hit。"""
    legacy = f"pre_rerank_{metric}_hit"
    value = row[legacy] if legacy in row else row.get(f"{metric}_hit")
    if not isinstance(value, bool):
        raise ValueError(f"缺少有效模型命中标记 {metric}")
    return value


def validate_report(report: dict) -> None:
    """提前拒绝损坏的基线，避免完成收费调用后才发现无法比较。"""
    try:
        _cohort(report)
        if not isinstance(report["run"], dict):
            raise ValueError("run 必须是对象")
        for row in report["results"]:
            if row["status"] == "completed":
                for metric in ("top1", "top3"):
                    _model_hit(row, metric)
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("对比报告缺少必要的客户或模型结果字段") from exc


def _metrics(report: dict) -> dict:
    rows = report["results"]
    completed = [r for r in rows if r["status"] == "completed"]
    return {
        name: {
            "hits": sum(_model_hit(row, metric) for row in completed),
            "denominator": len(rows),
        }
        for name, metric in (("top1", "top1"), ("recall_at_3", "top3"))
    }



def compare_reports(baseline: dict, current: dict) -> dict:
    """比较同客户、同标签的原始模型推荐结果。"""
    validate_report(baseline)
    validate_report(current)
    reasons = []
    if _cohort(baseline) != _cohort(current):
        reasons.append("客户集合或真实产品标签不同，不能比较全量命中率")
    for label, report in (("基线", baseline), ("本次", current)):
        if report.get("run", {}).get("mode") != "real" or not report.get("run", {}).get("finished_at"):
            reasons.append(f"{label}不是已结束的真实回测")
        if any(r.get("status") not in {"completed", "hard_filter_false_negative"} for r in report["results"]):
            reasons.append(f"{label}存在失败或未完成用例，请补齐后再判断模型提升")
    old, new = _metrics(baseline), _metrics(current)
    metrics = {}
    if not reasons:
        for name in old:
            old_rate = old[name]["hits"] / old[name]["denominator"]
            new_rate = new[name]["hits"] / new[name]["denominator"]
            delta = new_rate - old_rate
            metrics[name] = {
                "baseline": {**old[name], "rate": round(old_rate, 6)},
                "current": {**new[name], "rate": round(new_rate, 6)},
                "delta_percentage_points": round(delta * 100, 4),
                "direction": "improved" if delta > 0 else "regressed" if delta < 0 else "unchanged",
            }
    return {
        "status": "comparable" if not reasons else "not_comparable",
        "metric_scope": "model_original_only",
        "reasons": reasons,
        "baseline_model": baseline.get("run", {}).get("model"),
        "current_model": current.get("run", {}).get("model"),
        "metrics": metrics,
        "notes": [
            "分母为全部验证客户，硬过滤排除项计为未命中。",
            "只比较两次模型原始 Top3，不读取旧报告中后处理产生的命中结果。",
            "比较的是整套运行配置；模型、提示词、接口和思考配置可能不同，不能将差异全部归因于模型版本。",
        ],
    }


def compare_prompt_reports(baseline: dict, current: dict) -> dict:
    """固定旧版成功客户集合，比对相同模型的原始排名；不混合两轮结果。"""
    validate_report(baseline)
    validate_report(current)
    reasons = []
    if _cohort(baseline) != _cohort(current):
        reasons.append("需要对相同全量客户执行新版回测")
    for label, report in (("旧版", baseline), ("新版", current)):
        if report["run"].get("mode") != "real" or not report["run"].get("finished_at"):
            reasons.append(f"{label}不是已结束的真实回测")
    for field in ("model", "api_format", "thinking_budget", "api_base_url"):
        if baseline["run"].get(field) != current["run"].get(field):
            reasons.append(f"两轮 {field} 不同，不作为同配置提示词实验")
    if any(r["status"] not in {"completed", "hard_filter_false_negative"} for r in current["results"]):
        reasons.append("新版仍有失败或未完成客户，先补齐新版结果")
    ids = {r["customer_id"] for r in baseline["results"] if r["status"] == "completed"}
    new_rows = [r for r in current["results"] if r["customer_id"] in ids]
    if not ids or len(new_rows) != len(ids) or any(r["status"] != "completed" for r in new_rows):
        reasons.append("旧版成功客户在新版中未全部完成模型调用")
    result = {"status": "not_comparable", "reasons": reasons, "metrics": {}}
    if not reasons:
        result = compare_reports(
            {**baseline, "results": [r for r in baseline["results"] if r["customer_id"] in ids]},
            {**current, "results": new_rows},
        )
    result.update({
        "scope": "baseline_completed_customers",
        "customer_ids": sorted(ids),
        "excluded_baseline_customer_ids": sorted(set(_cohort(baseline))-ids),
        "notes": [
            "仅比较旧版成功的固定客户集合，两轮均取原始模型排名；这不是全量93条指标。",
            "旧版失败及硬过滤排除客户不进入此子集；新版新增成功结果仍计入新版全量报告。",
            "同一模型与请求配置下评估提示词变化，但生成随机性及服务端模型更新仍可能影响结果。",
        ],
    })
    return result


def print_comparison(comparison: dict) -> None:
    print("\n===== 与历史全量基线比较 =====")
    if comparison["status"] != "comparable":
        print("暂不判断提升：" + "；".join(comparison["reasons"]))
    if comparison.get("metric_scope") == "model_original_only":
        print("口径：两轮都取模型原始 Top3。")
    labels = {"top1": "Top1", "recall_at_3": "Recall@3"}
    for name, item in comparison["metrics"].items():
        old, new = item["baseline"], item["current"]
        print(
            f"{labels[name]}：{old['hits']}/{old['denominator']} ({old['rate']:.2%})"
            f" -> {new['hits']}/{new['denominator']} ({new['rate']:.2%})，"
            f"变化 {item['delta_percentage_points']:+.2f} 个百分点"
        )
    prompt = comparison.get("prompt_comparison")
    if prompt:
        print("\n===== 同模型提示词比较（固定旧版成功客户子集） =====")
        if prompt["status"] != "comparable":
            print("暂不判断提升：" + "；".join(prompt["reasons"]))
        else:
            for name, item in prompt["metrics"].items():
                old, new = item["baseline"], item["current"]
                print(f"{labels[name]}：{old['hits']}/{old['denominator']} ({old['rate']:.2%})"
                      f" -> {new['hits']}/{new['denominator']} ({new['rate']:.2%})，"
                      f"变化 {item['delta_percentage_points']:+.2f} 个百分点")
            print("该子集指标不代表全量93条结果。")
