"""生成可离线打开的独立 HTML 回测报告，不加载外部脚本或样式。"""

from __future__ import annotations

import json
import os
from html import escape
from pathlib import Path
from urllib.parse import quote


STATUS = {
    "completed": "推荐完成", "failed": "调用 / 解析失败",
    "hard_filter_false_negative": "真实产品被硬规则排除",
    "dry_run": "离线检查 · 未调用模型", "pending": "等待处理",
}

CSS = """
:root{color-scheme:light;--ink:#172b42;--muted:#63758a;--line:#dce4ec;--brand:#126b65}
*{box-sizing:border-box}body{margin:0;background:#f2f5f8;color:var(--ink);font:15px/1.7 system-ui,'Microsoft YaHei',sans-serif}
main{max-width:1280px;margin:auto;padding:40px 28px 64px}header{margin-bottom:28px}
h1{font-size:34px;line-height:1.3;margin:8px 0}h2{font-size:21px;margin:0 0 18px}h3{font-size:16px;margin:0 0 10px}
p{margin:8px 0}.eyebrow{font-size:12px;letter-spacing:2px;color:var(--brand);font-weight:700}
.muted,small{color:var(--muted)}a{color:var(--brand);overflow-wrap:anywhere}nav{display:flex;gap:20px;flex-wrap:wrap;margin-top:18px}
section{background:white;border:1px solid var(--line);border-radius:16px;padding:26px;margin:22px 0}
.cards{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px}.card{background:#eaf4f2;border-radius:12px;padding:20px}
.card strong{display:block;font-size:29px;line-height:1.5}.notice{border-left:4px solid #d59331;background:#fff8eb;padding:12px 16px;margin:18px 0}
.scroll{overflow:auto}table{width:100%;border-collapse:collapse;text-align:left}th,td{padding:12px;border-bottom:1px solid var(--line);vertical-align:top}th{font-size:13px;color:var(--muted);white-space:nowrap}
dl{display:grid;grid-template-columns:150px minmax(0,1fr);margin:0}dt,dd{padding:8px 0;margin:0;border-bottom:1px solid var(--line);overflow-wrap:anywhere}
dt{color:var(--muted)}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f7fa;padding:16px;border-radius:8px;font-size:13px;max-height:420px;overflow:auto}
details{border:1px solid var(--line);border-radius:10px;padding:13px 16px;margin:10px 0}summary{cursor:pointer;overflow-wrap:anywhere}summary:hover{color:var(--brand)}
.badge{display:inline-block;border-radius:5px;background:#eaf4f2;color:#126b65;padding:1px 8px;font-size:12px;margin:0 8px}.bad{background:#fff0e7;color:#a44b17}
.recommendations{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin:16px 0}.recommendation{background:#f5f8fa;padding:16px;border-radius:10px;overflow-wrap:anywhere}.recommendation ul{padding-left:20px}
.toolbar{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:14px}input,select{font:inherit;padding:9px 12px;border:1px solid #b9c9d7;border-radius:8px;background:white;color:var(--ink)}input{min-width:260px;flex:1}
[hidden]{display:none!important}footer{font-size:13px;color:var(--muted)}
@media(max-width:760px){main{padding:22px 14px}section{padding:18px}.cards{grid-template-columns:repeat(2,minmax(0,1fr))}.recommendations{grid-template-columns:1fr}h1{font-size:27px}dl{grid-template-columns:110px minmax(0,1fr)}}
@media print{body{background:white}main{padding:0}nav,.toolbar{display:none}section{break-inside:avoid}.scroll{overflow:visible}pre{max-height:none}details{break-inside:avoid}}
"""

SCRIPT = """
const search = document.getElementById('search');
const filter = document.getElementById('filter');
const rows = [...document.querySelectorAll('.customer')];
function update() {
  const query = search.value.trim().toLocaleLowerCase();
  let count = 0;
  for (const row of rows) {
    const visible = row.textContent.toLocaleLowerCase().includes(query)
      && (!filter.value || row.dataset.status === filter.value
          || (filter.value === 'miss' && row.dataset.miss === 'true'));
    row.hidden = !visible;
    if (visible) count++;
  }
  document.getElementById('visible-count').textContent = `显示 ${count} / ${rows.length} 位客户`;
}
search.addEventListener('input', update);
filter.addEventListener('change', update);
update();
"""


def _text(value) -> str:
    return escape(str(value) if value is not None else "—", quote=True)


def _json(value) -> str:
    return "<pre>" + _text(json.dumps(value, ensure_ascii=False, indent=2)) + "</pre>"


def _rate(value) -> str:
    return f"{value:.2%}" if isinstance(value, (int, float)) else "未评估"


def _list_items(value) -> str:
    # 推荐解析器未强制这些可选字段的类型，展示层兼容空值或单段文字。
    items = value if isinstance(value, list) else [] if value is None else [value]
    return "".join(f"<li>{_text(item)}</li>" for item in items) or '<li class="muted">未提供</li>'


def _fields(items) -> str:
    return "<dl>" + "".join(f"<dt>{_text(k)}</dt><dd>{v}</dd>" for k, v in items) + "</dl>"


def _table(headers, rows) -> str:
    return ('<div class="scroll"><table><thead><tr>'
            + "".join(f"<th>{_text(h)}</th>" for h in headers)
            + "</tr></thead><tbody>"
            + "".join("<tr>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>" for row in rows)
            + "</tbody></table></div>")


def _link(path: Path, html_path: Path, label: str) -> str:
    try:
        url = quote(Path(os.path.relpath(path.resolve(), html_path.parent.resolve())).as_posix(), safe="/")
    except ValueError:  # Windows 跨盘文件。
        url = path.resolve().as_uri()
    return f'<a href="{_text(url)}">{_text(label)}</a>'


def _comparison(item: dict, title: str) -> str:
    body = f"<h3>{_text(title)}</h3>"
    if item.get("status") != "comparable":
        body += '<p class="notice">暂不判断提升：' + _text("；".join(item.get("reasons", []))) + "</p>"
    else:
        rows = []
        for key, metric in item.get("metrics", {}).items():
            values = []
            for side in ("baseline", "current"):
                m = metric[side]
                values.append(f"{_text(m['hits'])} / {_text(m['denominator'])}（{_rate(m['rate'])}）")
            rows.append([_text({"top1": "Top1", "recall_at_3": "Recall@3"}.get(key, key)),
                         *values, f"{metric['delta_percentage_points']:+.2f} 个百分点"])
        body += _table(["指标", "基线", "本次", "变化"], rows)
    body += "".join(f'<p class="muted">{_text(note)}</p>' for note in item.get("notes", []))
    if item.get("baseline_path"):
        body += f'<p class="muted">基线来源：{_text(item["baseline_path"])}</p>'
    return body


def _customer(row: dict, case: dict) -> str:
    status = row.get("status", "pending")
    target = row["validation_target"]
    recommendation = row.get("recommendation") or {}
    hit = ("Top3 命中" if row.get("top3_hit") else "Top3 未命中") if status == "completed" else "未产生模型推荐"
    top3 = recommendation.get("top3", [])
    names = " → ".join(str(p.get("product_name") or p.get("product_id")) for p in top3)
    body = (f'<details class="customer" data-status="{_text(status)}" '
            f'data-miss="{str(status == "completed" and not row.get("top3_hit")).lower()}">'
            f'<summary><b>{_text(row["customer_id"])}</b><span class="badge {"" if status == "completed" else "bad"}">'
            f'{_text(STATUS.get(status, status))}</span>{_text(hit)} · 真实产品：'
            f'{_text(target.get("product_name") or target["product_id"])}'
            + (f'<br><small>推荐顺序：{_text(names)}</small>' if names else "") + "</summary>")
    body += _fields([
        ("候选产品数", _text(row.get("hard_filter", {}).get("candidate_count"))),
        ("处理耗时", _text(row.get("elapsed_seconds")) + " 秒"),
        ("真实产品名次", _text(row.get("target_rank"))),
    ])
    if row.get("error"):
        body += f'<p class="notice">失败原因：{_text(row["error"])}</p>'
    exclusion = row.get("hard_filter", {}).get("target_exclusion")
    if exclusion:
        body += "<h3>硬规则排除原因</h3>" + _json(exclusion)
    body += '<div class="recommendations">'
    for product in top3:
        body += (f'<article class="recommendation"><h3>#{_text(product.get("rank"))} '
                 f'{_text(product.get("product_name") or product.get("product_id"))}</h3>'
                 f'<p>{_text(product.get("bank_name"))} · 匹配分 {_text(product.get("match_score"))}</p>'
                 f'<p class="muted">置信度：{_text({"high": "高", "medium": "中", "low": "低"}.get(str(product.get("confidence"))))}</p>')
        for key, label in (("reasons", "推荐理由"), ("missing_information", "待确认信息")):
            body += f"<b>{label}</b><ul>" + _list_items(product.get(key)) + "</ul>"
        body += "<details><summary>完整推荐字段</summary>" + _json(product) + "</details></article>"
    body += "</div>"
    if recommendation.get("disclaimer"):
        body += f'<p class="muted">{_text(recommendation["disclaimer"])}</p>'
    body += "<details><summary>本次客户输入与验证标签</summary>" + _json(case) + "</details>"
    body += "<details><summary>完整结果记录</summary>" + _json(row) + "</details></details>"
    return body


def save_html_report(report: dict, path: Path, *, result_path: Path,
                     cases: list[dict], catalog: dict, comparison: dict | None = None) -> None:
    """仅消费本次运行数据；HTML 内嵌结果和输入快照，可单文件离线阅读。"""
    run, summary = report["run"], report["summary"]
    model, hard = summary["model"], summary["hard_filter"]
    offline = run.get("mode") == "dry_run"
    body = ('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>贷款推荐回测报告 · {_text(run.get("finished_at"))}</title><style>{CSS}</style></head><body><main>'
            '<header><div class="eyebrow">DONGRONG / EVALUATION REPORT</div><h1>贷款产品推荐回测报告</h1>'
            f'<p class="muted">{"离线检查 · 未调用模型" if offline else _text(run.get("model"))} · '
            f'完成时间 {_text(run.get("finished_at"))}</p><nav><a href="#results">最终结果</a>'
            '<a href="#sources">数据来源</a><a href="#customers">逐客户详情</a>'
            + _link(result_path, path, "原始结果 JSON") + '</nav></header><section id="results"><h2>最终结果</h2><div class="cards">')
    for label, value, note in (
        ("本次客户", summary["case_count"], f'完成推荐 {model["completed_count"]} 位'),
        ("Top1 命中率", _rate(summary["end_to_end"]["top1_accuracy"]), f'命中 {model["top1_hits"]} / {summary["case_count"]}'),
        ("Recall@3", _rate(summary["end_to_end"]["recall_at_3"]), f'命中 {model["top3_hits"]} / {summary["case_count"]}'),
        ("需关注客户", model["failed_count"] + hard["false_negative_count"],
         f'失败 {model["failed_count"]} · 硬规则排除 {hard["false_negative_count"]}'),
    ):
        body += f'<div class="card">{label}<strong>{_text(value)}</strong><small>{"未评估模型命中" if offline and label in {"Top1 命中率", "Recall@3"} else _text(note)}</small></div>'
    body += '</div><p class="muted">Top1 为首个推荐命中真实产品；Recall@3 为前三个推荐包含真实产品。以上均以本次全部客户为分母，硬规则排除及调用失败计为未命中；只统计模型原始推荐。</p>'
    if offline or model["failed_count"]:
        body += '<p class="notice">' + ("本次仅验证本地输入和硬过滤，模型命中率未评估，不能据此判断推荐效果。" if offline else "本次存在调用或解析失败，命中率包含这些未命中项，暂不判断模型提升。") + '</p>'
    body += f'<p>仅成功客户：Top1 {_rate(model["top1_accuracy_on_completed"])} · Recall@3 {_rate(model["recall_at_3_on_completed"])}（分母 {model["completed_count"]}）。硬过滤保留真实产品 {hard["target_retained_count"]} / {summary["case_count"]}。</p></section>'
    body += '<section id="sources"><h2>数据来源</h2><p>客户贷前信息与产品库用于推荐；真实放贷产品标签仅用于本地核对。下方保留本次输入快照。</p>'
    sources = run.get("data_sources", {})
    labels = {"customers_path": "运行客户文件", "customers_origin": "客户上游来源", "available_customer_count": "可验证客户总数", "products_path": "运行产品文件", "products_origin": "产品上游来源", "product_count": "产品总数", "system_prompt_path": "系统提示词", "user_prompt_path": "用户提示词"}
    body += _fields([(labels.get(key, key), _json(value) if isinstance(value, dict) else _text(value)) for key, value in sources.items()])
    body += '<details><summary>查看本次产品库快照</summary>' + _json(catalog) + '</details>'
    body += '<details><summary>运行配置与抽样方式</summary>' + _json(run) + '</details></section>'
    body += '<section><h2>按真实产品统计</h2>' + _table(
        ["真实产品", "客户数", "完成推荐", "Top1 命中", "Top3 命中", "全组 Recall@3"],
        [[_text(p.get("product_name") or p["product_id"]), _text(p["case_count"]), _text(p["completed_count"]),
          _text(p["top1_hits"]) if not offline else "未评估", _text(p["top3_hits"]) if not offline else "未评估",
          _rate(p["end_to_end_recall_at_3"])] for p in summary["by_product"]]) + '</section>'
    body += '<section><h2>历史比较</h2>'
    if comparison is None:
        body += '<p class="muted">本次未启用历史基线比较。</p>'
    else:
        body += _comparison(comparison, "历史全量基线 · 模型原始排名")
        if comparison.get("prompt_comparison"):
            body += _comparison(comparison["prompt_comparison"], "同模型提示词比较 · 固定旧版成功客户子集")
        body += _link(result_path.with_name(result_path.stem + "_comparison.json"), path, "比较结果 JSON")
    body += '</section><section id="customers"><h2>逐客户最终结果</h2><div class="toolbar"><input id="search" type="search" aria-label="搜索客户、产品或理由" placeholder="搜索客户 ID、产品或推荐理由"><select id="filter" aria-label="按结果筛选"><option value="">全部结果</option><option value="completed">推荐完成</option><option value="miss">Top3 未命中（已完成）</option><option value="failed">调用 / 解析失败</option><option value="hard_filter_false_negative">硬规则排除</option><option value="dry_run">离线检查</option></select></div><p id="visible-count" class="muted" aria-live="polite"></p>'
    case_by_id = {c["customer_id"]: c for c in cases}
    body += "".join(_customer(row, case_by_id.get(row["customer_id"], {})) for row in report["results"])
    body += '</section><footer>本报告用于产品匹配与回测分析，最终额度、利率及审批结果以金融机构审核为准。页面可离线查看；来源路径及 JSON 链接对应本机文件。</footer>'
    body += f'</main><script>{SCRIPT}</script></body></html>'
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(body, encoding="utf-8")
    temporary.replace(path)
