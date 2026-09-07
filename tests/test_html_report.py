from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import main
from evaluate_real_cases import summarize
from html_report import save_html_report


class HtmlReportTest(unittest.TestCase):
    def test_dry_run_creates_sibling_html_without_model_call(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "离线报告.json"
            with patch("evaluate_real_cases.call_recommendation") as call, redirect_stdout(io.StringIO()) as log:
                code = main.main(["--dry-run", "--no-compare", "--customer-id", "C001",
                                  "--customer-id", "C047", "--output", str(output)])
            call.assert_not_called()
            html = output.with_suffix(".html").read_text(encoding="utf-8")
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(code, 0)
            self.assertIn("HTML 报告：", log.getvalue())
            self.assertIn("未调用模型", html)
            self.assertIn("未评估", html)
            self.assertIn("C001", html)
            self.assertIn("C047", html)
            self.assertIn("硬规则排除原因", html)
            self.assertIn("产品库-使用版.xlsx", html)
            self.assertIn("customers_validatable.json", html)
            self.assertIn("本次未启用历史基线比较", html)
            self.assertEqual(report["run"]["data_sources"]["available_customer_count"], 93)
            self.assertEqual(set(report), {"run", "summary", "results"})

    def test_results_errors_and_sources_are_escaped_and_metrics_keep_denominators(self):
        unsafe = '<script>alert("injected")</script>'
        rows = [
            {"customer_id": "C1", "validation_target": {"product_id": "P1"},
             "status": "completed", "target_retained": True, "top1_hit": False, "top3_hit": True,
             "recommendation": {"top3": [{"product_id": "P1", "rank": 2, "reasons": [unsafe],
                                          "missing_information": None, "confidence": []}]}},
            {"customer_id": "C2", "validation_target": {"product_id": "P1"},
             "status": "failed", "target_retained": True, "error": unsafe},
        ]
        report = {"run": {"mode": "real", "data_sources": {"products_origin": unsafe}},
                  "summary": summarize(rows), "results": rows}
        comparison = {"status": "not_comparable", "reasons": [unsafe], "metrics": {}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.html"
            save_html_report(report, path, result_path=path.with_suffix(".json"),
                             cases=[{"customer_id": "C1", "input": {"value": unsafe}}],
                             catalog={"source": unsafe}, comparison=comparison)
            html = path.read_text(encoding="utf-8")
        self.assertNotIn(unsafe, html)
        self.assertIn("&lt;script&gt;alert(&quot;injected&quot;)&lt;/script&gt;", html)
        self.assertIn("50.00%", html)
        self.assertIn("100.00%", html)
        self.assertIn("失败原因", html)
        self.assertIn("暂不判断提升", html)
        self.assertIn('href="report_comparison.json"', html)
        self.assertEqual(html.count("<script>"), 1)


if __name__ == "__main__":
    unittest.main()
