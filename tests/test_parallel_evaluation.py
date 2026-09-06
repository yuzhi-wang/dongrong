from __future__ import annotations

import copy
import io
import json
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from unittest.mock import patch

import main
import evaluate_real_cases as evaluation
from report_comparison import DEFAULT_BASELINE_PATH, DEFAULT_PROMPT_BASELINE_PATH, compare_prompt_reports, compare_reports


class ParallelEvaluationTest(unittest.TestCase):
    def test_entry_defaults_and_explicit_selection(self):
        with patch("main.evaluate", return_value=0) as evaluate:
            self.assertEqual(main.main([]), 0)
        self.assertEqual(evaluate.call_args.kwargs, {
            "default_all": True, "default_workers": 16,
            "default_baseline": DEFAULT_BASELINE_PATH,
        })
        for argv, all_cases in (([], True), (["--sample-size", "3"], False),
                                (["--customer-id", "C001"], False)):
            args = evaluation.parse_args(argv, default_all=True, default_workers=16)
            self.assertEqual(args.all, all_cases)
            self.assertEqual(args.workers, 16)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            evaluation.parse_args(["--history-rerank"])

    def test_invalid_limits_rejected(self):
        for argv in (["--workers", "0"], ["--request-interval", "nan"],
                     ["--request-interval", "-1"], ["--sample-size", "0"]):
            with self.subTest(argv=argv), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    evaluation.parse_args(argv)

    def test_full_run_mocked_transport_preserves_order_and_single_writer(self):
        cases = evaluation.load_cases()
        state = {"active": 0, "peak": 0, "calls": 0}
        lock = threading.Lock()
        first_wave = threading.Barrier(4, timeout=5)
        finished = []
        caller = threading.get_ident()
        original_save = evaluation.save_report

        def fake_call(**kwargs):
            self.assertEqual(kwargs["api_key"], "mock-only")
            kwargs["before_request"]()
            prompt = kwargs["input_text"]
            self.assertNotIn("validation_target", prompt)
            customer = json.loads(prompt.split("## 客户贷前信息")[1].split("## 剩余候选产品")[0])
            products = json.loads(prompt.split("## 剩余候选产品")[1].split("## 本地过滤摘要")[0])["products"]
            with lock:
                state["calls"] += 1
                number = state["calls"]
                state["active"] += 1
                state["peak"] = max(state["peak"], state["active"])
            try:
                if number <= 4:
                    first_wave.wait()
                time.sleep(0.03 if customer["customer_id"] == cases[0]["customer_id"] else 0.001)
                rankings = [{"rank": i, "product_id": p["产品唯一ID"], "match_score": 100-i}
                            for i, p in enumerate(products, 1)]
                body = {"customer_id": customer["customer_id"], "top3": rankings[:3]}
                return {"id": "mock", "model": "qwen3.8-max", "status": "completed",
                        "output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(body)}]}]}
            finally:
                with lock:
                    state["active"] -= 1
                    finished.append(customer["customer_id"])

        def checked_save(report, path):
            self.assertEqual(threading.get_ident(), caller)
            original_save(report, path)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "mock.json"
            with patch("evaluate_real_cases.load_api_key", return_value="mock-only"):
                with patch("evaluate_real_cases.call_recommendation", side_effect=fake_call), \
                     patch("evaluate_real_cases.save_report", side_effect=checked_save), \
                     patch("dashscope_client.urlopen", side_effect=AssertionError("No real network allowed")), \
                     redirect_stdout(io.StringIO()):
                    status = main.main(["--workers", "4", "--request-interval", "0", "--output", str(output)])
            self.assertEqual(len(list(output.with_name("mock_raw").glob("*.json"))), 91)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertNotIn("_raw_response", report["results"][0])
            comparison = json.loads(output.with_name("mock_comparison.json").read_text(encoding="utf-8"))
        self.assertEqual(status, 0)
        self.assertEqual(state["calls"], 91)
        self.assertEqual(state["peak"], 4)
        self.assertNotEqual(finished[0], cases[0]["customer_id"])
        self.assertEqual(set(report), {"run", "summary", "results"})
        self.assertEqual([r["customer_id"] for r in report["results"]], [c["customer_id"] for c in cases])
        self.assertEqual(report["summary"]["model"]["completed_count"], 91)
        self.assertEqual(report["summary"]["model"]["failed_count"], 0)
        self.assertEqual(report["summary"]["hard_filter"]["false_negative_count"], 2)
        self.assertEqual(comparison["status"], "comparable")
        self.assertEqual(set(comparison["metrics"]), {"top1", "recall_at_3"})
        self.assertNotIn("history_rerank", report["run"])
        self.assertEqual(comparison["prompt_comparison"]["status"], "comparable")
        self.assertEqual(comparison["prompt_comparison"]["metrics"]["recall_at_3"]["baseline"]["hits"], 41)
        for row in report["results"]:
            if row["status"] == "completed":
                self.assertNotIn("pre_rerank_top3_hit", row)
                self.assertNotIn("pre_rerank_top1_hit", row)
                for ranking in row["recommendation"]["top3"]:
                    self.assertEqual(ranking["match_score"], 100-ranking["rank"])
                    self.assertNotIn("model_rank", ranking)
                    self.assertNotIn("history_count", ranking)
                for key in ("ranking", "rankings", "ranked_product_count", "ranking_scope", "ranking_method", "history_rerank"):
                    self.assertNotIn(key, row["recommendation"])
                self.assertEqual(len(row["recommendation"]["top3"]), 3)

    def test_bad_baseline_rejected_before_call(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text('{"results": []}', encoding="utf-8")
            with patch("evaluate_real_cases.load_api_key", return_value="mock-only"), \
                 patch("evaluate_real_cases.call_recommendation") as call, redirect_stderr(io.StringIO()):
                self.assertEqual(main.main(["--baseline", str(path)]), 1)
            call.assert_not_called()

    def test_invalid_output_is_saved_before_reporting_failure(self):
        raw = '{"customer_id":"C001","top3":[]}'
        response = {"id": "raw-test", "model": "mock", "status": "completed",
                    "output": [{"type": "message", "content": [{"type": "output_text", "text": raw}]}]}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "bad_output.json"
            with patch("evaluate_real_cases.load_api_key", return_value="mock-only"), \
                 patch("evaluate_real_cases.call_recommendation", return_value=response), redirect_stdout(io.StringIO()):
                self.assertEqual(main.main(["--customer-id", "C001", "--no-compare", "--output", str(output)]), 0)
            report = json.loads(output.read_text(encoding="utf-8"))
            row = report["results"][0]
            self.assertEqual(row["status"], "failed")
            self.assertIn("实际为 0", row["error"])
            saved = json.loads(Path(row["response"]["raw_path"]).read_text(encoding="utf-8"))
            self.assertEqual(saved["response"]["output"][0]["content"][0]["text"], raw)
            self.assertEqual(saved["response"]["id"], "raw-test")
            self.assertNotIn("_raw_response", row)

    def test_customer_failures_are_saved_and_remaining_cases_continue(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "failed.json"
            with patch("evaluate_real_cases.load_api_key", return_value="mock-only"), \
                 patch("evaluate_real_cases.call_recommendation", side_effect=RuntimeError("mock failure")) as call, \
                 redirect_stdout(io.StringIO()):
                status = main.main(["--customer-id", "C001", "--customer-id", "C002", "--customer-id", "C047",
                                    "--request-interval", "0", "--output", str(output)])
            report = json.loads(output.read_text(encoding="utf-8"))
            comparison = json.loads(output.with_name("failed_comparison.json").read_text(encoding="utf-8"))
        self.assertEqual(status, 0)
        self.assertEqual(call.call_count, 2)
        self.assertEqual(report["summary"]["case_count"], 3)
        self.assertEqual(report["summary"]["model"]["failed_count"], 2)
        self.assertEqual(comparison["status"], "not_comparable")


class ComparisonTest(unittest.TestCase):
    def setUp(self):
        self.baseline = json.loads(DEFAULT_BASELINE_PATH.read_text(encoding="utf-8"))

    def test_full_baseline_and_percentage_point_change(self):
        current = copy.deepcopy(self.baseline)
        row = next(r for r in current["results"] if r["status"] == "completed" and not r["pre_rerank_top1_hit"])
        row["pre_rerank_top1_hit"] = True
        result = compare_reports(self.baseline, current)
        self.assertEqual(result["status"], "comparable")
        for name, hits in (("top1", 19), ("recall_at_3", 45)):
            self.assertEqual(result["metrics"][name]["baseline"]["hits"], hits)
            self.assertEqual(result["metrics"][name]["baseline"]["denominator"], 93)
        self.assertEqual(result["metrics"]["top1"]["direction"], "improved")
        self.assertEqual(result["metrics"]["top1"]["delta_percentage_points"], round(100/93, 4))

    def test_legacy_and_top3_only_reports_compare_original_hits(self):
        current = copy.deepcopy(self.baseline)
        current["run"] = {"model": "test", "mode": "real", "finished_at": "2026-09-06"}
        for row in current["results"]:
            if row["status"] == "completed":
                row["top1_hit"] = row.pop("pre_rerank_top1_hit")
                row["top3_hit"] = row.pop("pre_rerank_top3_hit")
                row["recommendation"] = {"top3": []}
        comparison = compare_reports(self.baseline, current)
        self.assertEqual(comparison["status"], "comparable")
        self.assertEqual(set(comparison["metrics"]), {"top1", "recall_at_3"})
        self.assertEqual(comparison["metrics"]["recall_at_3"]["delta_percentage_points"], 0)

    def test_prompt_comparison_uses_fixed_old_success_cohort(self):
        old = json.loads(DEFAULT_PROMPT_BASELINE_PATH.read_text(encoding="utf-8"))
        current = copy.deepcopy(self.baseline)
        current["run"] = copy.deepcopy(old["run"])
        comparison = compare_prompt_reports(old, current)
        self.assertEqual(comparison["status"], "comparable")
        metric = comparison["metrics"]["recall_at_3"]
        self.assertEqual(metric["baseline"]["hits"], 41)
        self.assertEqual(metric["current"]["hits"], 42)
        self.assertEqual(metric["current"]["denominator"], 86)
        self.assertEqual(len(comparison["excluded_baseline_customer_ids"]), 7)
        current["run"]["model"] = "different-model"
        self.assertEqual(compare_prompt_reports(old, current)["status"], "not_comparable")
        current["run"]["model"] = old["run"]["model"]
        current["results"][0]["status"] = "failed"
        self.assertEqual(compare_prompt_reports(old, current)["status"], "not_comparable")

    def test_incomparable_reports_never_claim_improvement(self):
        for mutation in ("cohort", "failed", "dry_run", "unfinished"):
            current = copy.deepcopy(self.baseline)
            row = next(r for r in current["results"] if r["status"] == "completed")
            if mutation == "cohort":
                current["results"].pop()
            elif mutation == "failed":
                row["status"] = "failed"
            elif mutation == "dry_run":
                current["run"]["mode"] = "dry_run"
            elif mutation == "unfinished":
                del current["run"]["finished_at"]
            with self.subTest(mutation=mutation):
                result = compare_reports(self.baseline, current)
                self.assertEqual(result["status"], "not_comparable")
                self.assertEqual(result["metrics"], {})
