from __future__ import annotations

import unittest

from evaluate_real_cases import stratified_sample, summarize


def _case(customer_id: str, product_id: str) -> dict:
    return {
        "customer_id": customer_id,
        "input": {},
        "validation_target": {
            "product_id": product_id,
            "product_name": product_id,
        },
    }


class StratifiedSampleTest(unittest.TestCase):
    def test_first_round_covers_each_product(self) -> None:
        cases = [
            _case("A1", "A"),
            _case("A2", "A"),
            _case("A3", "A"),
            _case("B1", "B"),
            _case("B2", "B"),
            _case("C1", "C"),
        ]
        selected = stratified_sample(cases, sample_size=3, seed=7)
        self.assertEqual(
            {case["validation_target"]["product_id"] for case in selected},
            {"A", "B", "C"},
        )

    def test_sample_is_repeatable(self) -> None:
        cases = [_case(f"A{i}", "A") for i in range(5)]
        first = stratified_sample(cases, sample_size=2, seed=11)
        second = stratified_sample(cases, sample_size=2, seed=11)
        self.assertEqual(
            [case["customer_id"] for case in first],
            [case["customer_id"] for case in second],
        )


class SummaryTest(unittest.TestCase):
    def test_dry_run_metrics_are_not_reported_as_zero(self) -> None:
        results = [
            {
                "validation_target": {"product_id": "A", "product_name": "甲"},
                "target_retained": True,
                "status": "dry_run",
            }
        ]
        summary = summarize(results)
        self.assertEqual(summary["model"]["not_exercised_count"], 1)
        self.assertIsNone(summary["end_to_end"]["recall_at_3"])
        self.assertIsNone(summary["by_product"][0]["end_to_end_recall_at_3"])

    def test_distinguishes_model_and_end_to_end_metrics(self) -> None:
        results = [
            {
                "validation_target": {"product_id": "A", "product_name": "甲"},
                "target_retained": True,
                "status": "completed",
                "top1_hit": False,
                "top3_hit": True,
                "response": {
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 5,
                        "total_tokens": 15,
                    }
                },
            },
            {
                "validation_target": {"product_id": "B", "product_name": "乙"},
                "target_retained": False,
                "status": "hard_filter_false_negative",
            },
        ]
        summary = summarize(results)
        self.assertEqual(summary["hard_filter"]["false_negative_count"], 1)
        self.assertEqual(summary["model"]["pre_rerank_recall_at_3_on_completed"], 1.0)
        self.assertEqual(summary["model"]["recall_at_3_on_completed"], 1.0)
        self.assertEqual(summary["end_to_end"]["recall_at_3"], 0.5)
        self.assertEqual(summary["usage"]["total_tokens"], 15)

    def test_reports_pre_rerank_and_final_metrics_separately(self) -> None:
        results = [
            {
                "validation_target": {"product_id": "A", "product_name": "甲"},
                "target_retained": True,
                "status": "completed",
                "pre_rerank_top1_hit": False,
                "pre_rerank_top3_hit": False,
                "top1_hit": True,
                "top3_hit": True,
                "response": {"usage": {}},
            }
        ]
        summary = summarize(results)
        self.assertEqual(summary["model"]["pre_rerank_top3_hits"], 0)
        self.assertEqual(summary["model"]["top3_hits"], 1)
        self.assertEqual(summary["end_to_end"]["recall_at_3"], 1.0)


if __name__ == "__main__":
    unittest.main()
