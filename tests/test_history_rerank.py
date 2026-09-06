from __future__ import annotations

import unittest

from history_rerank import rerank_recommendation


def _prior() -> dict:
    return {
        "sample_count": 13,
        "default_history_weight": 0.6,
        "products": [
            {"product_id": "A", "count": 1},
            {"product_id": "B", "count": 1},
            {"product_id": "C", "count": 10},
            {"product_id": "X", "count": 1},
        ],
    }


def _recommendation() -> dict:
    return {
        "customer_id": "T001",
        "ranked_product_count": 4,
        "top3": [
            {"rank": 1, "product_id": "A", "match_score": 90},
            {"rank": 2, "product_id": "B", "match_score": 80},
            {"rank": 3, "product_id": "D", "match_score": 70},
        ],
        "rankings": [
            {"rank": 1, "product_id": "A", "match_score": 90},
            {"rank": 2, "product_id": "B", "match_score": 80},
            {"rank": 3, "product_id": "D", "match_score": 70},
            {"rank": 4, "product_id": "C", "match_score": 60},
        ],
    }


def _products() -> dict[str, dict]:
    return {
        product_id: {
            "产品唯一ID": product_id,
            "产品名称": f"产品{product_id}",
            "银行名称": f"银行{product_id}",
            "最高额度（元）": 100000,
            "最低年利率": 0.03,
            "最高年利率": 0.06,
            "最长期限（月）": 36,
        }
        for product_id in ("A", "B", "C", "D")
    }


class HistoryRerankTest(unittest.TestCase):
    def test_high_frequency_product_can_be_promoted(self) -> None:
        result = rerank_recommendation(
            _recommendation(),
            _prior(),
            _products(),
        )
        self.assertEqual(result["top3"][0]["product_id"], "C")
        self.assertEqual(result["top3"][0]["model_rank"], 4)
        self.assertEqual(result["top3"][0]["product_name"], "产品C")

    def test_does_not_restore_product_missing_from_candidates(self) -> None:
        result = rerank_recommendation(
            _recommendation(),
            _prior(),
            _products(),
        )
        self.assertNotIn("X", [item["product_id"] for item in result["rankings"]])

    def test_leave_one_out_subtracts_current_target(self) -> None:
        result = rerank_recommendation(
            _recommendation(),
            _prior(),
            _products(),
            exclude_product_id="C",
        )
        c_row = next(
            item for item in result["rankings"] if item["product_id"] == "C"
        )
        self.assertEqual(c_row["history_count"], 9)
        self.assertEqual(result["history_rerank"]["history_sample_count"], 12)
        self.assertTrue(result["history_rerank"]["leave_one_out"])

    def test_final_scores_and_ranks_are_consistent(self) -> None:
        result = rerank_recommendation(
            _recommendation(),
            _prior(),
            _products(),
        )
        rankings = result["rankings"]
        self.assertEqual(
            [item["rank"] for item in rankings],
            list(range(1, len(rankings) + 1)),
        )
        self.assertEqual(
            [item["match_score"] for item in rankings],
            sorted(
                [item["match_score"] for item in rankings],
                reverse=True,
            ),
        )
        self.assertEqual(
            [item["product_id"] for item in result["top3"]],
            [item["product_id"] for item in rankings[:3]],
        )


if __name__ == "__main__":
    unittest.main()
