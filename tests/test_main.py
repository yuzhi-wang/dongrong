from __future__ import annotations

import json
import unittest

from hard_filter import filter_products
from main import (
    CUSTOMER_PATH,
    PRODUCTS_PATH,
    SYSTEM_PROMPT_PATH,
    build_user_prompt,
    load_json,
    parse_recommendation,
)


class PromptConstructionTest(unittest.TestCase):
    def test_prompt_omits_local_hard_rule_audit(self) -> None:
        catalog = load_json(PRODUCTS_PATH)
        customer = load_json(CUSTOMER_PATH)
        prompt = build_user_prompt(filter_products(catalog, customer), customer)

        self.assertNotIn("candidate_audit", prompt)
        self.assertNotIn("unknown_rules", prompt)
        self.assertNotIn("是否禁止当前逾期", prompt)
        self.assertNotIn("信用卡张数上限", prompt)
        self.assertNotIn("是否优先推荐", prompt)
        self.assertNotIn("{{", prompt)
        self.assertLess(prompt.index("## 客户贷前信息"), prompt.index("## 剩余候选产品"))


class SystemPromptSemanticsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")

    def test_does_not_reward_unused_credit_limit(self) -> None:
        self.assertIn("不得因为某款产品额度上限更高", self.prompt)
        self.assertNotIn("再比较额度余量", self.prompt)

    def test_treats_empty_product_fields_as_unknown(self) -> None:
        self.assertIn("产品字段为 `null` 或空值时，表示产品信息未知", self.prompt)
        self.assertIn("只有字段明确写出“无要求”或“未限制”", self.prompt)

    def test_keeps_audience_paths_independent(self) -> None:
        self.assertIn("不得把客群1的条件与客群2", self.prompt)
        self.assertIn("必须按相同编号对齐后比较", self.prompt)

    def test_forbids_substituting_employment_for_contribution_months(self) -> None:
        self.assertIn("在职时长不能证明公积金或社保缴存时长", self.prompt)


class RecommendationParsingTest(unittest.TestCase):
    def test_accepts_compact_full_ranking_and_top_products(self) -> None:
        payload = {
            "customer_id": "C001",
            "ranked_product_count": 2,
            "top3": [
                {"rank": 1, "product_id": "A", "match_score": 90},
                {"rank": 2, "product_id": "B", "match_score": 80},
            ],
            "rankings": [
                {"rank": 1, "product_id": "A", "match_score": 90},
                {"rank": 2, "product_id": "B", "match_score": 80},
            ],
        }
        result = parse_recommendation(json.dumps(payload), {"A", "B"})
        self.assertEqual(result["top3"][0]["product_id"], "A")

    def test_rejects_top3_that_disagrees_with_ranking(self) -> None:
        payload = {
            "ranked_product_count": 2,
            "top3": [
                {"rank": 1, "product_id": "B", "match_score": 80},
                {"rank": 2, "product_id": "A", "match_score": 90},
            ],
            "rankings": [
                {"rank": 1, "product_id": "A", "match_score": 90},
                {"rank": 2, "product_id": "B", "match_score": 80},
            ],
        }
        with self.assertRaisesRegex(RuntimeError, "Top3 必须与 rankings"):
            parse_recommendation(json.dumps(payload), {"A", "B"})


if __name__ == "__main__":
    unittest.main()
