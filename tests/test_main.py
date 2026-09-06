from __future__ import annotations

import json
import unittest

from hard_filter import filter_products
from recommendation import (
    CUSTOMER_PATH,
    PRODUCTS_PATH,
    SYSTEM_PROMPT_PATH,
    build_user_prompt,
    load_json,
    parse_recommendation,
    preserve_model_recommendation,
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

    def test_output_contract_uses_actual_candidate_count(self) -> None:
        catalog = load_json(PRODUCTS_PATH)
        customer = load_json(CUSTOMER_PATH)
        filtered = filter_products(catalog, customer)
        for count in (1, 2, 3, len(filtered["candidates"])):
            partial = {**filtered, "candidate_count": count, "candidates": filtered["candidates"][:count]}
            prompt = build_user_prompt(partial, customer)
            with self.subTest(count=count):
                self.assertIn(f"本次共有 {count} 个候选", prompt)
                self.assertIn("`top3` 数组必须恰好包含 3 个产品对象", prompt)
                self.assertIn("只有候选总数不足 3 个时，才返回全部候选", prompt)
                self.assertNotIn("{{", prompt)


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

    def test_output_example_passes_production_ranking_validation(self) -> None:
        example, _ = json.JSONDecoder().raw_decode(self.prompt[self.prompt.index("{\n"):])
        parsed = parse_recommendation(json.dumps(example), {"候选A", "候选B", "候选C"})
        self.assertEqual(len(parsed["top3"]), 3)
        for row in parsed["top3"]:
            self.assertIn(row["confidence"], {"high", "medium", "low"})


class RecommendationParsingTest(unittest.TestCase):
    def test_top3_only_is_valid_without_ranking_other_candidates(self) -> None:
        top3 = [{"rank": i, "product_id": key, "match_score": 95-i} for i, key in enumerate("ABC", 1)]
        result = parse_recommendation(json.dumps({"top3": top3}), set("ABCDEFGHIJKL"))
        self.assertEqual(result["top3"], top3)
        self.assertNotIn("rankings", result)

    def test_rejects_wrong_count_duplicate_and_unknown_top3(self) -> None:
        for ids, message in (("AB", "实际为 2"), ("AAB", "重复产品"), ("ABX", "候选集合之外")):
            top3 = [{"rank": i, "product_id": key, "match_score": 95-i} for i, key in enumerate(ids, 1)]
            with self.subTest(ids=ids), self.assertRaisesRegex(RuntimeError, message):
                parse_recommendation(json.dumps({"top3": top3}), set("ABCD"))

    def test_preserving_model_order_does_not_rescale_or_mutate(self) -> None:
        original = {"rankings": [{"rank": 1, "product_id": "A", "match_score": 98},
                                 {"rank": 2, "product_id": "B", "match_score": 41}],
                    "top3": [{"rank": 1, "product_id": "A", "match_score": 98, "reasons": ["evidence"]},
                             {"rank": 2, "product_id": "B", "match_score": 41}]}
        result = preserve_model_recommendation(original, {"A": {"产品名称": "甲"}, "B": {"产品名称": "乙"}})
        self.assertEqual([r["match_score"] for r in result["top3"]], [98, 41])
        self.assertEqual([r["product_id"] for r in result["top3"]], ["A", "B"])
        self.assertEqual(result["top3"][0]["reasons"], ["evidence"])
        self.assertEqual(result["top3"][0]["product_name"], "甲")
        self.assertNotIn("model_rank", original["rankings"][0])
        for key in ("ranking", "rankings", "ranked_product_count", "ranking_scope", "ranking_method", "history_rerank"):
            self.assertNotIn(key, result)
        self.assertNotIn("model_rank", result["top3"][0])
        self.assertNotIn("product_name", original["top3"][0])

    def test_malformed_json_structure_is_a_reportable_failure(self) -> None:
        for payload in ([], {"rankings": [None], "top3": []},
                        {"rankings": [{"product_id": []}], "top3": []}):
            with self.subTest(payload=payload), self.assertRaises(RuntimeError):
                parse_recommendation(json.dumps(payload), {"A"})

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

    def test_rejects_top3_scores_that_increase(self) -> None:
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
        with self.assertRaisesRegex(RuntimeError, "Top3 未按 match_score"):
            parse_recommendation(json.dumps(payload), {"A", "B"})


if __name__ == "__main__":
    unittest.main()
