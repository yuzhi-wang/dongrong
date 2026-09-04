from __future__ import annotations

import json
import unittest
from pathlib import Path

from hard_filter import filter_products


ROOT_DIR = Path(__file__).resolve().parents[1]


class HardFilterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = json.loads(
            (ROOT_DIR / "data" / "loan_products.json").read_text(encoding="utf-8")
        )
        cls.customer = json.loads(
            (ROOT_DIR / "data" / "test_customer.json").read_text(encoding="utf-8")
        )
        cls.result = filter_products(cls.catalog, cls.customer)

    def test_counts_are_consistent(self) -> None:
        self.assertEqual(self.result["total_product_count"], 12)
        self.assertEqual(
            self.result["candidate_count"] + self.result["excluded_count"],
            12,
        )

    def test_jianyidai_is_excluded_by_credit_card_limit(self) -> None:
        excluded = {
            item["product_name"]: item for item in self.result["excluded_products"]
        }
        self.assertIn("建易贷", excluded)
        failures = excluded["建易贷"]["failed_rules"]
        self.assertTrue(
            any(rule["field"] == "信用卡张数上限" for rule in failures),
            failures,
        )

    def test_shandiandai_remains_a_candidate(self) -> None:
        candidate_names = {item["产品名称"] for item in self.result["candidates"]}
        self.assertIn("闪电贷", candidate_names)

    def test_missing_values_do_not_trigger_exclusion(self) -> None:
        flash = next(
            item
            for item in self.result["candidate_audit"]
            if item["product_name"] == "闪电贷"
        )
        self.assertFalse(flash["failed_rules"])
        self.assertTrue(flash["unknown_rules"])


if __name__ == "__main__":
    unittest.main()
