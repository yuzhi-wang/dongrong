"""贷款产品的本地基础硬规则过滤。

只处理能够从产品 JSON 和客户 JSON 中确定比较的结构化规则。缺失值或
自然语言复合客群条件不会被当作失败，而会记录为待确认，避免误删候选产品。
"""

from __future__ import annotations

import re
from typing import Any


UNLIMITED_TEXTS = {"", "未限制", "不限", "无", "无要求", "不限制"}


def _text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _is_unlimited(value: Any) -> bool:
    text = _text(value)
    return value is None or text in UNLIMITED_TEXTS or text.startswith("无要求")


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = _text(value).replace(",", "")
    match = re.fullmatch(r"(?:≤|<=)?\s*(-?\d+(?:\.\d+)?)\s*", text)
    return float(match.group(1)) if match else None


def _rule(
    field: str,
    status: str,
    reason: str,
    *,
    customer_field: str | None = None,
    customer_value: Any = None,
    product_value: Any = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {"field": field, "status": status, "reason": reason}
    if customer_field:
        result["customer_field"] = customer_field
        result["customer_value"] = customer_value
    if product_value is not None:
        result["product_value"] = product_value
    return result


def _check_region(product: dict, customer: dict) -> dict | None:
    restriction = product.get("地区限制")
    if _is_unlimited(restriction) or "全国" in _text(restriction):
        return None
    city = customer.get("city")
    if city is None:
        return _rule("地区限制", "unknown", "客户城市缺失", customer_field="city")
    if _text(city) not in _text(restriction):
        return _rule(
            "地区限制",
            "fail",
            f"客户地区 {_text(city)} 不在产品地区 {_text(restriction)} 内",
            customer_field="city",
            customer_value=city,
            product_value=restriction,
        )
    return _rule("地区限制", "pass", "客户地区符合产品限制")


def _check_product_type(product: dict, customer: dict) -> dict | None:
    required = product.get("产品类型")
    if _is_unlimited(required):
        return None
    actual = customer.get("product_type")
    if actual is None:
        return _rule("产品类型", "unknown", "客户需求的产品类型缺失")
    expected_text = _text(required).replace("贷款", "").replace("贷", "")
    actual_text = _text(actual).replace("贷款", "").replace("贷", "")
    if expected_text not in actual_text and actual_text not in expected_text:
        return _rule(
            "产品类型",
            "fail",
            f"客户需求类型 {_text(actual)} 与产品类型 {_text(required)} 不符",
            customer_field="product_type",
            customer_value=actual,
            product_value=required,
        )
    return _rule("产品类型", "pass", "产品类型符合客户需求")


def _check_age(product: dict, customer: dict) -> list[dict]:
    minimum = _number(product.get("最低年龄（周岁）"))
    maximum = _number(product.get("最高年龄（周岁）"))
    if minimum is None and maximum is None:
        return []
    age = _number(customer.get("age"))
    if age is None:
        return [_rule("年龄", "unknown", "客户年龄缺失", customer_field="age")]
    if minimum is not None and age < minimum:
        return [_rule(
            "最低年龄（周岁）",
            "fail",
            f"客户年龄 {age:g} 低于最低年龄 {minimum:g}",
            customer_field="age",
            customer_value=age,
            product_value=minimum,
        )]
    if maximum is not None and age > maximum:
        return [_rule(
            "最高年龄（周岁）",
            "fail",
            f"客户年龄 {age:g} 高于最高年龄 {maximum:g}",
            customer_field="age",
            customer_value=age,
            product_value=maximum,
        )]
    return [_rule("年龄", "pass", "客户年龄在产品范围内")]


def _prohibits(value: Any) -> bool:
    if value is True or value == 1:
        return True
    text = _text(value)
    return text in {"是", "禁止", "不允许", "禁入"} or text.startswith("禁止")


def _check_prohibition(
    product: dict,
    customer: dict,
    product_field: str,
    customer_field: str,
    label: str,
) -> dict | None:
    if not _prohibits(product.get(product_field)):
        return None
    actual = customer.get(customer_field)
    if actual is None:
        return _rule(product_field, "unknown", f"客户{label}信息缺失", customer_field=customer_field)
    if bool(actual):
        return _rule(
            product_field,
            "fail",
            f"产品禁止{label}，客户已知存在该情况",
            customer_field=customer_field,
            customer_value=actual,
            product_value=product.get(product_field),
        )
    return _rule(product_field, "pass", f"客户不存在{label}")


def _check_numeric_max(
    product: dict,
    customer: dict,
    product_field: str,
    customer_field: str,
) -> dict | None:
    limit_value = product.get(product_field)
    if _is_unlimited(limit_value):
        return None
    limit = _number(limit_value)
    if limit is None:
        return _rule(
            product_field,
            "unknown",
            "产品规则不是可直接比较的单一数值",
            customer_field=customer_field,
            customer_value=customer.get(customer_field),
            product_value=limit_value,
        )
    actual = _number(customer.get(customer_field))
    if actual is None:
        return _rule(
            product_field,
            "unknown",
            f"客户字段 {customer_field} 缺失",
            customer_field=customer_field,
            customer_value=customer.get(customer_field),
            product_value=limit_value,
        )
    if actual > limit:
        return _rule(
            product_field,
            "fail",
            f"{customer_field}={actual:g} 超过产品上限 {limit:g}",
            customer_field=customer_field,
            customer_value=actual,
            product_value=limit,
        )
    return _rule(
        product_field,
        "pass",
        f"{customer_field}={actual:g} 未超过产品上限 {limit:g}",
    )


def _check_five_year_overdue(product: dict, customer: dict) -> dict | None:
    field = "近5年逾期累计逾期次数上限"
    limit_value = product.get(field)
    if _is_unlimited(limit_value):
        return None
    limit = _number(limit_value)
    if limit is None:
        return _rule(field, "unknown", "产品近5年逾期规则不是单一数值", product_value=limit_value)

    count = _number(customer.get("overdue_count_5y"))
    if count is None and customer.get("overdue_in_last_5_years") is False:
        count = 0.0
    if count is None and customer.get("overdue_in_last_5_years") is True and limit == 0:
        count = 1.0
    if count is None:
        return _rule(
            field,
            "unknown",
            "客户近5年逾期次数不明确",
            customer_field="overdue_count_5y",
            product_value=limit,
        )
    if count > limit:
        return _rule(
            field,
            "fail",
            f"客户近5年逾期次数 {count:g} 超过上限 {limit:g}",
            customer_field="overdue_count_5y",
            customer_value=count,
            product_value=limit,
        )
    return _rule(field, "pass", "客户近5年逾期次数未超过产品上限")


def _check_identity(product: dict, customer: dict) -> dict | None:
    allowed = product.get("允许的身份")
    if _is_unlimited(allowed):
        return None
    identity = customer.get("customer_type")
    if identity is None:
        return _rule(
            "允许的身份",
            "unknown",
            "客户客群身份缺失",
            customer_field="customer_type",
        )
    if _text(identity) not in _text(allowed):
        return _rule(
            "允许的身份",
            "fail",
            f"客户客群身份 {_text(identity)} 不在允许范围内",
            customer_field="customer_type",
            customer_value=identity,
            product_value=allowed,
        )
    return _rule("允许的身份", "pass", "客户客群身份在允许范围内")


def _requested_amount_range(customer: dict) -> tuple[float, float] | None:
    exact = _number(customer.get("requested_amount"))
    if exact is not None:
        return exact, exact
    text = _text(customer.get("requested_amount_range"))
    numbers = [float(value) for value in re.findall(r"\d+(?:\.\d+)?", text)]
    if not numbers:
        return None
    multiplier = 10000 if "万" in text else 1
    if len(numbers) == 1:
        return numbers[0] * multiplier, numbers[0] * multiplier
    return min(numbers[:2]) * multiplier, max(numbers[:2]) * multiplier


def _check_amount(product: dict, customer: dict) -> dict | None:
    maximum = _number(product.get("最高额度（元）"))
    if maximum is None:
        return None
    requested = _requested_amount_range(customer)
    if requested is None:
        return _rule("最高额度（元）", "unknown", "客户申请金额缺失", product_value=maximum)
    requested_min, requested_max = requested
    if requested_min > maximum:
        return _rule(
            "最高额度（元）",
            "fail",
            f"客户最低需求 {requested_min:g} 元已超过产品最高额度 {maximum:g} 元",
            customer_field="requested_amount_range",
            customer_value=customer.get("requested_amount_range"),
            product_value=maximum,
        )
    if requested_max > maximum:
        return _rule(
            "最高额度（元）",
            "unknown",
            f"产品仅覆盖客户申请区间的一部分，最高额度 {maximum:g} 元",
            customer_field="requested_amount_range",
            customer_value=customer.get("requested_amount_range"),
            product_value=maximum,
        )
    return _rule("最高额度（元）", "pass", "产品最高额度覆盖客户申请金额")


NUMERIC_MAX_RULES = (
    ("近2年逾期累计逾期次数上限", "overdue_count_2y"),
    ("近1年逾期累计逾期次数上限", "overdue_count_1y"),
    ("近半年逾期累计逾期次数上限", "overdue_count_6m"),
    ("近12个月贷款审批查询次数上限", "loan_inquiries_12m"),
    ("近6个月贷款审批查询次数上限", "loan_inquiries_6m"),
    ("近4个月贷款审批查询次数上限", "loan_inquiries_4m"),
    ("近3个月贷款审批查询次数上限", "loan_inquiries_3m"),
    ("近2个月贷款审批查询次数上限", "loan_inquiries_2m"),
    ("近1个月贷款审批查询次数上限", "loan_inquiries_1m"),
    ("负债总额上限（除房贷车贷）", "total_non_mortgage_car_debt"),
    ("在贷机构数上限", "lending_institution_count"),
    ("网贷笔数上限", "online_loan_count"),
    ("经营贷款笔数上限", "business_loan_count"),
    ("非银机构数上限", "non_bank_institution_count"),
    ("信用卡使用率上限", "credit_card_utilization"),
    ("信用卡张数上限", "credit_card_accounts"),
)


def evaluate_product(product: dict, customer: dict) -> dict[str, Any]:
    """对单一产品执行基础硬规则检查。"""
    checks: list[dict] = []
    for result in (
        _check_region(product, customer),
        _check_product_type(product, customer),
        *_check_age(product, customer),
        _check_prohibition(product, customer, "是否禁止当前逾期", "current_overdue", "当前逾期"),
        _check_prohibition(
            product,
            customer,
            "是否禁止90天及以上逾期",
            "has_90_day_overdue",
            "90天及以上逾期",
        ),
        _check_five_year_overdue(product, customer),
        _check_identity(product, customer),
        _check_amount(product, customer),
    ):
        if result:
            checks.append(result)

    for product_field, customer_field in NUMERIC_MAX_RULES:
        result = _check_numeric_max(product, customer, product_field, customer_field)
        if result:
            checks.append(result)

    failed_rules = [check for check in checks if check["status"] == "fail"]
    unknown_rules = [check for check in checks if check["status"] == "unknown"]
    passed_rules = [check for check in checks if check["status"] == "pass"]
    return {
        "product_id": product.get("产品唯一ID"),
        "bank_name": product.get("银行名称"),
        "product_name": product.get("产品名称"),
        "decision": "excluded" if failed_rules else "candidate",
        "failed_rules": failed_rules,
        "unknown_rules": unknown_rules,
        "passed_rule_count": len(passed_rules),
    }


def filter_products(catalog: dict, customer: dict) -> dict[str, Any]:
    """过滤完整产品目录，返回候选、排除项与可审计原因。"""
    products = catalog.get("products") or []
    declared_count = catalog.get("product_count")
    if declared_count != len(products):
        raise ValueError(f"产品数量不一致：声明 {declared_count}，实际 {len(products)}")

    product_ids = [product.get("产品唯一ID") for product in products]
    if any(not product_id for product_id in product_ids) or len(set(product_ids)) != len(product_ids):
        raise ValueError("产品唯一ID为空或重复")

    evaluations = [evaluate_product(product, customer) for product in products]
    excluded_ids = {
        evaluation["product_id"]
        for evaluation in evaluations
        if evaluation["decision"] == "excluded"
    }
    candidates = [
        product for product in products if product.get("产品唯一ID") not in excluded_ids
    ]
    excluded = [
        evaluation for evaluation in evaluations if evaluation["decision"] == "excluded"
    ]
    candidate_audit = [
        evaluation for evaluation in evaluations if evaluation["decision"] == "candidate"
    ]

    return {
        "total_product_count": len(products),
        "candidate_count": len(candidates),
        "excluded_count": len(excluded),
        "candidates": candidates,
        "excluded_products": excluded,
        "candidate_audit": candidate_audit,
    }
