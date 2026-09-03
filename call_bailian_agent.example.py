"""通过 HTTP 调用阿里云百炼单智能体应用，验证贷款产品推荐结果。"""

from __future__ import annotations

import json
import os
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


# ========================= 调用配置 =========================

# 百炼公网业务域名。使用私网调用时，替换为对应的 PrivateLink 域名。
API_BASE_URL = os.getenv("DASHSCOPE_API_BASE_URL", "https://dashscope.aliyuncs.com")

# 请通过环境变量提供应用 ID 和 API Key，避免将真实凭据提交到 Git。
APP_ID = os.getenv("DASHSCOPE_APP_ID", "")
API_KEY = os.getenv("DASHSCOPE_API_KEY", "")

# 单次请求超时时间，单位为秒。
TIMEOUT_SECONDS = 120

# 待验证的贷前客户信息。缺失字段使用 None，不要填写最终放款产品等结果字段。
TEST_CUSTOMER = {
    "customer_id": "C001",
    "city": "苏州",
    "age": 35,
    "customer_type": "工薪族",
    "identity": "个人",
    "marital_status": "已婚",
    "education": "高中或以下",
    "company_type": "重点扶持类企业（高新技术企业、专精特新‘小巨人’）",
    "months_employed": "3年以上",
    "monthly_income": 6450.74,
    "salary_bank": "平安银行股份有限公司",
    "income_payment_type": "打卡工资（缴个税）",
    "social_security_months": "3年以上",
    "social_security_gap_last_year": False,
    "has_provident_fund": True,
    "provident_fund_months": None,
    "provident_fund_base": 4759.06,
    "has_house": True,
    "house_status": "按揭房",
    "house_count": 1,
    "house_area_sqm": 82.0,
    "has_car": False,
    "current_overdue": False,
    "overdue_in_last_5_years": False,
    "has_90_day_overdue": False,
    "loan_inquiries_1m": 0,
    "loan_inquiries_6m": None,
    "credit_card_accounts": 20,
    "active_credit_card_accounts": 9,
    "credit_card_overdue_accounts": 0,
    "other_loan_accounts": 5,
    "active_other_loan_accounts": 1,
    "mortgage_accounts": 2,
    "active_mortgage_accounts": 1,
    "requested_amount": None,
    "requested_amount_range": "11-30万",
    "product_type": "信用",
    "priority": ["利率"],
    "funding_urgency": "立刻",
    "max_acceptable_annual_rate": None,
    "preferred_repayment_method": None,
    "repayment_source": "薪资收入",
}

# ===========================================================


def validate_config() -> None:
    """在发起请求前检查必需配置。"""
    missing: list[str] = []
    if not API_BASE_URL.startswith(("http://", "https://")):
        missing.append("DASHSCOPE_API_BASE_URL")
    if not APP_ID:
        missing.append("DASHSCOPE_APP_ID")
    if not API_KEY:
        missing.append("DASHSCOPE_API_KEY")

    if missing:
        joined = "、".join(missing)
        raise ValueError(f"请先设置环境变量：{joined}")


def build_prompt(customer: dict) -> str:
    """将客户信息作为 JSON 传给已经配置好提示词和知识库的智能体。"""
    customer_json = json.dumps(customer, ensure_ascii=False, indent=2)
    return (
        "请根据已绑定的贷款产品知识库，为以下客户推荐 Top3 产品。"
        "缺失信息按待确认处理，不要预测最终获批额度或利率。\n\n"
        f"客户贷前信息：\n{customer_json}"
    )


def call_agent(prompt: str) -> dict:
    """按照百炼单智能体应用 HTTP 接口发起一次非流式请求。"""
    url = f"{API_BASE_URL.rstrip('/')}/api/v1/apps/{APP_ID}/completion"
    payload = {
        "input": {"prompt": prompt},
        "parameters": {},
        "debug": {},
    }
    request = Request(
        url=url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            response_text = response.read().decode("utf-8")
            return json.loads(response_text)
    except HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"百炼接口返回 HTTP {exc.code}：{error_body}"
        ) from exc
    except URLError as exc:
        raise RuntimeError(f"无法连接百炼接口：{exc.reason}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("百炼接口返回的内容不是有效 JSON") from exc


def main() -> int:
    try:
        validate_config()
        result = call_agent(build_prompt(TEST_CUSTOMER))
    except (ValueError, RuntimeError) as exc:
        print(f"调用失败：{exc}", file=sys.stderr)
        return 1

    output = result.get("output") or {}
    answer = output.get("text")

    print("\n===== 智能体推荐结果 =====")
    if answer:
        print(answer)
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))

    print("\n===== 调用信息 =====")
    print(f"request_id: {result.get('request_id', '')}")
    print(f"session_id: {output.get('session_id', '')}")
    print(f"finish_reason: {output.get('finish_reason', '')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
