# dongrong

东融科技推荐算法工程师面试项目，两天 MVP 方案：先用本地确定性规则排除明确不符合的贷款产品，再单次调用通义千问对剩余候选排序，最后用本地历史产品频率做轻量重排并返回 Top3。

## 文件说明

- `docs/产品库-使用版.xlsx`：可推荐的贷款产品库。
- `docs/脱敏_面试专用.xlsx`：脱敏历史放款记录。
- `outputs/01a0610a-1444-7f01-bb33-d1c13c6ce446/脱敏_面试专用_仅保留产品库产品.xlsx`：基础清洗结果，仅保留当前产品库中存在的产品，共保留 192 条、剔除 34 条。
- `data/loan_products.json`：由产品库完整转换得到的 12 款产品数据。
- `data/product_history_priors.json`：由历史可验证记录汇总的产品频率及重排权重。
- `data/test_customer.json`：不包含真实放款结果的测试客户输入。
- `hard_filter.py`：本地基础硬规则过滤，明确违反规则的产品不会传给模型。
- `history_rerank.py`：融合模型名次和历史产品频率，只重排候选且不恢复硬规则排除项。
- `dashscope_client.py`：DashScope Responses API 请求与响应解析。
- `main.py`：加载数据、执行过滤、调用 Qwen 并校验输出的入口。
- `prompts/`：system 与 user Markdown 提示词模板。

## 调用示例

在 PowerShell 中设置本次终端会话所需的环境变量，然后运行脚本：

```powershell
$env:DASHSCOPE_API_KEY = "你的API Key"
python -X utf8 main.py
```

当前默认模型为 `qwen3.7-max`，不需要智能体 APP ID 或知识库。请勿把真实 API Key 或本地配置文件提交到 Git。

## 真实用例测试

先只检查分层抽样和硬规则，不调用模型：

```powershell
python -X utf8 evaluate_real_cases.py --dry-run
```

默认从“可正常验证”组按真实产品分层抽取 12 条，每款产品 1 条：

```powershell
python -X utf8 evaluate_real_cases.py
```

也可以只测一个客户，或在提示词稳定后测试全部 93 条：

```powershell
python -X utf8 evaluate_real_cases.py --customer-id C001
python -X utf8 evaluate_real_cases.py --all
```

报告写入 `outputs/validation_runs/`，包含硬规则误删、重排前后 Top1、Recall@3、分产品结果、响应结构错误及 Token 用量。真实放款标签不会进入模型输入；当前93条回测会从该客户真实产品的历史计数中扣除1条，再执行留一重排。
