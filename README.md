# dongrong

东融科技推荐算法工程师面试项目，两天 MVP 方案：使用阿里云百炼低代码智能体和产品知识库，为客户推荐 Top3 贷款产品，并使用历史放款记录验证结果。

## 文件说明

- `docs/产品库-使用版.xlsx`：可推荐的贷款产品库。
- `docs/脱敏_面试专用.xlsx`：脱敏历史放款记录。
- `outputs/01a0610a-1444-7f01-bb33-d1c13c6ce446/脱敏_面试专用_仅保留产品库产品.xlsx`：基础清洗结果，仅保留当前产品库中存在的产品，共保留 192 条、剔除 34 条。
- `call_bailian_agent.example.py`：百炼单智能体 HTTP 调用示例，不包含真实凭据。

## 调用示例

在 PowerShell 中设置本次终端会话所需的环境变量，然后运行脚本：

```powershell
$env:DASHSCOPE_APP_ID = "你的应用ID"
$env:DASHSCOPE_API_KEY = "你的API Key"
python -X utf8 call_bailian_agent.example.py
```

请勿把真实的应用 ID、API Key 或本地配置文件提交到 Git。
