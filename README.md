# dongrong

东融科技推荐算法工程师面试项目：本地硬规则筛选贷款产品，再由通义千问从剩余候选中直接选出 Top3，保留三个推荐及理由，以模型原始 Top3 评估提示词效果。

完整交付结论、Astra与Qwen对比及93位客户结果见[最终结果文档](docs/final_results.md)。

## 直接运行 main 会做什么

`main.py` 默认使用 **qwen3.8-max、16 路并行**，处理 `data/validation_groups/customers_validatable.json` 中的 **全部 93 位客户**。每位客户依次执行硬过滤、模型排序和命中统计；模型只输出三个推荐，不生成全部候选的排名；按模型原始 Top3 顺序和分数输出。当前 2 位客户的真实产品被硬规则排除，因此通常需要 **91 次模型调用**；HTTP 429 重试会增加请求次数。

运行时显示完成进度、客户 ID、状态、Top1 / Top3 是否命中和耗时；结果逐条保存。全部结束后，比较两轮模型原始排名：一项是与 Qwen3.7-Max 的93条全量基线比较，另一项是与上次 Qwen3.8-Max 的固定86位成功客户比较，用于观察同模型下的提示词变化。

这是一个运行结束即退出的 Python 命令行项目，无需启动 Web 服务、数据库或 Docker。

## 本机运行流程

推荐 Python 3.12，代码使用 Python 3.10+ 语法及标准库，无需 `pip install`。运行直接读取仓库 JSON 与提示词，原始 Excel 无需重新转换。

在 PowerShell 中执行：

```powershell
Set-Location "C:/Users/wyz-0/Documents/GitHub/dongrong"
$PythonExe = "$env:LOCALAPPDATA/Programs/Python/Python312/python.exe"
& $PythonExe --version
& $PythonExe -X utf8 main.py
```

本机已验证解释器为 Python 3.12.4；当前 `python` WindowsApps 别名及 `py` 启动器不可用，所以示例使用完整路径。其他机器请替换实际目录和解释器路径。在 IDE 中选择该解释器后，直接 Run `main.py`，效果与上述命令一致，不需要额外参数。

### 密钥与模型配置

根目录 `dashscope_api_key.txt` 用于本机短期演示配置：UTF-8 编码，只放一行 Key，不加引号或变量名。Git 仓库不包含真实 Key；首次克隆后请自行创建该文件，或配置 `DASHSCOPE_API_KEY`。有效 Key 已配置时可以直接运行。程序优先读取非空的 `DASHSCOPE_API_KEY` 环境变量，其次读取该文件；不会自动加载 `.env`。环境变量会覆盖文件，文件位置不受终端工作目录影响。演示结束后应在百炼控制台撤销 Key，删除文件本身不会使已交付的 Key 失效。

以下环境变量可选，模型和地址的默认值如下：

```powershell
$env:DASHSCOPE_MODEL = "qwen3.8-max"
$env:DASHSCOPE_API_BASE_URL = "https://dashscope.aliyuncs.com"
# 如需使用环境变量提供密钥：
$env:DASHSCOPE_API_KEY = "替换为你的百炼 API Key"
& $PythonExe -X utf8 main.py
```

若此前终端设置过旧模型，请修改或移除 `DASHSCOPE_MODEL`。服务地址只填写根地址，程序会追加 `/compatible-mode/v1/chat/completions`。在线调用需要该地域的有效 Key、模型权限和网络连接。

当前采用 Chat Completions 流式 JSON 模式，`response_format={"type":"json_object"}`、`enable_thinking=true`、`thinking_budget=4096`、`temperature=0.1`。思考内容不写入推荐结果，最终文本拼接后仍需通过候选成员、唯一性、Top3 数量、连续名次和分数顺序校验。300 秒超时作用于连接及读取等待，持续收到流数据时总耗时可以更长。

### 并发设置依据

根据阿里云官方[限流文档](https://help.aliyun.com/zh/model-studio/rate-limit)及[动态限流说明](https://help.aliyun.com/zh/model-studio/quota-management)，截至 2026-09-06，`qwen3.8-max` 使用动态 TPM 限额，**没有公布固定的同时请求数量上限**。北京地域的公开档位为：

| 月消费金额 | TPM（每分钟输入与输出 Token 合计） |
| --- | ---: |
| 不超过 10 万元 | 5,000,000 |
| 超过 10 万元、不超过 100 万元 | 10,000,000 |
| 超过 100 万元 | 20,000,000 |

额度由同一主账号、同一模型共享，业务空间也可能另有较低限额，以控制台实际配置为准。短时突发或请求速率增长过快也可能触发 429；TPM 数值不能直接视为可用并发数。

项目默认 **16 个工作线程**，所有初始请求和重试共用 **0.25 秒启动间隔**。本机多次全量真实回测耗时约10至10分26秒；缓存、输出长度、账号负载和限流都会影响速度。需要更快时可以调整为32路：

```powershell
& $PythonExe -X utf8 main.py --workers 32
# 若限流频繁，可降低并发并拉长启动间隔：
& $PythonExe -X utf8 main.py --workers 8 --request-interval 0.5
```

仅 HTTP 429 自动重试，最多 2 次，使用指数退避、随机抖动，并等待至少 `Retry-After` 指定的时间。重试始终使用相同模型。401 / 403、连接中断、超时、JSON 或排名校验失败不自动重试，也不切换备用模型。

## 输出和命中率比较

默认结果路径为 `outputs/validation_runs/validatable_real_<时间戳>.json`，比较报告为同目录的 `validatable_real_<时间戳>_comparison.json`。可通过 `--output` 指定文件名；同一路径重跑会覆盖报告，程序会阻止覆盖正在使用的基线。

结果 JSON 保留顶层 **`run`、`summary`、`results`**。成功推荐只用 `recommendation.top3` 承载推荐列表，保留客户 ID 和免责声明，不再生成 `ranking` / `rankings` 及相关排名兼容元数据。响应 Token 用量仍保存在 `response` 中。`run` 记录模型、并行数、启动间隔、限流重试次数等运行配置。比较内容写入独立文件，不改变报告层级。

每次收到完整模型响应后，主线程先保存到同名 `<报告名>_raw/` 目录，再写回测结果。按输入序号命名为 `001.json` 等，文件内含客户ID、响应ID、用量及原样的最终输出文本；成功和结构失败均保存，不包含 API Key 或请求头。回测条目的 `response.raw_path` 指向对应文件。它保存最终正文，不包含思考过程或逐帧 SSE；连接中断等未收到完整响应的失败可能没有该文件。旧报告没有留存的原文无法恢复。

只有主线程写文件，每完成一位客户保存一次完整快照。终端按完成先后显示进度，JSON 中的客户始终按输入顺序排列。Windows 短暂文件占用会有限重试原子替换；持续权限错误仍会报错并保留上一份完整报告。当前不支持自动断点续跑。

自动比较使用仓库已有的 [Qwen3.7-Max 全量基线](outputs/validation_runs/validatable_b_all_93_with_history_rerank.json)，完成于 2026-09-04。该报告共 93 条，91 条完成模型调用，2 条硬过滤排除，没有技术失败。以下指标均以 **93 位客户**为分母，硬过滤排除计为未命中：

| 指标 | Qwen 原始推荐 |
| --- | ---: |
| Top1 | 19/93（20.43%） |
| Recall@3 | 45/93（48.39%） |

当前默认比较只取两轮的**原始模型排名**，对应旧 Qwen3.7 的 Top1 19/93、Recall@3 45/93；忽略旧报告的后处理命中结果。比较报告的 `metric_scope` 为 `model_original_only`，逐项记录命中数、分母、变化百分点和方向。全量客户集合及真实标签需要一致，有失败、未完成或仅离线检查时不判断全量提升。

同一比较文件中的 `prompt_comparison` 另取上次 [Qwen3.8 18:51报告](outputs/validation_runs/validatable_real_20260906_185128.json) 已成功的固定86位客户，基线为**原始 Top1 23/86（26.74%）、Recall@3 41/86（47.67%）**。只有新版全量完成、该86位全部成功，且模型、接口、思考预算和服务地址相同，才展示这一子集比较。旧版5条失败和2条规则排除不进入子集，新版新增成功仍进入全量统计；子集结果不冒充93条全量指标，不合并两轮响应。基线文件缺失时仅执行历史全量比较。

命中统计只使用 `top1_hit/top3_hit`，不再生成重复的 `pre_rerank_*` 或重排配置字段。Top3 每项保留 `rank`（1至3）、`match_score`、理由等推荐信息，名称按产品目录规范；全部候选数查看 `hard_filter.candidate_count`。历史对比仅在读取旧报告时兼容其原始模型命中标记，新比较文件的指标名统一为 `top1` 和 `recall_at_3`。

历史频率重排功能、配置及入口已删除。真实放贷产品仅用于本地核对，不参与模型推荐。旧回测报告作为历史实验记录保留，不会被改写或当作频率配置加载。

Qwen3.7 基线的接口与思考配置不同，因此跨模型比较反映整套配置变化；同模型提示词子集比较更接近本次实验目的，但仍受生成随机性及服务端模型更新影响。

单条模型调用失败仍会记录错误并继续其他客户。批量报告写完时退出码可以为 `0`，请同时检查 `summary.model.failed_count`、`summary.hard_filter.false_negative_count`、`results[].status/error` 和比较报告状态。补测可用 `--customer-id` 写入另一份报告，程序不会自动合并补测结果。

## 离线检查和其他运行方式

以下命令不读取 Key、不调用模型：

```powershell
& $PythonExe -X utf8 -m unittest discover -s tests -v
& $PythonExe -X utf8 main.py --dry-run
```

离线检查处理全部 93 位客户，只验证本地数据、硬过滤和报告生成；模型命中率为 `null`，比较报告为 `not_comparable`。

```powershell
# 指定客户：显式选择覆盖 main 的默认全量行为
& $PythonExe -X utf8 main.py --customer-id C001 --no-compare
# 分层抽样 12 位客户
& $PythonExe -X utf8 main.py --sample-size 12 --no-compare
# 指定报告位置
& $PythonExe -X utf8 main.py --output outputs/validation_runs/my_qwen38_run.json
# 自选历史基线，仍执行同口径检查
& $PythonExe -X utf8 main.py --baseline outputs/validation_runs/validatable_b_all_93_with_history_rerank.json
```

`--customer-id` 可以重复传入，重复 ID 会去重；`--all`、`--sample-size`、`--customer-id` 互斥。独立运行 `evaluate_real_cases.py` 时仍默认分层抽样12条、1个线程、不比较基线；也可显式传入 `--all --workers 16 --baseline <报告路径>`。

## 本机验证记录（2026-09-06）

- 46 项自动化测试通过，包括全 93 位客户的模拟调用、并行上限、乱序完成后保存顺序、单线程写报告、基线统计、HTTP 429 重试、单客户失败隔离、动态候选数量、提示词 JSON 示例、保持模型原始分数、固定86位子集比较、仅返回Top3的解析和失败原文落盘。模拟结果只用于验证程序，不作为模型质量证据。
- `main.py --dry-run` 已在本机执行，93 条全部处理，91 条真实产品通过硬过滤，2 条已知冲突与历史基线一致。
- 此前 Qwen3.8-Max 接口配置已完成 C001 的[真实调用](outputs/validation_runs/local_qwen38_stream_real_20260906_C001.json)：95.488 秒，输入 9,320 Tokens，输出 4,995 Tokens，JSON 解析与重排正常。最终 Top3 为融e借、诚易贷、网捷贷，未命中真实产品闪电贷。这只证明接口兼容性。
- 用户已于18:51手动执行[首次 Qwen3.8 全量回测](outputs/validation_runs/validatable_real_20260906_185128.json)：16路并行，耗时10分08秒，86条成功、5条输出结构失败、2条硬规则排除。因此自动全量比较为 `not_comparable`。在相同86位成功客户中，重排后 Recall@3 从旧模型62/86（72.09%）降至60/86（69.77%），Top1 从29/86（33.72%）升至35/86（40.70%）。这些是相同子集指标，不是完整93条结果。

- 用户20:28执行的回测耗时10分26秒，82条成功、9条结构失败、2条硬规则排除。相同79位两轮成功客户的原始 Recall@3 为37/79 → 38/79，Top1 为21/79 → 19/79，未体现可靠改善。
- 用户21:24执行的 Top3 回测耗时10分钟，87条成功、4条结构失败、2条硬规则排除。原始 Recall@3 为40/87（45.98%），全93条为40/93（43.01%）；与20:28回测相同78位成功客户比较为37/78 → 38/78，尚无可靠的召回提升结论。

## Astra 人工复核

使用当前 Codex 对话中的 Astra 对全部93位客户和12款产品完成一次复核，先记录 Top3，再关联真实产品标签。结果为 Top3 50/93（53.76%）；在 Qwen 23:15回测成功的相同86位客户中，Astra为49/86，Qwen为42/86，但 Top1为15/86对21/86。此复核已有部分标签出现在先前对话中，且使用批量字段阅读与人工记录，**不是盲测，也不是同协议模型API基准**。

[复核结果及逐客户理由](outputs/astra_review/review.json)和[分析结论](outputs/astra_review/findings.md)用于定位提示词、资料及评估口径问题，不加载为线上规则、示例或产品频率。没有额外调用付费模型。

## 当前召回优化提示词

提示词只维护 `prompts/loan_recommendation_system.md` 和 `prompts/loan_recommendation_user.md` 这一套，修改直接更新原文件，不设置版本号、哈希或历史副本。按字段的实际用途决定它如何影响排序：必要条件、补充信息、门槛放宽、产品明确的辅助偏好、客户需求，以及不参与排序的其他信息。不按“看起来更优质”加分；同一事实不重复奖励，达到门槛后不默认数值越高越好。

此次统一修正工资银行、公积金、社保、在职、单位、学历、房车等字段的解释。公积金基数不等于月缴额；问卷额度区间上限不默认作为必需金额；没有明确期限时不偏向长期限；利率区间交叉时不固定先比上限或下限。比较产品独立客群路径，直接选出三个推荐及理由；模型无需输出或维护完整候选排名。`customer_id`、`top3`、`disclaimer` 是模型唯一需要返回的顶层字段。

修订依据见[条件与权重审计](docs/condition_weight_audit_20260906.md)和[首次召回分析](outputs/validation_runs/qwen38_recall_prompt_review_20260906.md)，这些记录描述当时的分析结果。历史回测报告保留用于命中率对比，新报告不再记录提示词版本或哈希；`output_contract=top3_only` 表示仅输出三个推荐。

**最近23:15真实回测有5条结构失败，Top3为42/93；此后针对条件对象混用的提示词修正尚未再次真实回测。** 直接 Run `main.py` 使用当前提示词完整执行93条。重点看原始 Recall@3、失败数和分产品结果，不能将不同轮次的补测与成功结果混成同配置报告。93条已用于提示词修订，其上的改善属于开发集回测结果，泛化效果需要独立新样本验证。

两条已知规则冲突：C047 近 1 个月贷款审批查询 5 次，超过网捷贷上限 4 次；C209 申请区间下限 31 万元，超过家庭综合消费贷最高额度 20 万元。两条均保留在 93 条统计中，不调用模型，需结合业务口径核对样本或规则。

接口设置参考官方[文本生成](https://help.aliyun.com/zh/model-studio/text-generation)、[结构化输出](https://help.aliyun.com/zh/model-studio/qwen-structured-output)、[深度思考](https://help.aliyun.com/zh/model-studio/deep-thinking)。如需使用业务空间专属域名，按[地域与域名说明](https://help.aliyun.com/zh/model-studio/regions)设置服务根地址。

## 主要文件

| 文件 | 用途 |
| --- | --- |
| `main.py` | 默认全量并行回测入口 |
| `evaluate_real_cases.py` | 客户选择、并行评估、进度与结果保存 |
| `recommendation.py` | 共用模型配置、提示词构建和结果校验 |
| `report_comparison.py` | 基线校验、命中率对比与终端展示 |
| `dashscope_client.py` | 流式 JSON 请求、全局启动间隔及限流重试 |
| `api_key.py` / `dashscope_api_key.txt` | 密钥读取逻辑 / 短期配置文件 |
| `hard_filter.py` | 本地硬规则过滤 |
| `data/loan_products.json` | 12 款产品目录 |
| `data/validation_groups/customers_validatable.json` | 93 位可验证客户及真实产品标签 |
| `data/test_customer.json` | 保留的单客户示例，main 默认不再读取 |
| `prompts/` | System 和 User 提示词模板 |
| `outputs/validation_runs/` | 历史及新生成回测报告 |
| `docs/` | 产品库和脱敏历史记录 Excel 原始资料 |

运行问题处理：401 / 403 检查 Key、地域和模型权限；持续 429 降低 `--workers` 并增大 `--request-interval`；中文输出异常使用 `-X utf8`；缺少文件时确认已拉取完整仓库并保留目录结构。
