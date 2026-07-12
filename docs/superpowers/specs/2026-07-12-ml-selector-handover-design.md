# ML Selector 交接与结果存储设计

**日期：** 2026-07-12  
**范围：** ML Evidence Selector V1 交接、V2 启动与实验结果存储  
**目标：** 让没有参与 V1 的项目成员可以快速理解已经完成的工作、正确解读结果，并按冻结协议继续 Selector 2.0。

## 1. 背景与结论边界

V1 已完成候选构造、特征提取、轻量模型训练、多个数据集诊断和 Gate 检查。它提供了方法与失败模式方面的信息，但不是项目的最终实验：

- V1 没有证明 selector 能稳定减少 harmful evidence；
- 已观察到的主要正向信号是更多有用证据进入 Top-10；
- FinanceBench 已在 V1 中被提前使用，因此不能再作为 V2 的未见最终盲测集；
- V2 将重新训练 selector，并重新冻结训练、开发与最终评估协议。

V1 应在交接材料中标记为“探索性诊断实验”。其数据可用于理解方法和失败原因，但不得作为 V2 的独立最终评估证据。

## 2. 文档结构

交接材料集中到 `docs/ml-selector/`，只保留四个职责明确的入口：

| 文件 | 唯一职责 |
|---|---|
| `README.md` | 交接入口；说明当前状态、关键结论、阅读顺序和下一步 |
| `V1_EXPERIMENT_RECORD.md` | 记录 V1 的问题、方法、执行内容、结果、Gate、局限和可复用资产 |
| `V2_EXPERIMENT_PLAN.md` | 定义 Selector 2.0 的研究问题、数据协议、方法、对照、统计和停止门 |
| `V2_EXPERIMENT_TRACKER.md` | 逐次记录 V2 的 Run ID、环境、输入、状态、指标和产物 |

`README.md` 是唯一入口。其他文档不得重复整段结论，只通过链接形成清晰阅读路径。

## 3. 现有文件处理

当前未提交的 `docs/W5 ML selector/` 不保留原目录结构：

- `current_findings_top10_evidence_reranking.md` 的有效内容合并进 V1 记录和交接入口；
- `EXPERIMENT_PLAN.md` 作为 V2 计划的主体内容；
- `EXPERIMENT_TRACKER.md` 作为 V2 Tracker 的主体内容；
- 三份 `historical_*.md` 只提炼研究演进和被推翻假设，不保留全文；
- 四份 HTML 渲染文件删除；
- 两份带日期、且与固定别名逐字节相同的 plan/tracker 删除；
- `DESIGN_REVIEW.md` 的有效结论合并进 V2 计划；
- `MANIFEST.md` 随重复结构一并删除。

已提交的旧 V1 文件按以下方式处理：

- `docs/ml-selector-experiment-plan.md` 的方法和协议摘要合并进 V1 记录，原文件删除；
- `docs/ml-selector-experiment-tracker.md` 的执行状态、Gate 和产出合并进 V1 记录，原文件删除；
- `docs/ml_selector_validation_results.html` 的关键数字和图表引用合并进 V1 记录，生成的 HTML 删除；
- `docs/assets/ml_selector_validation/` 中三张结果图保留，作为 V1 记录的图形证据；
- `docs/data/ml_selector_validation/` 中的小型 JSON 结果与审计文件保留；
- 项目级 `docs/results-summary.md` 保留，并更新其 ML Selector 部分到新交接入口。

## 4. 结果存储分层

### 4.1 Git 中保存

Git 保存体积小、能支持结论核验和后续设计的文件：

- 汇总指标；
- 关键 per-query 统计结果；
- 显著性检验结果；
- split、protocol 和 label schema manifest；
- 候选、特征和标签的审计摘要；
- 必要的配置、随机种子、版本和文件哈希。

现有 `docs/data/ml_selector_validation/` 作为 V1 精简证据包。整理时核对其内容是否足以支持 V1 记录中的数字，不迁移重复副本。

### 4.2 不迁移的 V1 中间产物

老师的私有服务器 `it097952` 上约 986 MB 的 V1 输出不整体迁移。以下内容默认视为可重新生成的中间产物：

- base/generated/validated pools；
- feature cache；
- ranked Top-20 全量 JSONL；
- shards；
- 下载和计时日志；
- smoke-test 临时输出。

交接文档记录原始位置、总体积、目录分类和“不属于长期共享存储”的事实，但不把不可访问的私有路径写成后续工作的依赖。

### 4.3 V2 新产物

V2 每次正式运行必须产生一个可共享的精简结果包，至少包含：

- `run_manifest.json`：Run ID、Git commit、命令、配置、seed、数据与 split hash、运行环境；
- aggregate metrics；
- 支持配对统计的 per-query metrics；
- Gate 结果和统计检验；
- 必要审计摘要与文件哈希。

模型 checkpoint、大型缓存和全量预测不提交普通 Git。只有当它们无法合理重建且继续工作确实需要时，才放入团队可访问的共享存储，并在 Tracker 中记录稳定位置、大小和 SHA256。

## 5. V2 数据与评估规则

- FinanceBench 状态固定为 `EXPOSED_DIAGNOSTIC_ONLY`。
- FinanceBench 不得用于 V2 调参、阈值选择、模型选择、停止决策或最终独立评估。
- 如果 V2 报告 FinanceBench，只能作为历史可比的已暴露诊断结果，并明确标注该限制。
- V2 的最终评估集必须在训练和开发阶段保持未见，并在计划中提前冻结。
- 在查看最终评估结果前，代码、模型、配置、随机种子集合和成功判据必须冻结。

## 6. Tracker 记录规范

V2 每个 Run ID 至少记录：

| 字段 | 含义 |
|---|---|
| Run ID / milestone | 唯一编号及所属阶段 |
| status | `TODO`、`RUNNING`、`DONE`、`FAIL` 或 `STOPPED` |
| Git commit | 运行所用代码版本 |
| command/config | 可复现命令或冻结配置文件 |
| data/split | 数据集、split 与 manifest/hash |
| seed | 随机种子或 seed ensemble |
| host/job | 服务器和 Slurm Job ID；非 Slurm 运行明确标记 |
| outputs | 精简结果包的共享路径 |
| metrics/gate | 关键指标与是否通过停止门 |
| interpretation | 一句话说明该运行改变了什么认识 |

状态更新必须与精简结果文件同步完成，避免 Tracker 声称完成但没有证据文件。

## 7. 接手流程

接手人按以下顺序工作：

1. 阅读 `docs/ml-selector/README.md`；
2. 阅读 V1 记录，理解已尝试的方法和不能再作出的主张；
3. 检查 V1 精简证据包，而不是依赖私有服务器；
4. 阅读并冻结 V2 计划；
5. 从 V2 Tracker 中第一个未完成的 MUST Run 继续；
6. 每次运行后同步更新结果包和 Tracker。

## 8. 验证标准

整理完成后必须满足：

- `docs/W5 ML selector/` 不再存在；
- 交接入口能链接到所有保留文档和精简结果；
- 不存在内容完全相同的 plan/tracker 副本；
- V1 被明确标记为探索性诊断，FinanceBench 暴露限制在入口、V1 记录和 V2 计划中一致；
- V1 文档中的关键数字能追溯到 Git 中的 JSON/CSV；
- V2 Tracker 包含完整的运行记录字段；
- 仓库不新增大型缓存、全量 JSONL、checkpoint 或私有凭据；
- 所有相对链接有效，Git 状态只包含本次整理范围内的预期改动。
