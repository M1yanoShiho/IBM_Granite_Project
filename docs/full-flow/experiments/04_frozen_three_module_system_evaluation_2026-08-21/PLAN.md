# 冻结三模块系统完整实验路线（v4）

**日期：** 2026-08-22  
**状态：** `MASTER ROADMAP FROZEN / GOAL 2 PASS / GOAL 3 ACTIVE / NO HELD-OUT SCORED`  
**取代范围：** 本文件取代 v3 的“单一连续大任务”执行方式；科学实验矩阵不变。v3 保存在 `snapshots/PLAN_v3_2026-08-22.md`。

## 0. 这份文件怎样使用

这是一张**总路线图**，不是一个一次性执行到底的大目标。

整项工作分成五个防偏移目标。按用户 2026-08-22 的最新执行指令，
这些目标是连续任务中的检查点：

- 目标内部：可以连续运行、自动接力、自动恢复，尽量不中断；
- 目标结束：必须交付结果并给出 `PASS` 或 `FAIL`；
- `PASS`：立即把下一目标设为唯一 active goal 并继续；
- `FAIL`：停在当前目标内修正，不得带着问题继续消耗正式算力；
- Goal 5 `FINAL PASS/FAIL` 是整项任务的终点。

因此，总目标不变，但实际执行单位是：

```text
Goal 1 数据与评分准备
  -> PASS 自动接力
Goal 2 五系统接线与开发集检查
  -> PASS 自动接力
Goal 3 主系统比较
  -> PASS 自动接力
Goal 4 模块消融
  -> PASS 自动接力
Goal 5 统计、表格与最终核验
```

## 1. 总目标

最终只回答三个问题：

1. 冻结后的完整系统 `Ours` 是否优于四个公平、互补的 RAG baseline？
2. Retriever、Selector、Generator 三个模块分别对完整系统贡献什么？
3. 变化发生在检索、证据保留、答案、引用，还是最终可靠答案率？

HotpotQA、MuSiQue 和 RGB 分别报告，不用一个总平均掩盖单个数据集的退化。

## 2. 冻结的系统和比较对象

### 2.1 Ours

```text
Hybrid RRF Retriever
  -> NLI harm/protect Selector
  -> grounded Granite GR-C Generator + TRUE verifier
  -> cited answer
```

- Retriever：StrongBM25 与 Granite dense embedding 用 RRF 融合，`k=60`，输出 Top10。
- Selector：冻结的 NLI harm/protect selector，seed 13，threshold `0.9212157130241394`，最多删除 2 条，至少保留 7 条，异常时全部保留。
- Generator：冻结的 Granite 4.1-3B grounded GR-C generator；主实验报告已独立训练的 seeds 13、42、73，不挑最好 seed；greedy decoding，最多 256 个新 token。
- TRUE 是运行时 verifier；MiniCheck 只用于实验结束后的独立评分，不属于系统模块。

### 2.2 四个 baseline

| System | Retriever | Selector / Pruner | Generator |
|---|---|---|---|
| Dense RAG | Granite dense Top10 | keep-all | Direct base Granite |
| Hybrid RAG | Hybrid RRF Top10 | keep-all | Direct base Granite |
| Granite Rerank RAG | Hybrid RRF Top40 -> Granite reranker -> Top10 | keep-all | Direct base Granite |
| Provence RAG | Hybrid RRF Top10 | official Provence, threshold 0.1 | Direct base Granite |
| **Ours** | Hybrid RRF Top10 | NLI harm/protect | Grounded Granite GR-C + TRUE |

四个 baseline 分别代表：普通语义 RAG、较强混合检索、标准相关性重排、公开上下文剪枝。`Granite Rerank RAG` 的 reranker 属于 Retriever 阶段，最终仍只给 Generator 最多 10 条证据；它不是 Selector。

- Granite reranker：`ibm-granite/granite-embedding-reranker-english-r2`；Goal 2 固定实际 revision/hash。
- Provence：`naver/provence-reranker-debertav3-v1@ef49e233e3c6e50efc476c68f1390f8a63add4d4`，threshold `0.1`，`always_select_title=True`，`reorder=False`。

### 2.3 三个消融

从 `Ours` 出发，每次只替换一个模块：

| Configuration | Only change |
|---|---|
| Full | 无，完整系统 |
| w/ Dense Retriever | Hybrid RRF -> Granite dense |
| w/ Top-10 | NLI harm/protect -> keep-all Top10 |
| w/ Direct Generator | Grounded GR-C + TRUE -> Direct base Granite |

`Full`、Retriever 消融和 Selector 消融使用 GR-C seed 13；Direct Generator 消融使用冻结的 base Granite。`Full` 直接复用主系统实验中的 Ours seed-13 输出。

## 3. 冻结数据

| Dataset | n | Role |
|---|---:|---|
| HotpotQA | 400 | 多跳问答与引用 |
| MuSiQue answerable | 400 | 更复杂的多跳问答 |
| RGB noise | 300 | 噪声候选池下的全过程表现 |

三套数据是项目此前预留、尚未正式评分的系统级测试集；不是这次临时挑选，也不是报告模板硬性指定。ordered IDs、顺序和 hashes 保持不变；MuSiQue 只使用配对 ID 的 `#ans` 版本。

正式数据必须按题隔离：每道题只能在它自己的候选池中检索，不能把不同问题的候选拼成一个全局语料库。

## 4. 指标与最终表格

报告表格只用五个短代号：

| Code | Meaning |
|---|---|
| Ret. | 官方支持证据进入 Retriever 最终 Top10 的比例 |
| Sel. | 官方支持证据最终保留在 Generator 输入中的比例 |
| Ans. | HotpotQA/MuSiQue token F1；RGB accuracy |
| Cit. | 逐题 MiniCheck citation F1 的宏平均 |
| RAR | 答案正确、引用合法且引用完全支持答案的题目比例 |

`Ret.` 与 `Sel.` 使用同一批官方 support units 作分母；检索漏掉的证据在 `Sel.` 中也计 0。`Cit.` 对空答案或非法引用记 0。`RAR` 逐题为二值：normalized gold-alias exact match 通过、引用索引合法，且 MiniCheck citation precision = recall = 1。

`RAR` 是主指标；其余四项解释提升或退化发生在哪一步。最终主文只放：

1. Table 1：五个完整系统的比较；
2. Table 2：三个模块的消融。

每张表分别包含 HotpotQA、MuSiQue、RGB 三个面板，第一列始终是 `System` 或 `Configuration`。英文模板见 `RESULT_TABLES.md`。

## 5. 所有阶段共同遵守的规则

### 5.1 公平性

1. 同一数据集的所有系统使用完全相同的 query IDs 和原始候选池。
2. Runtime 文件只含问题、候选证据和 source IDs；gold answer、support labels 与 `component_id` 只存在于 scorer-only sidecar。
3. 系统运行时不得访问 scorer-only sidecar；gold 只在答案冻结后用于评分。
4. 所有系统固定相同的最终 Top10 上限、Generator 输入 token 上限、上下文格式、截断规则、citation prompt、引用格式、输出长度和 greedy decoding。
5. Granite reranker 可看 Hybrid Top40，但 `Ret.` 按 rerank 后最终 Top10 计算。
6. Provence threshold 固定为 0.1；所有模型 revision/hash 在正式运行前冻结，不在 held-out 上调参。
7. 所有系统由同一评分程序计算五项指标；TRUE 与 MiniCheck 严格分离。
8. held-out 结果出现后，不改阈值、样本、seed、模型、prompt 或主指标定义。

### 5.2 失败处理

- 生成失败、空答案、非法引用或评分失败都保留在共同分母中；对应无法通过的 `Ans./Cit./RAR` 记 0，不静默删除样本。
- 每个系统必须覆盖相同 query 集合，并记录 missing/error reason。
- 某系统在某数据集的 runtime error rate 超过 1%，该数据集 bundle 判为技术无效，先修复并整包重跑，不能用剩余样本凑表格。

### 5.3 统计

- Table 1：Ours 的 `Ans./Cit./RAR` 报告三个独立 Generator training seeds 的 mean ± sample SD；固定上游产生的 `Ret./Sel.` 报一个数。
- 四个确定性 baseline 各运行一次。不得把同一 checkpoint 重跑三次冒充三次独立实验。
- Table 2：`Full`、Retriever 消融和 Selector 消融报告 GR-C seed 13 的单点结果；Direct Generator 消融报告 frozen base Granite 的单点结果。
- `RAR` 使用 paired component-cluster bootstrap，10,000 次，bootstrap seed 13，报告差值与 95% CI。
- 三个 seed 的 SD 只说明 Generator 训练随机性下的稳定性，不作为显著性检验。

## 6. 五个独立执行目标

### Goal 1 — 数据与评分准备

**目的：** 先保证“输入没有泄漏、每道题的候选池正确、五个指标能算对”。

**本目标只做：**

1. 核对三套 frozen ID manifest 的数量、顺序和 hash；
2. 建立按 query 隔离的 runtime bundle；
3. 将 gold answer、support labels、`component_id` 放入物理独立的 scorer-only sidecar；
4. 用 synthetic/revealed development fixtures 实现并验证 `Ret./Sel./Ans./Cit./RAR`；
5. 固定失败记分规则和机器可读字段。

允许程序化生成 held-out manifest、runtime bundle 和 sidecar，但只能核对数量、schema 与 hash；不得人工查看 held-out 内容，不得生成答案、评分或据此修改方法。

**交付物：** 数据 manifest、runtime/scorer 隔离说明、scorer validation report、`PASS/FAIL`。

**PASS 条件：** 400/400/300 IDs 与 hash 一致；候选池逐题隔离；runtime 无 gold 字段；五指标在 development fixture 上通过已知答案测试；失败样本不被删除。

**停止边界：** 达到 PASS 后立即停止。不得接 baseline，不得运行正式 held-out。

**预计：** 2–4 小时。

### Goal 2 — 五系统接线与开发集检查

**依赖：** Goal 1 `PASS`。本目标已由用户启动并完成。

**目的：** 在不消耗正式测试集的前提下，证明全部正式配置都能由同一入口运行。

**本目标只做：**

1. 将 Dense、Hybrid、Granite Rerank、Provence、Ours 接入统一 runner；
2. 冻结 Direct Granite、dense embedder、Granite reranker、Provence、Selector、三个 GR-C adapter、TRUE、MiniCheck 的 revision/hash；
3. 固定候选预算、token cap、prompt、截断、引用格式和输出路径；
4. 在 synthetic/revealed development 数据上检查全部 10 个 run arms：4 baselines + Ours seeds 13/42/73 + 3 ablations；
5. 核对每个 arm 都能产生 retrieval trace、selected context、answer 和 score。

**交付物：** 10 份冻结配置、model/config manifest、smoke report、`PASS (= READY)/FAIL`。

**PASS 条件：** 10 个 run arms 全部完成至少一题；system identity 正确；无 gold leakage；所有输出可由统一 scorer 读取；正式输出目录仍为空。

**接力边界：** `PASS (= READY)` 后立即冻结本目标交付物，并将 Goal 3
设为新的唯一 active goal；不得在 Goal 2 内运行 held-out。

**预计：** 4–8 小时。

### Goal 3 — 主系统比较与 Table 1

**依赖：** Goal 2 `PASS (= READY)`。本目标已按用户自动接力指令启动。

**目的：** 回答完整系统是否优于四个 baseline。

**运行内容：**

- 4 个 baseline：`4 x 1,100 = 4,400` 个答案；
- Ours seeds 13/42/73：`3 x 1,100 = 3,300` 个答案；
- 合计：**7,700** 个正式答案输出。

三个数据集可以在两张 GPU 上按完整 dataset bundle 调度；同一数据集内的 7 个主实验运行臂不得拆散后拼接。任务内部允许连续运行、断点续跑和自动评分。

**交付物：** 逐题结果、五指标汇总、Table 1、Ours 的 mean ± SD、Ours 与各 baseline 的 paired RAR CI、运行审计、`PASS/FAIL`。

**PASS 条件：** 每个 arm 覆盖全部冻结 IDs；错误率不超过 1%；Table 1 可追溯到逐题记录；Ours 三个 seed 均完整，不能挑最好 seed。

**接力边界：** Table 1 完成并核验为 `PASS` 后，冻结本目标输出并自动将
Goal 4 设为新的唯一 active goal。

**预计：** 18–30 小时。

### Goal 4 — 模块消融与 Table 2

**依赖：** Goal 3 `PASS`。

**目的：** 说明 Retriever、Selector、Generator 各自承担的作用。

**运行内容：** 三个消融分别跑 1,100 题，共 **3,300** 个新答案。Retriever/Selector 消融使用 GR-C seed 13，Direct Generator 消融使用 frozen base Granite；`Full` 使用 Goal 3 已冻结的 Ours seed-13 结果，不重新生成。Goal 4 必须读取 Goal 3 保存的同一 runtime/config fingerprints，避免跨阶段输入漂移。

**交付物：** 三个消融的逐题结果、Table 2、Full 与各消融的 paired RAR CI、模块贡献结论、`PASS/FAIL`。

**PASS 条件：** 每行只替换一个模块；其余输入和设置与 Full 一致；所有 frozen IDs 完整；Table 2 可追溯到逐题记录。

**接力边界：** Table 2 完成并核验为 `PASS` 后自动进入 Goal 5；不得改方法
或补做内部网格。

**预计：** 10–18 小时。

### Goal 5 — 统计、表格与最终核验

**依赖：** Goal 4 `PASS`。

**目的：** 将已有结果变成可以直接放进报告、且能被复核的最终证据。

**本目标只做：**

1. 复核 mean、sample SD、paired bootstrap CI；
2. 检查 IDs、共同分母、missing/error、hash 和表格数值来源；
3. 生成最终 CSV、JSON、Markdown/LaTeX 两张表；
4. 写清哪些结论成立、哪些只有正向信号、哪些不成立。

**交付物：** Table 1、Table 2、机器可读汇总、bootstrap 文件、final audit、`FINAL PASS/FAIL`。

**PASS 条件：** 表中每个数值可追溯到逐题结果；Markdown/LaTeX 与 CSV/JSON 一致；没有选择性删除、挑 seed 或越过声明范围。

**停止边界：** 不产生新答案，不回头调参；若发现正式结果无效，只报告问题并明确需要重跑哪个完整 bundle。

**预计：** 2–4 小时。

## 7. 连续运行与算力安排

“持续执行”覆盖五个目标，但每次只能有一个 active goal。PASS 才能自动接力，
FAIL 必须留在当前目标修复。

- Goal 1–2 主要是准备与小规模检查；
- Goal 3 是最长的主实验，可在两张 A4000 上连续运行；
- Goal 4 在 Goal 3 PASS 后自动接力并单独连续运行；
- Goal 5 不需要新的生成算力。

Goal 3 与 Goal 4 合计 **11,000** 个正式答案输出。按 2026-08-21 最后确认的两张 RTX A4000 状态估算：

- 条件理想、依赖已缓存、无整包重跑：约 **2–3 个自然日**；
- 更安全的项目排期：约 **3–4 个自然日**；
- 若只有一张 GPU、Provence/Granite reranker 安装冲突，或某 bundle 错误率超过 1%，时间会延长。

启动 Goal 3 前必须重新确认服务器实时占用；“2–3 天”是有条件目标，不是无条件保证。

## 8. 结果保存

```text
docs/full-flow/experiments/04_frozen_three_module_system_evaluation_2026-08-21/
  GOAL_1_HANDOFF.md
  FINAL_REPORT.md                         # Goal 5 后产生
  artifacts/preflight_manifest.json      # Goal 1/2 后产生
  results/per_query_metrics.csv           # Goal 3/4 后产生
  results/summary_metrics.csv             # Goal 5 后产生
  results/bootstrap_ci.json               # Goal 3/4/5 后产生
```

服务器保存逐题生成和 trace：`runs/experiment04/<dataset>/<configuration>/<seed>/`。Git 只保存可复核的小文件。CSV 中均值、标准差、方差、seed、dataset、system、metric、n_queries 和 missing_reason 必须是独立字段。

## 9. 本轮明确不做

- 不重训 Selector，不再搜索 selector threshold/delete cap；
- 不重训或挑选新的 Generator recipe；
- 不做完整三模块全因子网格；
- 不加入无法公平复现的额外大型 RAG 系统；
- 不把 MuSiQue unanswerable、RGB counterfactual 或 NQ 混入主表；
- 不把旧开发结果填进新的系统表；
- 不用 held-out 结果修改方法。

Retriever、Selector、Generator 已完成的模块实验继续保留，用来解释为什么选择冻结方法；本计划不重复模块内部实验。
