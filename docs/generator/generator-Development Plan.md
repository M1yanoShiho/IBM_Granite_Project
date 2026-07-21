# Generator 模块开发计划

> 分支：`refactor/three-module-baseline`
> 模块路径：`src/evidence_rag/generator/`
> 测试路径：`tests/generator/`
> 文档路径：`docs/generator/`（如需新建）

---

## 1. 背景与目标

当前主线是三模块架构 `Retriever → Selector → Generator`。Generator 目前只有两个基础实现：`ExtractiveGenerator`（占位）和 `GraniteGenerator`（LLM 驱动，做单遍生成 + generation-time citation）。

本计划要在此基础上实现一个 **checklist 驱动的「生成 → 验证 → 修补」闭环**，核心贡献是：

- **不依赖模型生成时自己声明的引用**，而是在生成后独立验证每个 claim 与证据的支持关系，产出**验证过的、更可信的 citation**。
- 用一份**运行时从 query 推导转化得出的 checklist**（`QueryChecklist`），同时驱动两条核查：
  - **Completeness（补漏）**：检查答案是否覆盖 checklist 里所有 `required_facts`，缺口则带着更细的子问题回到证据里再找一遍，找到补上、找不到诚实标注。
  - **Faithfulness / Attribution（删假）**：检查答案里每个 claim 是否有证据支持，无支持的（含被证据反驳的）claim 删除或标记存疑。


---

## 2. 架构约束

来自 `docs/README.md`，与本模块直接相关：

- Generator 只能使用 Pipeline 交回的 `SelectedEvidenceSet`，**不得自行发起新的检索**。本计划的「重看一遍」严格限定在**已选证据范围内**，不触碰检索库。
- 跨模块接口 `GenerationResult`（`answer` + `cited_evidence_ids`）**当前不改动**。所有验证中间结果都留在 Generator 内部。
  - 注意 contract 硬校验：非空答案必须至少有一个引用；空答案不能有引用（拒答通过「空答案」表达）。

---

## 3. 核心流程

```
输入：Query + QueryChecklist + SelectedEvidenceSet
  │
1. 初版生成：用 selected evidence 生成初版答案
  │
2. claim 拆分：拆成 atomic + self-contained 的 claim 列表（LLM 拆）
  │              并回检「claim 是否忠实于答案」，防验证器自身幻觉
  │
3. 逐 claim 验证（attribution）：每个 claim × 每条 evidence
  │   做 NLI entailment 判断 + entity 一致性检查
  │   ├─ entailment 且 entity 一致 → 有支持，记录 claim ↔ evidence id
  │   └─ neutral / contradiction / entity 不一致 → 无支持（contradiction 单独记录）
  │
4. completeness 检查：对 checklist 每个 required_fact
  │   ├─ 已覆盖 → 跳过
  │   └─ 未覆盖 → 生成一个具体缺口子问题
  │
5. 带细问题重看证据（限已选证据内）：
  │   ├─ 补漏（触发1）：缺口子问题回证据里找 → 找到补进答案+记引用；找不到标注「未检索到」
  │   └─ 删假（触发2）：无支持 claim 回证据再确认 → 确实无支持则删除/标记存疑
  │
6. 组装最终输出：
      ├─ answer：修补后的答案
      └─ cited_evidence_ids：步骤 3/5 验证出来的真实引用（非模型声明）
```

### 关键设计决定

| 决策 | 结论 |
|---|---|
| claim 拆分方式 | **LLM 拆**，强制 atomic + self-contained（消解指代）；拆后回检「claim 忠实于答案」 |
| 支持关系判定 | 主线用**专用 NLI 模型**；LLM-as-a-judge 仅作二审/对照（目前先不弄） |
| claim × evidence | 第一版**全跑不剪枝**（selected evidence 量小，可接受） |
| 支持阈值 | `entailment 且 entity 一致 = 有支持`；neutral/contradiction 均记无支持，contradiction 额外记录 |
| entity 一致性检查 | **完整版**（见 4） |
| 补检索范围 | 仅限已选证据内，**不碰检索库** |
| 迭代边界 | 「验证-修补」只跑**一轮**，补进的内容来自证据、天然带来源，不再二次验证，防循环 |
| checklist 来源 | 前两阶段现成可用，开发期用手写假 checklist|

---

## 4. Entity 一致性检查（完整版规格）

目的：捕获「证据支持了 claim 的表述，但实体张冠李戴」的错误（对接项目的反事实干扰特色）。

**覆盖的实体类型：**

- 组织/公司实体（含别名归一，如 `Pfizer` / `Pfizer Inc.` / `辉瑞` 视为同一实体）
- 人名
- 年份 / 日期 / 时间区间
- 数字与金额（含单位与量纲，如 `$1.2B` / `12亿美元`）
- 地区 / 地理位置
- 产品 / 项目 / 型号名称

**判定流程：**

1. 从 claim 与候选支持证据中分别抽取上述类型的关键实体。
2. 对实体做规范化（大小写、缩写展开、货币/数字量纲统一、常见别名映射）。
3. 逐类型比对 claim 与证据的实体集合。
4. 若 claim 中任一关键实体在证据中找不到一致对应（或与证据中同角色实体冲突），判为 **entity mismatch**，即使 NLI 判 entailment 也记为「无支持」。
5. entity mismatch 的 case 单独记录，作为反事实干扰捕获能力的评测证据。

---

## 5. 分工


### 内部解耦接口

放在 `generator/` 下的内部 models 文件：

- **`DraftAnswer`**：初版答案文本 + 拆分出的 claim 列表（每个 claim 含 claim 文本、在答案中的位置）。
- **`VerificationReport`**：
  - 每个 claim 的验证结果：支持 / 无支持、支持它的 evidence ids、entity 是否一致、是否被 contradiction。
  - 每个 required_fact 的覆盖情况：已覆盖 / 未覆盖（含缺口子问题）。

数据流：A 产出 `DraftAnswer` → B 消费产出 `VerificationReport` → A 依据它做修补组装。

---

### 负责人 A：生成与编排（Generation & Orchestration）

| 阶段 | 任务 | 产出 |
|---|---|---|
| A1 | 复用/整理现有 `GraniteGenerator` 初版生成逻辑，产出 `DraftAnswer` | 初版答案生成器 |
| A2 | claim 拆分（LLM 拆，强制 atomic + self-contained），含「claim 忠实于答案」回检 | claim 拆分器 |
| A3 | 最终修补组装：读 `VerificationReport`，执行补漏、删假，组装 `GenerationResult` | 修补编排器 |
| A4 | 带细问题重看证据（已选证据内，针对缺口子问题调 Granite 抽答案） | 证据重查器 |
| A5 | 诚实拒答/标注（找不到→标「未检索到」；空答案走拒答，满足 contract 校验） | 拒答处理 |
| A6 | 单轮上限控制（验证-修补只跑一轮，防循环） | 边界控制 |

### 负责人 B：验证与核查（Verification & Attribution）

| 阶段 | 任务 | 产出 |
|---|---|---|
| B1 | NLI 模型选型并接入（如 DeBERTa-NLI），封装 claim × evidence entailment 判断 | NLI 验证器 |
| B2 | claim ↔ evidence 逐对验证（全跑不剪枝），输出每个 claim 支持情况 + 支持它的 evidence ids | 归因验证核心 |
| B3 | entity 一致性检查（见 4） | entity 核查器 |
| B4 | completeness 检查（对 checklist 每个 required_fact 判断是否覆盖，未覆盖生成缺口子问题） | 完整性检查器 |
| B5 | 汇总产出 `VerificationReport`；单独记录 contradiction 信号（供亮点分析） | 验证报告组装 |

### 两人共同 / 交界任务

| 任务 | 说明 |
|---|---|
| 冻结内部接口 | 定 `DraftAnswer` / `VerificationReport` 格式 |
| fixture 数据 | 各自备假数据独立开发：A 用假 `VerificationReport` 测 A3；B 用假 `DraftAnswer` 测 B2 |
| 评测搭建 | 接入带引用标注的数据集（ALCE/ASQA 类），定义核心指标，对照 generation-time citation baseline |
| 集成测试 | A+B 接起来跑通全流程，放 `tests/generator/`，保证符合 `GenerationResult` contract |
| 双指标报告 | 按 `docs/README.md` 第 6 节：报告模块指标 + 接入完整 pipeline 后的系统指标 |

---

## 6. 评测方案

按 `docs/README.md` 第 5 节要求，测试要保留过程信息，能定位失败发生在哪一环。

**测三件事：**

1. **citation 准确率**：验证出来的 citation 是否比 generation-time citation 更准（需带引用标注的数据集，或人工核查一批）。
2. **completeness 修补率**：completeness 检查发现的缺口中，有多少被成功补上（且补的内容确有证据）。
3. **幻觉删除率**：无支持/被反驳的 claim 中，有多少被正确删除或标记。

**额外指标：**

- entity mismatch 捕获数：验证器抓到多少「表述被支持但实体错误」的 case。
- contradiction 捕获数：被证据直接反驳的 claim 数量。

**对照基线**：`GraniteGenerator` 的 generation-time citation。

**报告要求**：同时给出模块级指标与接入完整 pipeline 后的系统级指标；不能只凭模块指标变好就断言系统变好。现在先不用管。

---

## 7. 推进顺序（避免互相等）

1. **第一步（一起）**：冻结内部接口 + 各自搭 fixture。
2. **并行主体**：A 做 A1 → A2；B 做 B1 → B2 → B3。三块互不依赖。
3. **中段汇合**：A4/A5 与 B4 完成后，第一次集成，跑通「生成 → 验证 → 修补」骨架。
4. **收尾（一起）**：评测搭建 + 调优 + 双指标报告。

---

## 8. 待定 / 依赖项

| 事项 | 状态 | 是否阻塞 |
|---|---|---|
| 带引用标注的评测数据集最终选型 | 待定 | 否，但需早于调优阶段确定 |

---

## 9. 范围边界（本计划不做）

- 不修改 `contracts/`（跨模块接口）。
- 不发起检索库层面的新检索（仅在已选证据内重看）。
- 不做多轮迭代式生成（只跑一轮验证-修补）。
- 不向 `GenerationResult` 增加对外字段（如需，留待第二阶段三组讨论）。
