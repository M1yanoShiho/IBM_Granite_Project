# Selector 数据集提案(spec §14 第 1 项)

**日期:** 2026-07-20

**状态:** 提案,供团队会议决策;**未批准执行**。批准后 materializer 走
brainstorm → spec → plan → TDD 落地。

**用途:** 为 Gated Corroboration Selector 的 E1(边/簇检测组件评估)与
E2(gate-on/off 配对确认)提供训练无关、provenance 干净、跨模块可共享的评估数据。
关联:[spec](../superpowers/specs/2026-07-20-gated-corroboration-selector-design.md) §12/§14、
[RELATED_WORK](RELATED_WORK.md)、[TRAINING_PLAN](TRAINING_PLAN.md)(旧 V2 计划,本提案取代其数据部分)。

---

## 1. 一句话决定

**不沿用旧 artifacts,重写构造代码,保留构造配方。**

| 层 | 决定 | 理由 |
|---|---|---|
| 旧 artifacts(冻结 300/600 集、缓存、split manifest) | **弃用** | 查询集已被多轮调参污染(记忆标注 contaminated q-set);历史缓存事故(corpus-fingerprint bug f63e905);且物理上不在本仓库(`data/` 只有一个 sample contract) |
| 旧构造代码(`src/niah/`) | **不移植,重写** | 旧格式与 `JsonlDatasetAdapter` 不兼容;构造逻辑不大,重写比带旧假设移植干净 |
| 旧构造配方(五条不变量) | **保留** | 经 review 硬化过的确定性 provenance 协议,是"harmful 标签来自 deterministic rule 而非模型判断"(judge-kappa≈0)的**唯一现成实现**——丢了它 E2 的 harm 指标与 E1 的 missed-conflict 就没有标签来源 |

## 2. 这份数据要供养三件事

| 消费方 | 需要的标签 | 来源 |
|---|---|---|
| **E2** harmful-in-context ↓ | 每题哪条候选是植入的假针 | 反事实注入的 mutation log(deterministic) |
| **E2** Required Recall 非劣 −0.01 | 每题哪条证据是答对必需 | official qrels → `GoldCase.relevant_document_ids` |
| **E1** false-conflict / missed-conflict rate | 每题的 gold 答案别名集 | official answers → `GoldCase.reference_answers` |

现有 `GoldCase` 的 `relevant_document_ids` + `reference_answers` **已经覆盖后两项**。
唯一的 schema 缺口是第一项(harmful 针标签),见 §5。

## 3. 基座数据与切分

**基座:** dpr-w100(Wikipedia 100-word 段落)语料 + **NQ**(主域)与
**TriviaQA**(迁移域)查询——两者经 ir_datasets 都带 official 短答案与别名,
且正是先前认证运行与 07-10 救援文档预注册的 NQ→TriviaQA 迁移对。
(精确 ir_datasets id 属 materializer 落地细节;确认 dpr-w100 语料版本与模块一一致。)

**切分(注意:门是规则式,无学习参数,故无需 train split):**

| split | 用途 | 备注 |
|---|---|---|
| **NQ dev** | E1 组件评估报告 + 参数敏感性扫描(次要) | 唯一"看得到结果"的开发集 |
| **NQ test** | E2 主确认 + E1 test 报告 | 冻结后一次性运行 |
| **TriviaQA transfer** | 域漂移测试:NQ-dev 参数原样搬过来跑 | 把 ContractNLI −0.134 的事故形态变成设计内必测项(spec §7.1 失效方向验证) |

**无 train split** 是相对旧 V2 LightGBM 计划(需 2000 train query)的实质简化:
规则式门不训练,可信度反而更高(研究者自由度更少)。

## 4. 反事实注入配方(保留自旧 recipe,逐条冻结)

每题至多注入一根确定性假针:

1. 只接受**单一 normalized gold 值**、且某个 gold surface alias 在 needle 段落中**恰好出现一次**的查询(多答案题排除——门在多答案形态必须关闭,spec §1);
2. 替换值从 **seed 冻结的 answer bank** 选取,且不等于当前 gold;
3. 替换值与 gold 属**同一机械字符串类别**(integer / decimal / year-date / proper-name token-length bucket / common-noun token-length bucket);无法归类或无同类替换 → 丢弃该题;
4. 替换后硬性满足:所有 gold alias 无残留、替换值不属于 gold aliases、只有目标 span 变化、文本可由 mutation log 反向恢复;
5. **mutation log** 记录:answer-bank hash、原值、替换值、字符串类别、seed、变换位置、变换前后文本 hash;
6. **零重叠审计**:fresh 查询与任何旧 split 在 query / 父页面 / answer entity / passage hash / 模板族上零重叠(旧 manifest 在旧分支,`git show` 取来做 hash 对撞);冻结 manifest/hash **在看任何模型结果之前**(预注册)。

全程确定性、CPU、可单测,无 GPU、无人工标注、无 LLM 裁判。

## 5. harmful 标签放哪(schema 缺口——承重决策)

**硬约束(已核对源码):** `GoldCase` 与 `DatasetManifest` 均为
`FrozenModel(frozen=True, extra="forbid", strict=True)`。所以 harmful 针标签
**无法**塞进现有 `GoldCase`,也无法在不改 schema 的情况下从 manifest 引用新文件。

两个方案:

| 方案 | 做法 | 取舍 |
|---|---|---|
| **A(推荐)** manifest 小版本升级 | `DatasetManifest.schema_version` 1.0 → 1.1,加**可选** `provenance_file: NonEmpty \| None = None`;sidecar `provenance.jsonl` 按 query_id 记 harmful evidence id + mutation record;并入 `dataset_signature` | 向后兼容(旧 manifest 缺字段→None);harmful 标签纳入同一签名/复现体系;**动了跨模块共享契约,需 infra owner 签字** |
| **B(零改动回退)** 约定命名 sidecar | eval harness 直接从数据目录按约定读 `provenance.jsonl`,不进 manifest | 零契约变更;但 harmful 标签不被 manifest 声明、不进 signature,可发现性与复现性弱 |

推荐 A:与本项目"frozen 契约 + signature 复现"的纪律一致,harmful 标签是评估的
一等公民,应纳入签名。这是跨模块决策(见 §9)。

## 6. gate-active 子集与样本量

门只在**冲突池**触发(存在竞争簇 + 票差 ≥ margin)。无冲突的题门保持沉默,
对 harmful-rate 差值零贡献。因此配对比较的**有效样本 = 假针落进池且形成竞争簇的题**。

- 需要数据**富集冲突题**:注入的反事实要真被检索进 top-k 且被抽取成竞争答案——
  这取决于检索器(见 §9 pool 耦合);
- **先在 NQ dev 量 gate-fire rate**,据此按配对 discordance 做 power 计算,
  **在冻结 test 前**定 N(纪律沿用旧 §4.3;不事后报 observed power);
- 统计单位 = query,不是 top-k 里的 passage(避免伪重复)。

## 7. 参数策略

- **margin=2 / support_cap=1 预注册为固定值**(主),依据 spec §7 设计论证而非 dev 表现——消除"调到测试集"的自由度,统计上最干净;
- dev 上的 margin∈{1,2,3} / cap∈{1,2,off} 敏感性扫描作**次要**证据(示稳健,不驱动 Gate);
- corroboration α **冻结在已认证的 0.6**(不在本轮重调),使门成为 gate-on/off 之间**唯一**变量。

## 8. 统计协议(沿用项目现行)

- 主对照:同一重排下 gate-on vs gate-off,**query-level 配对**随机化检验 + 按父页面 grouped bootstrap CI;
- 双 Gate:harmful-in-context 显著 ↓(点估计方向 + CI)**且** Required Recall 非劣下界 −0.01;
- 次级/敏感性用 Holm 校正;报告 effect size + CI + per-query raw,不只 p 值;
- 一次 fresh run 产出全部冻结系统,不先看 gate-on 再决定是否跑 baseline。

## 9. 跨模块决策项(会议决定,不由 selector 单方定)

1. **语料规模与索引**(**模块一**):建议沿用先前认证运行的 ~1M dpr-w100 qrels-aware 子采样(全量 21M fp32 不可行,记忆:≈730GB→IVFPQ);能否复用现有索引由模块一定;
2. **候选池冻结 vs 实时检索 + 池耦合**:E2 gate-on/off 要用**同一池**——建议**预冻结每题 top-k 池**(可复现 + 两臂同池),但池由模块一的检索器生成,且"假针是否落池"依赖检索器;此为 selector↔retriever 的显式接口决策;
3. **provenance sidecar 契约**(§5 方案 A):`DatasetManifest` 1.1 需 infra owner 批;
4. **样本量**:dev 量完 gate-fire rate 后预注册 N(§6);
5. **FinanceBench 继续排除**确认(团队既有约定;金融数字投票碎裂虽是门的理想压力域,但按约定不进 selector)。

## 10. 明确不做

- 不做 train split / 学习型门(规则式,无参可训);
- 不沿用旧 artifacts / 旧 `src/niah` 代码;
- 不做真 SAME_SOURCE(dpr-w100 pre-chunked,document_id=passage id,父文章分组不可得——与 spec §6.3/§13 一致,dedup 在本域多为隐性,到多段落来源的域才发力);
- 不用 FinanceBench;不做多答案/列表题;
- 不用 LLM 当标签裁判(judge-kappa≈0);gold 标签只来自 official + deterministic mutation log。

## 11. 构建计划(批准后)

materializer = 纯确定性 CPU 工具,吃 base QA 集 → 出
`documents.jsonl` + `queries.jsonl` + `gold_cases.jsonl` + `provenance.jsonl` + `manifest.json`(§5-A 格式),
自带 mutation log 与零重叠审计;全程 TDD,无 GPU。落地顺序:
brainstorm → spec → plan → TDD;E1 的 false/missed-conflict 计算器同期实现
(spec §12,其 slurm 与 E1 台账 BEFORE 一并补齐后才可提交,见
[hpc-run-log](../hpc-run-log.md))。

## 12. 待会议确认的开放问题

§9 全部五项;另加:E1 false-conflict rate 的预注册阈值(数据集冻结时定);
是否需要一个 oracle-drop 上界系统(按 provenance 精确剔针,给门的理论天花板)。
