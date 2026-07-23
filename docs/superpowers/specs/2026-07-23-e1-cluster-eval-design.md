# E1 — 边/簇检测组件评估(pool 级,B2 独立 harness)设计

状态:APPROVED(brainstorm 2026-07-23,Weikai)。落地 spec §12"边/簇检测组件评估"的 pool 级臂。

## 1. 目的与假设

E2 证明门显著降 harmful-in-context(−11.2pp),但代价是 Required Recall −4.8pp,且这两个数
都建立在"精确字符串答案簇正确刻画同意/冲突"这一未验证假设上。E1 量化这个假设在**真实 E2 池**
上的成立程度,把 E2 的因果归因(门降 harm)和代价(门伤 recall)与簇错误摘开。

**假设:** 在注入 NIAH 池上,簇的 missed-conflict rate 低(注入按设计使 gold 与 counterfactual
canonically 可分),needle-gold-recovery 高;若 missed-conflict 偏高,则门对部分注入题"失明",
E2 的 harm 收益被高估;若 needle-gold-recovery 偏低,则簇指标被抽取失败主导,须如实标注。

**统计单位 = 注入 query。** E1 是**单臂描述性评估**(刻画簇错误模式),非配对对照,故报率 + Wilson
95% 二项 CI + 分母 n,不做配对显著性检验。

## 2. 关键前提:注入选择偏差(必报框架句)

`find_injection_target`([injector.py](../../../src/evidence_rag/materializer/injector.py))只注入
gold 别名**全部 canonicalize 到单一 key** 的 query(`len(normalized) != 1 → 跳过`);`AnswerBank.select`
选 canonically 异于所有 gold 别名的 counterfactual。因此在注入集上:

- canonicalization 假冲突(gold 别名劈簇)与 value 级 missed-conflict(gold ≡ cf)**≈0 by construction**;
- 被过滤掉的正是别名难题,它们进不了 E2/E1 的池。

故 E1 报告**必须**并列一条确定性选择偏差量(见 §3 指标 5):否则 pool 级 missed/false-conflict 的
低值会被误读成"簇已验证正确",而真相是难题被上游过滤了。此条零 GPU、毫秒级、从 gold_cases.jsonl 直算。

## 3. 指标(精确定义 + 标签来源)

标签零新增人工标注,全部来自 provenance + official gold_cases(与 ArbGraph Table 4 的 200 人工对
形成方法学对照,呼应 judge-kappa≈0)。needle/counterfactual 由 **document_id 精确匹配**,无模糊文本匹配。

对每个注入 query(query_id ∈ provenance):窗口 = pool 按 retrieval_rank 排序取 top_n=20;
`extract`(Granite,passage_chars=600)→ `build_clusters(window, answers)`。needle 候选 =
document_id == needle_document_id 的候选;cf 候选同理。簇成员的 canonical 答案即 `AnswerCluster.answer`。

| # | 指标 | 定义 | 方向 | 分母 |
|---|---|---|---|---|
| 1 | **missed-conflict rate**(主) | needle 与 cf 落进**同一**簇(两者 canonical 答案相等) | lower | 两者均在窗口且均抽出 valid answer 的 query |
| 2 | **false-conflict rate** | needle 与窗口内**另一** gold-bearing 候选(文本含任一 reference_answer 别名,词边界匹配同 injector `_occurrences`)落进**不同**簇 | lower | needle **在簇** 且窗口内存在 ≥1 个**在簇的** gold-bearing other 的 query(二者均须有 valid answer,否则是 recovery 失败不计入;NIAH 单 needle 下预期小,显式报 n) |
| 3 | **needle-gold-recovery** | needle 簇的 canonical 答案 == canonicalize(gold_value) | higher | needle 在窗口的 query |
| 4 | **dedup determinism** | document_id 合票规则(同文档多 chunk = 一票)| ==100% | 纯属性单测,不过池 |
| 5 | **injection selection-bias**(框架句) | gold_cases 中 `len({canonicalize(a) for a in reference_answers}) != 1` 的比率;并列总 skip 率 =(n_gold_cases − n_provenance)/n_gold_cases | 报告值 | 全 base gold_cases,零 GPU |

指标 1–3 依赖抽取,须 GPU;指标 4 是 CPU 单测;指标 5 是 CPU 直算。指标 1–3 各报 Wilson 95% CI。

## 4. 架构(B2 独立 harness)

两个新文件 + 一个 slurm 三件套。**不碰任何冻结契约、不改门、不改 ExperimentWorkflow。**

- `src/evidence_rag/evaluation/cluster_eval.py` — **纯函数,无 I/O 无 LLM**。输入:每题的
  `{window 候选, 抽出的 answers, needle_document_id, counterfactual_document_id, gold_value,
  gold_aliases}`;调用真实 `build_clusters` 复用门的簇逻辑(不重实现);产出 per-case 指标 +
  聚合(率 + Wilson CI + n)。selection-bias 计数器亦在此(纯函数,读 canonicalize)。CPU 可测。
- `src/evidence_rag/evaluation/cluster_eval_cli.py` — I/O + 编排。读
  `candidate_sets.jsonl`(池含 text)+ `provenance.jsonl` + `gold_cases.jsonl`;建 Granite 抽取器;
  仅对注入 query 的窗口跑 `AnswerExtractionEngine.extract`;调 cluster_eval;写
  `results/e1-cluster-eval/cluster_eval_report.json`;打印 headline 到 .out。镜像
  `harm_cli.py` 的离线模式,唯一新增是抽取器(GPU)。
- 复用:`build_clusters`、`AnswerExtractionEngine`、`canonicalize_answer`、`is_valid_answer`、
  与 run_selector_gate 同一 Granite generator 工厂。

**数据流:** E2 已 dump 的 `runs/e2-gate-on/candidate_sets.jsonl`(gate-on/off 共池,任一臂皆可)
→ 过滤 provenance 里的注入 query → 重建窗口 → Granite 抽取(约 1479 题 × 20 ≈ 3 万次 generate,
与一臂 selector 同量级,~4–5h) → build_clusters → 指标 → report + .out headline。
false-conflict 的 gold 别名文本匹配用 gold_cases.jsonl 的 reference_answers(provenance 只带
gold_alias_used 单个,不够)。

## 5. 测试(TDD)

`tests/evaluation/test_cluster_eval.py`,全 CPU,用合成 window/answers/labels 直注(fake 抽取器):

- missed-conflict:needle 与 cf 同答案 → 1;异答案 → 0;任一无 valid answer → 不计入分母。
- false-conflict:构造第二个 gold-bearing 候选(文本含别名)与 needle 异簇 → 1;同簇 → 0;无第二 gold-bearing → 不计入分母。
- needle-gold-recovery:needle 抽出 gold 别名 → 1;抽出错值 → 0;needle 不在窗口 → 不计入分母。
- 别名等价:needle 抽 "JFK"、gold_value "John F. Kennedy" 且二者同 canonical → recovery=1(注入集保证同 key)。
- dedup determinism:同 document_id 多 chunk 合一票,断言 100%。
- selection-bias:多 canonical key 的 reference_answers → 计入 skip;单 key → 不计入。
- Wilson CI + 分母边界(n=0 → unscored,不为 0/0)。

## 6. HPC 三件套

- `scripts/run_cluster_eval.slurm` — 从 `run_selector_gate.slurm` **逐字**复制 env block;位置参数
  `CANDIDATES PROVENANCE GOLD_CASES OUTPUT`,带默认;跑 cluster_eval_cli 后 echo report 使 .out 带首读数;
  GPU(需 Granite):`--gres=gpu:3g.40gb:1`,account `coms039904`,partition `gpu`,qos `normal`。
- 台账:填补 `docs/hpc-run-log.md` E1 条目的 BEFORE(目的/假设/预期指标+方向/精确命令/commit),
  提交前齐备(pre-registration)。AFTER 拉回后填。
- 提交 CLI 参数前**逐一核对** cluster_eval_cli 的 argparse,不虚构 flag。

## 7. 非目标(v1 明确不做)

- 不改冻结 `SelectionResult`/`PipelineRun`/`CandidateSet` 契约,不改 `gated.py` 门逻辑;
- 不做单独的确定性审计臂(选择偏差折进 §3 指标 5,同一 report);
- 不测端到端答案质量(needle-gold-recovery 只作抽取先验,不作 generator 评估);
- 不引 NLI/语义等价(属 Graph 2.0);E1 只量现有 exact-string 簇的对错。

## 8. 待定项

1. Wilson CI 是否够,还是要 bootstrap(默认 Wilson;率的二项 CI 足够,除非评审要求配对);
2. false-conflict 分母若过小(单 needle NIAH),是否补一条"needle 是否被抽到 gold 簇"的退化 false-conflict 代理(默认先如实报小 n,不补)。
