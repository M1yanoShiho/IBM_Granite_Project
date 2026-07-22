# Harmful-in-Context Metric + E2/Refinement — 设计规格

**日期:** 2026-07-21

**状态:** 设计已过用户口头评审,直接落 spec → TDD。

**归属:** 选择器自有的评测件——量"门到底有没有把毒挡在生成上下文外",是选择器成功的定义性指标。
纯 CPU、离线、确定性。**A(基座数据集)是 E2 的外部依赖,不属本工作。**

**承接:** 注入器 B 的 `provenance.jsonl`([materializer/provenance.py](../../../src/evidence_rag/materializer/provenance.py))、
门/覆盖层(gated.py)、评测模块([evaluation/scoring.py](../../../src/evidence_rag/evaluation/scoring.py) +
[stage_evaluators.py](../../../src/evidence_rag/evaluation/stage_evaluators.py))。

## 1. 指标定义

对每道注入了反事实的题 q,设孪生 doc = `cf::q::needle`。

- **per-query:** `harmful_in_context(q) = 1.0` 若孪生 doc ∈ 选中证据集的 document_ids(喂给生成器的 top-k);否则 `0.0`;
- **未注入的题:** `None`(不计分)——没毒不可能被毒,只有注入题进分母;
- **聚合:** `Harmful Rate = 注入题里中毒进上下文的比例`,方向 **lower**。

## 2. 住哪与注册(最小侵入)

- 新文件 `src/evidence_rag/evaluation/harm.py`,产标准 `StageEvaluationReport`(selector 阶段),
  与现有 `evaluate_selector_stage` 并列,**不改**那个共享函数;
- 在 [scoring.py](../../../src/evidence_rag/evaluation/scoring.py) 注册 metric key
  `selector.core.harmful_in_context` → 方向 `lower`;`CORE_METRIC_VERSION` 从 `1.1` 升到 `1.2`
  (注册表变了);
- 复用现有 `aggregate_metrics`、`document_ids`、`unscored`、`signature`、`MetricValue`、
  `StageCaseEvaluation`、`StageEvaluationReport`,不重造。

## 3. provenance 来源(关键:离线,不碰 workflow)

harmful 标签在 B 的 `provenance.jsonl`,**不在** GoldCase(当初为此没塞进冻结契约)。所以 harm
指标**离线**算:读实验 dump 的 `selected_evidence_sets.jsonl` + 数据集旁 `provenance.jsonl` →
逐题判孪生有没有进选中集。**完全不改共享的 ExperimentWorkflow**,纯选择器侧后处理。

`provenance_harm_map(records) -> dict[str, str]` 从 `MutationRecord` 序列建 query_id → cf_doc_id。

## 4. guardrail(已存在,直接读)

Required Recall = 现有 `selector.core.conditional_document_recall`(gold doc 在不在选中集,条件于被检索到)
——workflow 跑选择器阶段时已算。E2 判定读它,不新写。

## 5. 诚实报告项:pool-hit rate

门只能作用于**进了候选池**的毒。`counterfactual_pool_hit_rate(candidate_sets, harm_map) -> float | None`
= 注入题里孪生进候选池的比例。E2 必须同时报它:若 pool-hit 很低,Harmful Rate 低不代表门强,
是检索没把毒捞上来。分开报,避免把 null 误读成"门很强"。诊断量,不进 metric 注册表。

## 6. 配对比较(E2 的统计)

本分支无现成 significance 模块(旧的在 week4_SPLADE),故自带一个小的、自足的:

`compare_harm(on_report, off_report, *, seed, iterations) -> HarmComparison`:
- 按 query_id 对齐两臂,取两臂都 scored(即注入)的题;
- `delta = mean(on) − mean(off)`(负 = 门降低 harm);
- **配对随机化检验**:每题以 0.5 概率交换 on/off,重算 delta,`p = 频率(|置换 delta| ≥ |观测|)`;
- **bootstrap CI**:重采样题(query-level)算 delta 的百分位 CI;
- `HarmComparison(delta, p_value, ci_low, ci_high, n_paired)`,frozen model。
- 按父页面分组的 grouped bootstrap 列为精校项(需语料 title 映射,见 §10)。

## 7. E2 实验方案(门挡不挡毒)

- **对照:** gate-on(`gated-corroboration`)vs gate-off(`corroboration`,同 α、只差踢不踢);
  **同检索器同池**(bm25 确定性)→ 两臂唯一变量是门 → Harmful 之差 = 门净因果;
- **判定(spec 双 Gate):** 主 = Harmful Rate 显著 ↓(`compare_harm` 的 delta<0 且 p 显著);
  guardrail = Required Recall 非劣下界 −0.01;两者都过才算门有效;
- **机制:** 两个 config TOML(仅 `[selector]` 不同)→ 现有 workflow 跑两遍 dump
  `selected_evidence_sets.jsonl` → 离线 `evidence-rag-harm-report`(新 CLI)算两臂 Harmful +
  pool-hit + `compare_harm` → 一张表。**必须上 HPC**(选择器抽答案要 Granite);
  `scripts/run_selector_gate.slurm` 已是两臂,够用;
- 覆盖层是**另一个**实验(E3),不在 E2。

## 8. 精校方案(把门调到最优)

1. **dev 先量地形(不看 test):** gate-fire rate + pool-hit rate;pool-hit 太低先回去问检索(bm25 vs dense);
2. **扫 operating point:** margin∈{1,2,3} × support_cap∈{1,2,off},每组一个点 →
   Harmful↓ vs Recall↓ 权衡曲线 → 选"Recall 不破 −0.01 下 Harmful 最低"的点,dev 冻结;
   默认 margin=2/cap=1 是起点不是终点;
3. **免校准验证:** 若拿到第二个域,dev 冻结的 margin/cap **原样搬**过去跑——验"相对判据抗跨域漂移"
   (ContractNLI −0.134 的正面反证);
4. **覆盖层 E3(独立,排 E2 之后):** coverage-on(`gated-coverage-corroboration`)vs off,
   主 grounded-answer F1/faithfulness,需多支撑题;
5. **预注册纪律:** dev 扫完冻结 margin/cap/coverage 开关 → test 一次性,不回调。

## 9. 组件与文件

| 文件 | 内容 |
|---|---|
| `src/evidence_rag/evaluation/harm.py` | `harmful_in_context` / `evaluate_selector_harm` / `provenance_harm_map` / `counterfactual_pool_hit_rate` / `HarmComparison` / `compare_harm` |
| `src/evidence_rag/evaluation/scoring.py` | 注册 `selector.core.harmful_in_context`(lower)+ `CORE_METRIC_VERSION` 1.1→1.2 |
| `src/evidence_rag/evaluation/harm_cli.py` | `evidence-rag-harm-report`:读两臂 selected_sets + provenance → Harmful/pool-hit/compare 表 |
| `pyproject.toml` | 注册 CLI 脚本 |
| 测试 | `tests/evaluation/test_harm.py`、`tests/evaluation/test_harm_cli.py` |

## 10. 测试(全确定性 TDD)

fake `SelectedEvidenceSet` / `CandidateSet` / `MutationRecord` fixture:
per-query harm(毒在/不在选中集)、未注入题为 None、Harmful Rate 聚合、pool-hit、
`compare_harm`(已知 delta/p 的构造样例 + seed 确定性)、报告 round-trip、CLI 端到端(读文件出表)。
无 GPU、无网络。

## 11. 非目标(v1 明确不做)

- 不改 ExperimentWorkflow / GoldCase / 冻结契约(离线后处理);
- 不做 A 基座加载器(外部依赖);
- 不做覆盖层 E3 的下游 F1 评测(排 E2 之后,另立);
- grouped-by-parent-page bootstrap 先不做(query-level 起步,§8 精校项);
- 不引入外部统计库(自带轻量随机化/bootstrap)。

## 12. 待定项

1. 父页面分组的 grouped bootstrap(需 dpr-w100 title→group 映射,精校期加);
2. E2 的注入题数 / power(dev 量 gate-fire 后预注册);
3. gate-off 基线到底用 `corroboration` 还是 `top-k`(建议 `corroboration` 同 α,隔离"踢"的净效应;`top-k` 作次级对照)。
