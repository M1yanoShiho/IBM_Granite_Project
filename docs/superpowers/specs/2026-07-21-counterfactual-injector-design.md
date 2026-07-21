# Counterfactual Injector (Materializer B) — 设计规格

**日期:** 2026-07-21

**状态:** 设计草稿(brainstorming 已定方向),待用户书面确认后进 writing-plans → TDD。

**模块:** Selector 评估数据构造工具;纯 CPU、确定性、无 GPU、无人工标注、无 LLM 裁判。

**承接:** [DATASET_PROPOSAL](../../selector/DATASET_PROPOSAL.md)(五条不变量配方、NQ 单域、
provenance 缺口)、gate/A2 spec(消费方 E1/E2/E3)、[benchmarks.py](../../../src/evidence_rag/infrastructure/benchmarks.py)
(输出格式 pattern)。

## 0. Scope 拆分

Materializer 拆两个可独立测试的增量:

- **A. NQ 基座加载器**(机械活,**独立 sibling spec**):dpr-w100 NQ → `documents/queries/gold_cases`
  格式 + qrels-aware 子采样到约 100k;负责 fresh-query 选择与零重叠审计。照 benchmarks.py 的
  BEIR pattern 改一个 dpr-w100 源。
- **B. 反事实注入器**(本 spec,新颖核心):吃一个加载好的 base 数据集,注入投毒孪生 + 产 provenance。

本 spec 只设计 **B**,把 base 数据集当已给输入。

## 1. 目标

给一个合格题的每一道,注入至多一根**确定性反事实针**(gold needle 的"投毒孪生":复制 gold
passage、把其中的答案值按机械类别替换掉),让检索池里出现"真值 vs 反事实"的可控冲突,
供门/覆盖层评测;并产出可复现、可反演、零人工标注的 harmful 标签。

投毒孪生与真 needle 文本近乎相同 → 检索天然共召回(bm25 下更是词汇近重复必然),
所以无需 pool 预冻结即可保证门有活干。

## 2. 输入 / 输出契约

**输入:** 一个 base `DatasetBundle`(dpr-w100 NQ 子采样;`documents` = 预切 passage,
`gold_cases` 带 `relevant_document_ids` 的 qrels + `reference_answers` 的 official 别名)。

**输出:**
- 注入后的 `documents.jsonl`(base 语料 + 反事实孪生 Documents);
- 原 `queries.jsonl` / `gold_cases.jsonl` 不变(孪生**不是** gold,gold 仍指向真 passage);
- `manifest.json`:复制自 base,不改 schema,但 `dataset_version` 追加 `+cf<seed>` 标记以
  区分注入集与基座(`dataset_signature` 由 `JsonlDatasetAdapter.load` 按注入后语料自动重算);
- `provenance.jsonl`(每注入题一条 `MutationRecord`,§7);
- 末尾用 `JsonlDatasetAdapter.load` 自校验注入后的数据集能载入(document_id 唯一等)。

## 3. 资格过滤(哪些题能注入)

复用 `answer_norm.canonicalize_answer`。保留同时满足的题:

1. **单一 gold 值:** `reference_answers` 全部规范化后落到**同一个**归一化值(多值→拒,属多答案);
2. **别名恰好一次:** 存在某条 gold passage(∈ `relevant_document_ids`),其文本含某个 gold
   surface alias **恰好一次**——该次出现即注入目标 span;
3. **可归机械类别:** gold 值能归入 integer / decimal / year-date / proper-name token 桶 /
   common-noun token 桶之一;
4. 存在同类别、≠gold 值、且不属于 gold aliases 的替换值(见 §6)。

任一不满足 → 该题不注入(记入 skip 报告,不产脏样本)。

## 4. 投毒孪生注入(五条不变量,冻结)

对每道合格题:

1. 复制目标 gold passage → 新 `Document`,`document_id = f"cf::{needle_id}"`,`source_uri`
   标注为合成(便于审计),`text` = 替换后的文本;
2. 只把那**一次** gold alias 出现替换为 §6 选出的同类别替换值;
3. **硬校验(全部必须通过,否则丢弃该注入):**
   - gold alias 在孪生文本中**零残留**(所有 alias 变体);
   - 替换值不属于 gold aliases;
   - 除目标 span 外文本逐字符不变;
   - 可由 mutation log 反向还原回原 passage;
4. 孪生作为新 Document 注入语料(不改 gold_cases,孪生非 gold)。

## 5. Answer bank(冻结、seeded)

`AnswerBank`:按机械类别组织的替换值池,seed 固定(默认 42),内容与 hash 记入每条
`MutationRecord`。替换值选择确定性(seed + 类别 + gold 值 → 稳定选一个 ≠gold、不属 aliases、
同类别的值);无同类可选值 → 该题不注入。机械类别判定复用/扩展 answer_norm 的数值与日期识别,
专名/名词按 token 长度桶。

## 6. Provenance(harmful 标签落点)

`MutationRecord`(pydantic FrozenModel,字段):`query_id`、`needle_document_id`(真 gold passage)、
`counterfactual_document_id`(注入孪生)、`gold_value`(归一化)、`gold_alias_used`(surface)、
`replacement_value`、`string_class`、`seed`、`char_span`(needle 内 start/end)、
`text_hash_before`、`text_hash_after`、`answer_bank_hash`。

写入 `provenance.jsonl`(每注入题一条)。评测 harness 读它算 harmful-in-context:
`counterfactual_document_id` 是否进了选中上下文。

## 7. 审计

- **B 负责(per-injection 不变量):** §4 的四条硬校验逐条通过;违规**停止并报错**,不产脏数据;
- **A / 预注册负责(零重叠):** fresh 查询 vs 旧 split 的 query/父页面/entity/hash/模板族零重叠
  (用 `git show` 老 manifest 做 hash 对撞)——属基座选择,不在 B;B 只在 `MutationRecord`
  里保存足够 hash 供审计复核。

## 8. 决定(已定,可推翻)

1. **Provenance 落点 = 约定式 `provenance.jsonl`**(零契约变更、无需 infra 签字、现在能建);
   manifest 1.1 声明 + 纳入 signature 留作 9/4 后硬化(DATASET_PROPOSAL §5-A)。
2. **子采样规模先 100k**(bm25 CPU 索引快、够现实),旋钮可扩 1M。
3. **第一版 E2 用现有 bm25**(BM25IndexPlugin,零 module-1 依赖);dense/q2d = module-1 增强(Track 2)。

## 9. 组件与文件

| 文件 | 内容 |
|---|---|
| `src/evidence_rag/materializer/__init__.py` | 包 |
| `src/evidence_rag/materializer/answer_bank.py` | `AnswerBank` + 机械类别判定 + 确定性选值 |
| `src/evidence_rag/materializer/injector.py` | 资格过滤 + 投毒孪生注入 + 不变量校验 |
| `src/evidence_rag/materializer/provenance.py` | `MutationRecord` 模型 + sidecar 读写 |
| `src/evidence_rag/materializer/cli.py` | `evidence-rag-materialize-niah` 入口 |

复用:`answer_norm.canonicalize_answer`、`contracts.models`(Document/GoldCase 等)、
`datasets`(DatasetBundle/JsonlDatasetAdapter/normalize_document)、benchmarks.py 的 provider 协议模式(测试用 fake)。

## 10. 测试(全确定性 TDD)

fake base `DatasetBundle` fixture;单测:资格过滤(单值/别名恰好一次/多值拒)、类别匹配替换、
零残留、可逆性、answer-bank 种子确定性、审计违规停止、注入后 `JsonlDatasetAdapter.load` round-trip、
CLI。无 GPU、无网络(provider 注入)。

## 11. 非目标(v1 明确不做)

- **不做切片/分块——工作单元是 dpr-w100 的预切 passage;** 分块属检索模块的 `Chunker`
  ([retriever/chunking.py](../../../src/evidence_rag/retriever/chunking.py)),不落选择器/Materializer
  这边(用户 2026-07-21 确认)。若将来换原始长文档/contextual retrieval(已 DEFERRED),
  切片由 module 1 出。
- 不做基座加载器 A(sibling spec);
- 不注入多答案/列表题(§3 资格过滤拒);
- 不做 pool 预冻结(检索在实验时 live 跑);
- 不做 dense 检索/module-1 索引(第一版 E2 用 bm25);
- 不用 LLM 生成反事实或当标签裁判(确定性 answer-bank + mutation log);
- v1 不改 manifest schema(约定式 sidecar;1.1 硬化 = 后续)。

## 12. 待定项

1. dpr-w100 NQ 的精确 ir_datasets id 与"queries 带 official answers"可用性(在 A 落地时核实);
2. 注入题数 / E2 power(先在 dev 量 gate-fire rate 再预注册,DATASET_PROPOSAL §6);
3. manifest 1.1 provenance_file 硬化(需 infra owner 签,9/4 后);
4. 专名/名词 token 桶的确切边界(实现细节,须单测)。
