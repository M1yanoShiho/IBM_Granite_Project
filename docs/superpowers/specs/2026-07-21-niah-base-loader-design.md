# NIAH Base Loader (Materializer A) — 设计规格

**日期:** 2026-07-21

**状态:** 设计已呈+口头批准,直接落 spec → TDD。

**归属:** **数据基建 tooling,不是选择器工作**(用户明确只做选择器;此件由 Claude 当 tooling 建,
解注入器 B 的输入依赖)。纯 CPU、确定性、provider 可注入(无网络依赖于测试)。

**承接:** [benchmarks.py](../../../src/evidence_rag/infrastructure/benchmarks.py) 的 BEIR 物化 pattern
(A 几乎是它的兄弟)、注入器 B([2026-07-21-counterfactual-injector-design](2026-07-21-counterfactual-injector-design.md))
的输入契约、ir_datasets `dpr-w100/natural-questions` schema(已 web 核实)。

## 1. 目标

把 `dpr-w100/natural-questions/dev` 物化成本项目的 `documents/queries/gold_cases + manifest`
格式,并做 **qrels-aware 子采样**(21M 全量不现实 → ~100k),产出注入器 B 能直接吃的 base 数据集。

## 2. ir_datasets schema(已核实,A 可行的根据)

- `DprW100Query`:`query_id: str`、`text: str`、**`answers: Tuple[str]`**(短答案别名,直接进 reference_answers);
- `TrecQrel`:`query_id`、`doc_id`、`relevance: int`、`iteration`(relevance>0 = gold passage);
- `DprW100Doc`:`doc_id`、`text`、`title`(passage,**预切**,无需分块);
- 语料 21M passage。

## 3. 输入 / 输出

**输入:** 一个 provider(镜像 benchmarks.py 的 `BenchmarkProvider`/`BenchmarkDataset` 协议:
`docs_iter()`/`queries_iter()`/`qrels_iter()`)。生产走 ir_datasets;测试注入 fake provider。

**输出(JsonlDatasetAdapter 格式):**
- `documents.jsonl` = 子采样后的语料(所有 gold passage + 采样 distractor);
- `queries.jsonl`(选中 query);
- `gold_cases.jsonl`(`relevant_document_ids` ← qrels rel>0;`reference_answers` ← query.answers);
- `manifest.json`(dataset_id `niah/dpr-w100-nq`、version、split);
- 末尾 `JsonlDatasetAdapter.load` 自校验。

## 4. qrels-aware 子采样(A 相对 benchmarks.py 的唯一新增)

1. 选 query(可 `--query-limit N` 取排序后前 N;fresh-query 审计见 §8);
2. `gold_docs` = 选中 query 的所有正 qrels doc_id;
3. `distractors` = 从其余 doc 里**确定性**(seed)采样,补到 `--corpus-size`(默认 100000);
   gold_docs 全保留,不被采样挤出;
4. `documents` = gold_docs ∪ distractors 对应的 `DprW100Doc`(title+text 合并成 text)。

**内存注意:** 不把 21M docs 全读进内存 —— 先扫 qrels 定 gold set,再流式扫 docs:gold 必留,
非 gold 用蓄水池/seed 采样到预算。O(corpus_size) 内存。

## 5. 字段映射(复用现有归一化)

- doc:`text = "\n\n".join([title, text] 非空项)`;`source_uri = "ir-datasets://dpr-w100/nq/document/<doc_id>"`;`normalize_document`;
- query:`normalize_query`;`reference_answers = tuple(sorted(set(query.answers)))`(空则 None);
- gold_case:`relevant_document_ids = tuple(sorted(正 qrels doc))`;每 query 至少一条正 qrels,否则丢弃该 query。

## 6. 组件与文件

| 文件 | 内容 |
|---|---|
| `src/evidence_rag/materializer/base_loader.py` | provider 协议(镜像 benchmarks)+ `materialize_niah_base(...)` + 子采样 |
| `src/evidence_rag/materializer/base_cli.py` | `evidence-rag-load-niah-base`(生产用 ir_datasets provider) |
| `pyproject.toml` | 注册 CLI |
| 测试 | `tests/materializer/test_base_loader.py`(fake provider,确定性) |

## 7. 测试(全确定性 TDD)

fake provider(几条 doc/query/qrel):字段映射、qrels rel>0 才算 gold、gold 全保留、
distractor seed 确定性采样到预算、无正 qrels 的 query 丢弃、`JsonlDatasetAdapter.load` round-trip、
CLI(注入 fake provider)。无 GPU、无网络。

## 8. 非目标 / 待定

- **不做切片**——dpr-w100 预切,分块归检索器 `Chunker`([retriever/chunking.py](../../../src/evidence_rag/retriever/chunking.py) 已有 PrechunkedChunker);
- 不做反事实注入(那是 B);A 只产干净 base;
- **fresh-query 零重叠审计**(与旧 contaminated q-set):v1 先不做,第一版 dev 结果不卡它;
  最终确认性 run 前再加(旧 manifest 在别的分支,`git show` 取来 hash 对撞);
- **Ops(不是代码):** 真实物化要在**登录节点**先 `ir_datasets` 下载 dpr-w100 NQ(21M passage,大);
  compute 节点离线。A 的代码用 fake provider 现在就能 TDD,实际跑 = 一次登录节点数据准备。
