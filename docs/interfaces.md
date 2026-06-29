# 接口契约（Interfaces / Contracts）

> 跨模块的"合同"——7 个人都对着这些写,才能并行不撞车、最后拼得上。
> **维护者：P6（毛威凯）。改任何一条都要团队同意**(见末尾变更规则)。
> 配套：[team-roles.md](team-roles.md) · [work-plan-no-gpu.md](work-plan-no-gpu.md)

代码里能落地的契约都已经落地了(`Protocol` / 类型别名 / dataclass),这份文档是
"导航 + 说明",真东西在代码里。

---

## 契约 1 — 统一检索器接口

**位置（权威）**：[src/retrieval/base.py](../src/retrieval/base.py) 的 `Retriever` Protocol +
`RetrievedChunk`。
**关系到**：P3（BM25,尤佳希/魏铭）、P4（dense,尤佳希/魏铭）、P6（消费方）。

每个检索器都必须实现:

```python
def retrieve(self, query: str) -> List[RetrievedChunk]:
    ...
```

`RetrievedChunk` 字段:`doc_id: str` / `text: str` / `score: float`（越大越相关）。

P3/P4 这样对齐(结构化 Protocol,不用显式继承):

```python
from src.retrieval.base import Retriever, RetrievedChunk

class BM25Retriever:        # 或 DenseRetriever
    def retrieve(self, query: str) -> List[RetrievedChunk]:
        ...

# 可在测试里断言实现了契约:
assert isinstance(BM25Retriever(...), Retriever)
```

> 注：`RetrievedChunk` 的正统位置是 `base.py`;`src.retrieval.retriever` 仍重新导出它,
> 所以 `from src.retrieval.retriever import RetrievedChunk` 也照样能用。**新代码请从 `base` 导入。**

---

## 契约 2 — Run 格式（检索结果 → 指标 的桥梁）

**位置（权威）**：[eval/ir_metrics.py](../eval/ir_metrics.py) 顶部的类型别名。
**关系到**：P2（指标,Frida）、P6（编排）。

```python
Run   = Dict[str, Dict[str, float]]   # {query_id: {doc_id: score}}
Qrels = Dict[str, Dict[str, int]]     # {query_id: {doc_id: relevance}}  ← benchmark 自带标注
```

- P6 把每个检索器的结果整理成一个 `Run` 交给 P2 的 `evaluate_run(run, qrels, k_values)`。
- P2 的所有指标函数都吃 `Run` + `Qrels`,返回 `{metric: value}`。

---

## 契约 3 — chunk → doc 聚合规则

**实现位置**：P6 的 [eval/run_benchmark.py](../eval/run_benchmark.py)（`build_run`）。
**关系到**：P5（吴泽楠）、P1（许展瑜）、P6。

检索器返回的是 **chunk**,但 qrels 是 **doc_id** 级。聚合规则:

> **一个 doc 的得分 = 它所有命中 chunk 的得分的最大值(max-pool)。**

SciFact 文档短,基本 1 doc = 1 chunk,这步近乎透传;长文档时才真正生效。

### 3b — chunk over-fetch + 截到 top-`max(k)` 篇 doc(长文档公平性)

dense 检索器排的是 **chunk**,但指标算的是 **doc**。长文档上一篇 doc 有多个 chunk,top-`max(k)` 个 chunk 可能塌缩成不足 `max(k)` 篇 distinct doc → recall@k 被低估,且与 doc 级直接返回 `max(k)` 篇的 BM25 不对等。规则:

> **dense 多取 chunk(`max(k) * dense_fanout`,默认 fanout=10),`build_run` 把 pool 后的 doc 截到分数最高的 `max(k)` 篇。**

这样所有检索器的 run 深度一致(都 `max(k)` 篇),@k 与 MRR 可比、不会因候选池变大而漂移。**SciFact(1 doc=1 chunk)上完全等价于旧行为,headline 数字不变。** 注意:`mean` pooling 的 doc 得分会随取的 chunk 数变化(均值在更多 chunk 上算),`max` pooling 不受影响——默认 `max`。

### 3c — chunk 单位:word(默认)/ token(opt-in)

`chunk_size`/`chunk_overlap` 默认按**空格分词的词**计;长文档(NQ)前可切到 **token 模式**(`--chunk-unit token`):按 embedding 模型自带 tokenizer 切,并把 chunk 大小**截到模型 `max_seq_length`**,避免 chunk 超长被编码时静默截断、以及 chunk-size 消融在截断点以上失真。**opt-in,默认 word,保证现有 word 模式结果可复现。**

---

## 附加约定 — embedding 归一化（P4 ↔ P5）

`indexer`（P5）建索引时若对向量做了 `faiss.normalize_L2`(用内积当余弦相似度),
那么 `retriever`（P4）编码 query 时**也必须用同样的归一化**,否则相似度算错。
两人开工前对齐一次:**统一归一化 + 内积**,还是不归一化 + L2 距离。

---

## 契约 4 — RAG 输入/输出（检索 → 生成 的桥梁）

**位置（权威）**：[src/rag_pipeline.py](../src/rag_pipeline.py) 的 `RAGResult` +
`RAGPipeline`。
**关系到**：P4/P3（提供 `Retriever`）、生成层（`LLMClient`）、P6（`eval/run_rag.py` 评测）、
P7（demo）、explainability（`citations.py`）。

RAG 是 headline 的一半(A+B,见 [meeting-1-questions.md](meeting-1-questions.md) Q1),
**不自带检索栈**——它复用契约 1 的 `Retriever`:

```python
from src.rag_pipeline import RAGPipeline, RAGResult

pipe = RAGPipeline(retriever, llm, top_k=4)   # retriever 满足契约 1
result = pipe.query("...")                    # -> RAGResult
```

`RAGResult` 字段:`answer: str` / `retrieved_chunks: List[RetrievedChunk]`
（**复用契约 1 的 `RetrievedChunk`**,带 `doc_id`/`score`,供 context-precision 打分和
citations 归因)。

评测侧(`eval/rag_metrics.py`)的 `evaluate_rag` 吃 `predictions / references /
contexts / retrieved_doc_ids / qrels`,返回 `{answer_em, answer_f1, answer_cover,
context_precision, faithfulness}`(EM/F1/cover-EM = 答案正确率,context_precision =
对 qrels 的 precision@k,faithfulness = 答案 token 被 context 覆盖率)。**答案质量评测
需要带 gold answer 的数据集**:`BenchmarkData.answers` 为可选字段
`Optional[Dict[str, List[str]]]`——**每题一组可接受答案**(列表,NQ 多别名;检索-only
集如 SciFact 为 `None`),见 Q5。

---

## 契约 5 — 向量索引搜索接口（P4 dense ↔ P5 indexer）

**实现位置**：P5 的 [src/ingestion/indexer.py](../src/ingestion/indexer.py)
（`VectorIndexer.build()` 返回、或 `load()` 加载的那个索引对象）。
**关系到**：P4（`DenseRetriever` 消费）、P5（提供）。

`DenseRetriever` 不自己存向量,而是把搜索委托给索引对象,所以那个对象必须暴露:

```python
def search(self, query_vector: List[float], top_k: int) -> List[RetrievedChunk]:
    ...
```

约定:

- 返回 `top_k` 个 `RetrievedChunk`,按相似度从高到低排好。
- `RetrievedChunk.doc_id` 必须是**父文档 id**（qrels 的 key），**不是 chunk_id**——
  `run_benchmark` 按 `doc_id` 对 qrels 打分,并按 `doc_id` 做 max-pool（契约 3）。
  `text` 是该 chunk 的文本,`score` 越大越相关。
- 相似度用**内积**:`Embedder` 输出已做 L2 归一化（见下方"附加约定"),所以内积 = 余弦,
  FAISS 用 `IndexFlatIP`。**不要用 L2 距离**（那是越小越近,语义相反）。

---

## 契约 6 — 结果表 CSV schema（P6 → P7）

**产出方**：P6 的 [eval/run_benchmark.py](../eval/run_benchmark.py)（`write_results_csv`）。
**关系到**：P6（写）、P7（`app/main.py`、`notebooks/` 读）。

`results/benchmark_results.csv` —— 每个 retriever 一行:

| 列 | 含义 |
| --- | --- |
| `retriever` | 检索器名:`granite_dense` / `bm25` / `st_dense`（**第一列,固定叫 `retriever`,不是 `model`**） |
| `precision@k` `recall@k` `ndcg@k` | 各 k 的指标,k 由 `BenchmarkConfig.k_values` 决定（默认 1/3/5/10） |
| `mrr` | 平均倒数排名 |

- 列名权威以 `write_results_csv` 为准:分组键是 `retriever`。
- P7 画图按 `retriever` 分组,headline 取 `precision@10` / `recall@10` / `ndcg@10` / `mrr`。

### 6b — ablation/sweep 模式的扩展列

默认单次 run 的 schema 不变（上表）。当用 `--append`（sweep 模式）跑时,`run` 会在每行前面加上本次配置列,然后才是 `retriever` 和指标列,多次 run 追加进同一个 CSV:

| 列 | 含义 |
| --- | --- |
| `dataset` | 数据集名 |
| `chunk_size` `chunk_overlap` | 本次切分参数 |
| `pooling` | `max` / `mean` |
| `embedding_model_id` | dense 模型覆盖值,未覆盖时为 `default` |
| `retriever` + 指标列 | 同上表 |

- sweep 用法:循环换 config 反复调 `run(...)`（或 CLI 反复跑 `--append`),全部追加进一张主 CSV。**跑新 sweep 前先删掉旧的主 CSV**,否则会接在旧数据后面。
- P7 读 ablation CSV 时,按这些配置列分组/筛选(例如固定 `retriever=granite_dense`,看 `chunk_size` 对 `ndcg@10` 的影响)。

---

## 变更规则

1. 这些契约定了之后,**不要私自改** —— 改一条会同时影响多个人。
2. 确需修改:群里提 → 相关负责人确认 → P6 改这份文档 + 对应代码 → 通知全员。
3. 每个人开工前,在群里回一句"按 interfaces.md 确认",作为轻量签字。
