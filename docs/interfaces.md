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

---

## 附加约定 — embedding 归一化（P4 ↔ P5）

`indexer`（P5）建索引时若对向量做了 `faiss.normalize_L2`(用内积当余弦相似度),
那么 `retriever`（P4）编码 query 时**也必须用同样的归一化**,否则相似度算错。
两人开工前对齐一次:**统一归一化 + 内积**,还是不归一化 + L2 距离。

---

## 变更规则

1. 这些契约定了之后,**不要私自改** —— 改一条会同时影响多个人。
2. 确需修改:群里提 → 相关负责人确认 → P6 改这份文档 + 对应代码 → 通知全员。
3. 每个人开工前,在群里回一句"按 interfaces.md 确认",作为轻量签字。
