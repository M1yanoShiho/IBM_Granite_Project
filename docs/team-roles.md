# Team Roles — Current Stage (CPU / Primary Pipeline)

Division of labour for the 7-person team for the work that needs **no GPU**
(the primary retrieval system + benchmark evaluation). See
[work-plan-no-gpu.md](work-plan-no-gpu.md) for the task detail and
[meeting-1-questions.md](meeting-1-questions.md) for open decisions.

**Why we can parallelise now:** the data-structure contracts (the `@dataclass`
definitions — `BenchmarkData`, `Chunk`, `RetrievedChunk`, …) already exist, so
everyone can code against a stable interface against stubs. **Lock these
interfaces in the kick-off meeting before splitting up.**

Workflow is TDD: each owner turns the relevant `xfail` placeholder in `tests/`
into a real assertion, then implements until green (`pytest -q`). Each owner also
writes up their own module for their individual report.

---

## Role assignments

Put your name in the "Owner" column at the kick-off. ⭐ = staff with a strong dev.

| Role | Owner | Modules | Notes / fit | Week-1 first task (unblocked) |
| --- | --- | --- | --- | --- |
| **P1 — Benchmark data loading** ⭐ | _TBD_ | `eval/benchmarks/loader.py` | Keystone — everyone's evaluation depends on it; do first | Pull SciFact, print stats, return `BenchmarkData` |
| **P2 — IR metrics** ⭐ | _TBD_ | `eval/ir_metrics.py` | Keystone, pure compute, fully independent | Implement + unit-test precision/recall/nDCG/MRR on a hand-made run+qrels |
| **P3 — BM25 baseline** | _TBD_ | `src/retrieval/bm25_baseline.py` | Most independent, pure Python — good warm-up / newcomer | Index the SciFact corpus, retrieve top-k for one query |
| **P4 — Embeddings + dense retriever** ⭐ | _TBD_ | `src/retrieval/embedder.py`, `src/retrieval/retriever.py` | The core Granite system — strongest ML person | Wrap `granite-embedding` + `sentence-transformers` backends |
| **P5 — Ingestion pipeline** | _TBD_ | `src/ingestion/{loaders,chunker,indexer}.py` | Feeds P4; coordinate the `Chunk` / FAISS-index interface with P4 | Token-aware `chunker` — testable immediately |
| **P6 — Integration + explainability** ⭐ | _TBD_ | `eval/run_benchmark.py`, `src/explainability/citations.py` | The "glue"; needs whole-flow view — good for the team lead | Define interfaces at kick-off; wire the orchestration skeleton against mock retrievers |
| **P7 — Visualisation + demo + results** | _TBD_ | `notebooks/`, `app/main.py` | Plays to the Data-Visualisation skill in the brief | Build comparison plots + Streamlit skeleton against a mock `results.csv` |

---

## Critical coordination points (agree before coding)

1. **Lock the shared contracts** — these `@dataclass`es are the hand-off points
   between people; everyone codes against them so the parts integrate:
   - `BenchmarkData` (P1) — `corpus / queries / qrels`
   - `Chunk` (P5) — `chunk_id / doc_id / text`
   - `RetrievedChunk` (P3, P4) — `doc_id / text / score`
2. **P4 ↔ P5** — the FAISS index + `Chunk` hand-off format.
3. **P1 / P2 / P6** — the scoring format: a `run` is
   `{query_id: {doc_id: score}}`, `qrels` is `{query_id: {doc_id: relevance}}`.

## Dependency / sequencing

- **P1 + P2 are the foundation** — start them first; P6 can only integrate, and
  others can only test on real data, once these exist. Staff with reliable devs.
- **P3 (BM25) is the most independent** — anyone can start immediately.
- **P4 + P5 are a tightly-coupled pair** (the dense-retrieval path) — run them in
  **parallel with the P1/P2/P3 path**, and start early as this is the most
  complex stream. Internal order within the pair:
  1. `embedder` (P4) **first** — both P5's `indexer` and P4's `retriever` need it.
  2. then P5's `chunker` (independent) + `indexer` (needs the embedder).
  3. then P4's `retriever` (needs the embedder + P5's built index).
  Day-1 unblocked starts: P4 → `embedder` standalone (embed text → check vector
  shape); P5 → `chunker` standalone (pure text splitting).
- **P6 integrates last** — but must define the interfaces at the Week-1 kick-off.

## Deferred to Phase 2 (needs GPU / HPC)

- **Generation + RAG** — `src/llm_client.py` (done) + `src/rag_pipeline.py`.
- **Secondary NIAH** — `eval/niah_runner.py`, `eval/metrics.py`.
- Exception: `src/data_processing.py` (NIAH data assembly) is **CPU-buildable** —
  a good stretch task for whoever finishes their primary task first.

## Shared by everyone

- **Tests** — turn your module's `xfail` placeholder in `tests/` green (TDD).
- **Report** — each person writes up their own component for the individual report.
