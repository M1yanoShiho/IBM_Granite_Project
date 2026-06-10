# Work Plan — What We Can Build Now (No GPU Required)

A practical task list for the work that needs **no GPU and no API access** — i.e.
everything we can start today on a normal laptop while GPU/HPC access is sorted
out for the generation layer.

## Why most of the project is CPU-only

- IBM confirmed **no watsonx.ai API** — Granite runs as open-source weights
  self-hosted from Hugging Face (see `meeting-1-questions.md`, Q3).
- The **primary deliverable** — a Granite-embedding **retrieval system** evaluated
  on standard IR benchmarks against BM25 and an open-source dense baseline — is
  embedding-based, not generation-based. Embedding models are small
  (`granite-embedding-278m` ≈ 278M params) and run fine on CPU.
- A **GPU is only needed for the generation layer** (RAG answers) and the
  secondary long-context "needle" test — not for retrieval or its evaluation.

So roughly **80% of the core work can be done CPU-only right now.** Dependencies
are already installed (`pip install -r requirements.lock`); the modules below
exist as documented skeletons (`raise NotImplementedError`) waiting to be filled.

---

## Tasks we can do now (CPU only)

Each maps to an existing file. Workflow is TDD: write/cure a test in `tests/`,
then implement until it passes (`pytest -q`).

- [ ] **Benchmark data loading** — `eval/benchmarks/loader.py`
      Load BEIR (start with **SciFact**) into `corpus / queries / qrels` via
      `ir-datasets` or Hugging Face `datasets`. Pure data wrangling.
- [ ] **IR metrics** — `eval/ir_metrics.py`
      Implement precision@k, recall@k, nDCG@k, MRR with `ranx`. Pure CPU compute.
- [ ] **BM25 baseline** — `src/retrieval/bm25_baseline.py`
      Tokenise corpus + score queries with `rank_bm25`. Pure Python.
- [ ] **Ingestion pipeline** — `src/ingestion/{loaders,chunker,indexer}.py`
      Load documents → chunk → build a persistent **FAISS (CPU)** index.
- [ ] **Embedder** — `src/retrieval/embedder.py`
      Wrap `granite-embedding` (the system) and `sentence-transformers` (the
      baseline). Both embed on CPU; slower than GPU but fine for small datasets.
- [ ] **Dense retriever** — `src/retrieval/retriever.py`
      Embed query → search the FAISS index → return ranked chunks.
- [ ] **Primary evaluation harness** — `eval/run_benchmark.py`
      Run all three retrievers (granite-dense / bm25 / st-dense) over the same
      queries, score with `ir_metrics`, write a comparison table to `results/`.
- [ ] **Explainability / citations** — `src/explainability/citations.py`
      Attribute results back to their source chunk/page.
- [ ] **Visualisation** — `notebooks/`
      Comparison tables and figures (matplotlib/seaborn) from `results/`.
- [ ] **Tests** — `tests/`
      Turn the `xfail` placeholders green as each function is implemented.

---

## Recommended order — a minimal loop that produces real numbers

This chain touches **no generative model** and yields our first real result
table:

1. `loader.py` — load **SciFact** (`corpus / queries / qrels`)
2. `ir_metrics.py` — metrics via `ranx`
3. `bm25_baseline.py` — BM25 retrieval
4. `run_benchmark.py` — wire them together → **"BM25 on SciFact: precision@k /
   recall@k"** table

Then **add the Granite-embedding dense retriever** (steps 5–6: `embedder.py` +
`retriever.py`) and re-run the same harness → the first data point for the core
hypothesis (Granite vs. BM25). This is exactly the **Wk 3–6** milestones in the
proposal time plan, all GPU-free.

---

## Deferred until GPU / HPC is available

- **RAG generation layer** — `src/rag_pipeline.py` calling `llm_client.generate()`
  (Granite 3B/8B generation). *Note: the retrieval half of RAG is CPU; only the
  answer-generation half needs a GPU.*
- **Secondary long-context "needle" test** — `eval/niah_runner.py`
  (feeds long contexts to a generative model).
- **RAGAS answer-quality evaluation** — uses an LLM judge.

Local-GPU note: a 6GB laptop GPU (e.g. RTX 3060 Laptop) can run the **3B model in
4-bit quantization** for development; full precision and the 8B/30B models belong
on the HPC. This requires replacing the CPU-only `torch` with a CUDA build first.

---

## How to run things

```bash
# reproduce the environment
pip install -r requirements.lock

# run the test suite (TDD loop)
python -m pytest -q

# once implemented: the primary benchmark
python -m eval.run_benchmark --dataset scifact
```
