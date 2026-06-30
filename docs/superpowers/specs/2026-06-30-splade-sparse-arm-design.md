# Design — SPLADE learned-sparse arm + convex hybrid (Phase 2)

- Date: 2026-06-30
- Branch context: `week4_SPLADE`
- Status: design approved, spec self-reviewed → inline TDD execution (Tasks 1–2 this session)
- Owner: P6 (Weikai Mao)
- Continues: [Phase 1 convex-hybrid spec](2026-06-29-convex-hybrid-fusion-design.md)

## 1. Background / problem

Phase 1 (convex hybrid, BM25 arm) found that a *weak* `rank_bm25` lexical arm, fused
correctly (per-query min-max convex combination, α tuned on dev), gives a small but
**significant** gain over pure `granite_dense` on the harder sets (NFCorpus +0.012
p=0.001, FiQA +0.008 p=0.0015; SciFact tie), and beats RRF everywhere. The tuned α was
high (0.75–0.90), i.e. BM25 carries only 0.10–0.25 weight.

The open question Phase 1 set up: **does a *strong* learned-sparse arm (SPLADE) help
more than weak BM25** — a bigger gain over dense, and/or a lower optimal α (more
lexical weight productively used)? Phase 2 answers it by swapping the lexical arm
BM25 → SPLADE and re-running the same experiment, filling the SPLADE column of the 2×2.

## 2. Goal & scope

**Goal.** Add a SPLADE retriever (encoder + sparse index + `Retriever`) and run the
Phase-1 convex-hybrid experiment with SPLADE as the lexical arm.

**In scope:** `SpladeEncoder` (transformers MLM + the SPLADE pooling recipe), a
`scipy.sparse` CSR `SparseIndex` + search, a `SparseRetriever`, a one-line `--lexical`
generalisation of `tune_alpha`, `run_benchmark` wiring (`splade` component +
`convex_hybrid_granite_splade` + a sparse-index cache), a slurm runner, the run on the
3 BEIR sets.

**Out of scope:** any change to the dense or RAG sides; efficient-SPLADE / asymmetric
query-vs-doc encoders; scaling beyond the 3 BEIR sets; the lexical-favourable stretch
set (still optional, as in Phase 1).

## 3. Locked decisions

| Decision | Choice | Rationale |
|---|---|---|
| Sparse search | **`scipy.sparse` CSR** `[N_docs × vocab]`, search = sparse matvec → top-k | Compact memory (~90 MB FiQA vs ~1 GB for a Python inverted index), fast, standard, fully TDD-able with hand-built matrices. +1 trivial pure-wheel dep. |
| Encoder | **`transformers.AutoModelForMaskedLM`** + the SPLADE pooling recipe | Portable across the HPC's transformers-4.x and local 5.x; the canonical SPLADE inference, unlike ST-version-specific APIs. |
| Model | **`naver/splade-cocondenser-ensembledistil`** (parametrized, env override) | Ungated → `hf download` works on the offline HPC login node (`splade-v3` is gated and would block the prefetch); established, strong zero-shot on BEIR. License CC-BY-NC-SA = fine for academic eval; the *delivered system* stays Granite/Apache. |
| SPLADE granularity | **Doc-level** (corpus + doc_ids, truncate >512 tokens), like `BM25Retriever` | Drop-in replacement for the BM25 arm; the convex hybrid's doc-level fusion is unchanged. Long-doc truncation noted as a limitation (fine for the short-ish BEIR sets). |
| Fusion / eval | **Reuse Phase 1** (`convex_fuse`, `ConvexHybridRetriever`, `tune_alpha`, significance, 2×2) | Phase 1's machinery is arm-agnostic (operates on `Run`s). Phase 2 is "add a `Retriever` arm". |

## 4. Architecture (5 pieces — 4 new, 1 wiring)

All respect the existing contracts: `Retriever.retrieve(query) -> List[RetrievedChunk]`
(contract 1), `Run` (contract 2), chunk→doc max-pool (contract 3). Mirrors the dense
stack: `Embedder`→`SpladeEncoder`, `FaissIndex`→`SparseIndex`, `DenseRetriever`→
`SparseRetriever`.

A SPLADE vector is `Dict[int, float]` (term_id → weight) — a **distinct type from
Phase 1's `fusion.Scores`** (`doc_id:str → score`); the convex fusion still operates on
doc-level `Run`s, unchanged. The encoder names it `TermWeights`; the generic index
names its own `SparseVector` (same shape, no import coupling).

### 4.1 `src/retrieval/splade_encoder.py` (NEW) — Task 1

```python
TermWeights = Dict[int, float]   # {term_id: weight}

def splade_pool(logits, attention_mask):   # pure: [B,S,V],[B,S] -> [B,V]
    # SPLADE: w_j = max_i ( log(1 + relu(logits_ij)) * mask_i )
    ...

class SpladeEncoder:
    def __init__(self, model_id=None, *, model=None, tokenizer=None): ...
    def encode(self, texts: Sequence[str]) -> List[TermWeights]: ...
```

- `splade_pool` is a **pure torch function** — the primary TDD unit (max over sequence of
  `log1p(relu(logits))`, padding masked to 0).
- `SpladeEncoder.encode`: tokenize (padding+truncation) → `model(**inputs).logits` →
  `splade_pool` → per row, `{int(term): float(w) for nonzero w}`. `model`/`tokenizer`
  are injectable (a `FakeSpladeModel`/`FakeTokenizer` for tests, like `Embedder`'s
  `FakeSentenceTransformer`); otherwise lazy-load `AutoModelForMaskedLM`/`AutoTokenizer`
  (honouring `MODEL_CACHE_DIR`, `SPLADE_MODEL_ID`). Default
  `naver/splade-cocondenser-ensembledistil`.

### 4.2 `src/retrieval/sparse_index.py` (NEW) — Task 2

```python
SparseVector = Dict[int, float]

class SparseIndex:
    def __init__(self, matrix, doc_ids, vocab_size): ...
    @classmethod
    def build(cls, doc_vectors: Sequence[SparseVector], doc_ids, vocab_size) -> "SparseIndex": ...
    def search(self, query: SparseVector, top_k: int) -> List[Tuple[int, float]]: ...
```

- `build`: list of `{term:weight}` → `scipy.sparse.csr_matrix [N_docs × vocab]` (via COO
  rows/cols/data).
- `search`: query → sparse `[1 × vocab]`; `scores = (matrix @ query.T)` → dense `[N_docs]`;
  return up to `top_k` docs **with score > 0** by **stable** descending sort (ties broken
  by ascending index, deterministic). Empty query, or no doc shares a term → `[]`.
  Filtering zeros (rather than padding with score-0 docs) matters downstream: a SPLADE
  arm that finds nothing for a query then contributes `{}` to the convex fusion → that
  query defers to the dense arm, instead of an all-zero set that per-query min-max would
  wrongly map to all-1.0.
- Pure sparse-math, no model — TDD-able with hand-built vectors.

### 4.3 `src/retrieval/sparse_retriever.py` (NEW) — Task 3

```python
class SparseRetriever:   # satisfies Retriever
    def __init__(self, encoder: SpladeEncoder, index: SparseIndex,
                 doc_ids, texts, top_k=10): ...
    def retrieve(self, query: str) -> List[RetrievedChunk]: ...
```

- `retrieve`: `encoder.encode([query])[0]` → `index.search(qv, top_k)` →
  `RetrievedChunk(doc_id, text, score)`. Doc-level (the index rows are documents), so
  `doc_id` is the parent document id straight away (contract 5). Mirrors `DenseRetriever`.

### 4.4 `eval/tune_alpha.py` (EDIT) — Task 4

- Add `--lexical` (default `bm25`, accepts `splade`); `_arm_runs` builds the lexical arm
  by that name via `_build_component` instead of hard-coding `"bm25"`. Everything else
  (sweep, curve, `best_alpha`) unchanged. Tests: `_arm_runs` is integration (HPC); add a
  parse test for `--lexical`.

### 4.5 `eval/run_benchmark.py` (EDIT) — Task 5

- `_build_component`: add a `"splade"` branch building a `SparseRetriever` over the
  corpus (encode docs → `SparseIndex.build` → wrap), with an **optional sparse-index
  cache** (pickle the CSR + doc_ids, keyed like `_cache_key` but for sparse) so encoding
  isn't repeated across the standalone + hybrid runs.
- `CONVEX_HYBRID_SPECS["convex_hybrid_granite_splade"] = ["granite_dense", "splade"]`.
- Tests: `_build_component("splade", …)` returns a `SparseRetriever` (fake encoder via
  monkeypatch); `convex_hybrid_granite_splade` builds a `ConvexHybridRetriever`.

### 4.6 `scripts/run_splade_hybrid.slurm` (NEW) — Task 6

Prefetch note for `naver/splade-cocondenser-ensembledistil` on the login node, then per
dataset: `tune_alpha --lexical splade` (curve + best α) then `run_benchmark` with
`granite_dense splade convex_hybrid_granite_bm25 convex_hybrid_granite_splade` +
`--per-query-out`. Same SBATCH header as `run_convex_hybrid.slurm`.

## 5. The SPLADE pooling recipe (Task 1 detail)

For a tokenized batch, the model gives logits `[B, S, V]`. SPLADE term weights:
`w_{b,j} = max over positions i of ( log(1 + ReLU(logits_{b,i,j})) * mask_{b,i} )`,
yielding `[B, V]`; keep non-zero `j` per row. `log1p(relu(·))` is monotone non-negative;
the `max` over the sequence is the SPLADE-max pooling; the mask zeroes padding so it
never contributes. (No FLOPS/L1 regularisation at inference — that's a training-time
term only.)

## 6. Experiment protocol

Same as Phase 1: SciFact / NFCorpus / FiQA; tune α on dev/train, publish the test
α-curve + headline at the tuned α; significance (paired randomization) for
convex-splade vs dense, vs RRF, and **vs convex-bm25** (the Phase-1 hybrid, included in
the same run for a direct weak-vs-strong-arm comparison). New retrievers: `splade`
(standalone) and `convex_hybrid_granite_splade`. Reuses `eval/significance.py` and the
α=1 consistency check (α=1 = pure dense, must reproduce published `granite_dense`).

## 7. TDD plan (tests first; full suite stays green — was 250 passed / 1 xfailed)

- `tests/test_splade_encoder.py` (NEW): `splade_pool` (max over seq; `log1p(relu)`;
  padding masked to 0; a hand-checked tiny `[1,2,V]` case); `SpladeEncoder.encode` with a
  `FakeSpladeModel`+`FakeTokenizer` → expected `{term:weight}` (incl. non-zero filtering).
- `tests/test_sparse_index.py` (NEW): `build` shape; `search` orders by dot product on a
  hand-built corpus; empty query → `[]`; query terms absent from all docs → `[]` (zeros
  filtered); `top_k` respected; deterministic tie-break.
- `tests/test_sparse_retriever.py` (NEW, Task 3): `isinstance Retriever`; with a fake
  encoder+index, `retrieve` returns `RetrievedChunk`s ranked by score, carries doc text,
  respects `top_k`.
- `tests/test_tune_alpha.py` (EDIT, Task 4): parse `--lexical`.
- `tests/test_run_benchmark.py` (EDIT, Task 5): `splade` component + convex-splade wiring
  (fake encoder).

## 8. Implementation tasks

1. **`SpladeEncoder` + `splade_pool`** (`src/retrieval/splade_encoder.py`) + tests. ← this session
2. **`SparseIndex`** (scipy CSR build + search) (`src/retrieval/sparse_index.py`) + tests. ← this session
3. **`SparseRetriever`** (`src/retrieval/sparse_retriever.py`) + tests.
4. **`tune_alpha --lexical`** generalisation + parse test.
5. **`run_benchmark` wiring** (`splade` component + `convex_hybrid_granite_splade` + sparse cache) + tests.
6. **`scripts/run_splade_hybrid.slurm`**.
7. **HPC run + record** the SPLADE column of the 2×2 in `docs/results-summary.md`.

## 9. Dependencies / license

- Add **`scipy`** to `requirements.txt` (pinned; likely already present transitively via
  scikit-learn/sentence-transformers — verify and pin the installed version).
- `naver/splade-cocondenser-ensembledistil` is **CC-BY-NC-SA-4.0** (non-commercial) —
  acceptable for academic evaluation; documented as a baseline/component, not the
  delivered system (Granite, Apache-2.0).

## 10. Risks / caveats

- SPLADE encoding is a BERT forward pass per doc — heavier than BM25; needs the HPC/GPU
  for FiQA (57k docs). Cache the CSR so it's encoded once.
- Doc-level truncation at 512 tokens under-represents long documents (fine for these
  sets; flag for any long-doc set).
- Expectation is *not* asserted: SPLADE may or may not beat BM25 in the hybrid — either
  is a clean result (stronger arm helps more, or the dense arm already dominates so even
  a strong sparse arm adds little). The α-curve + the convex-splade-vs-convex-bm25
  significance is the finding.

## 11. Definition of done (Phase 2)

- New unit tests pass; full suite green.
- `splade` + `convex_hybrid_granite_splade` run end-to-end on the 3 sets; α=1 consistency
  check holds; significance computed (vs dense, vs RRF, vs convex-bm25).
- `docs/results-summary.md` finding #4 / the 2×2 SPLADE column filled.
