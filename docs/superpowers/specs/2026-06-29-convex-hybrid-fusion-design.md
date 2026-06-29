# Design — Convex-combination hybrid retrieval (Phase 1: dense + BM25)

- Date: 2026-06-29
- Branch context: `week3` (integration)
- Status: design approved, pending spec review → implementation plan
- Owner: P6 (Weikai Mao)

## 1. Background / problem

The fair retrieval audit found that **the RRF hybrid (dense + BM25) is significantly
worse than pure `granite_dense` on every BEIR set tested** (SciFact / NFCorpus /
FiQA; see [results-summary.md](../../results-summary.md) finding #4). The current
[`HybridRetriever`](../../../src/retrieval/hybrid.py) fuses by **rank** (Reciprocal
Rank Fusion), deliberately discarding each arm's scores, with a fixed equal weight.

Two confounded weaknesses explain that negative result:

1. **Fusion method.** Equal-weight RRF ignores score magnitude and regresses toward
   the much weaker BM25 arm. The literature is clear that **convex combination of
   normalized scores beats RRF** and is more sample-efficient (Bruch et al., 2023).
2. **Lexical-arm quality.** The lexical arm is pure-Python `rank_bm25`, weaker than
   Anserini BM25 and far weaker than a learned-sparse encoder (SPLADE).

These are **separable levers**. This spec covers **Phase 1 only: fix the fusion
method (RRF → convex combination) while holding the lexical arm at BM25.** SPLADE
(lever 2) is Phase 2 and explicitly out of scope here, so the two effects never
confound.

Phase 1 outcome is a win either way: either a properly-fused hybrid finally beats
pure dense (a positive result), or it produces a **clean, well-explained negative**
("even per-query min-max convex fusion, α-tuned on dev, cannot beat the dense first
stage — the α-curve peaks at α→1; the dense arm dominates"). Both are
publishable-grade and cite the convex>RRF result, unlike the current "we used
equal-weight RRF and it lost".

## 2. Goal & scope

**Goal.** Add a convex-combination hybrid retriever and the tooling to α-tune it
without test leakage, re-run the hybrid experiment on the sets where RRF failed, and
report whether convex fusion changes the conclusion.

**In scope (Phase 1):**
- A pure fusion module (normalize + convex-combine two `Run`s).
- A `ConvexHybridRetriever` satisfying the existing `Retriever` contract.
- An offline α-tuning / sweep harness.
- Wiring into `eval/run_benchmark.py` so the tuned hybrid lands in the standard
  results + per-query + significance pipeline.
- Runs on SciFact / NFCorpus / FiQA (+ one lexical-favourable stretch set).

**Out of scope (deferred / unchanged):**
- SPLADE arm and any sparse-index infrastructure (Phase 2).
- Any change to the dense retriever, the RAG pipeline, or the cross-encoder reranker.
- Removing the RRF `HybridRetriever` — it stays, as the RRF cell of the final 2×2.

## 3. Locked decisions

| Decision | Choice | Rationale |
|---|---|---|
| Scope | **Stage it**: convex+BM25 now, SPLADE later | Risk-sequencing: convex+BM25 reuses 100% of existing infra (~1–2 days, no unknowns); SPLADE is the only piece with real infra risk. Phase 1's α-curve is also the go/no-go signal for SPLADE. The final report still gets the full 2×2 — built in two passes at ~zero extra artifact cost. |
| α selection | **Dev-split tune + publish the test α-curve** | No test leakage. The α-curve (nDCG@10 vs α on test) is the real scientific artifact — it answers "does ANY α beat pure dense?". The dev-chosen α is the honest deploy point. |
| Normalization | **Per-query min-max** to [0,1] | The scheme used by the convex-combination paper; simple, robust enough at these corpus sizes. TMM is a noted fallback. |
| Tuning module | **Separate `eval/tune_alpha.py`** | Keeps the sweep a focused, independently-testable unit; the alternative (folding a sweep mode into `run_benchmark`) bloats an already-large file. |
| Fusion math location | **One pure function**, reused by both the online retriever and the offline sweep | Single source of truth for the fusion math; no drift between "what we tuned" and "what we deploy". |

## 4. Architecture (4 pieces — 3 new, 1 wiring)

All pieces respect the existing contracts in [interfaces.md](../../interfaces.md):
`Retriever.retrieve(query) -> List[RetrievedChunk]` (contract 1), `Run =
{qid: {doc_id: score}}` (contract 2), chunk→doc **max-pool** (contract 3).

### 4.1 `src/retrieval/fusion.py` (new) — pure fusion math

```python
Scores = Dict[str, float]   # {doc_id: score}, one query

def minmax_normalize(scores: Scores) -> Scores: ...
def fuse_one(dense: Scores, lexical: Scores, alpha: float,
             normalize=minmax_normalize) -> Scores: ...
def convex_fuse(dense_run: Run, lexical_run: Run, alpha: float,
                normalize=minmax_normalize) -> Run: ...
```

- `minmax_normalize`: maps each query's arm scores to [0,1] via `(s - min)/(max - min)`.
  Edge cases: empty → `{}`; `max == min` (single doc or all-equal) → all map to **1.0**
  (a retrieved doc with no spread is treated as maximally relevant by that arm).
- `fuse_one`: normalize each arm, then for the **union** of doc_ids,
  `fused = alpha * nd.get(doc, 0.0) + (1 - alpha) * nl.get(doc, 0.0)`. A doc absent
  from an arm's pool contributes 0 for that arm. `alpha` is the **dense** weight;
  validated to `[0, 1]`.
- `convex_fuse`: loop `fuse_one` over the union of query ids → a fused `Run`. This is
  what the offline sweep calls.

### 4.2 `ConvexHybridRetriever` in `src/retrieval/hybrid.py` (new, alongside `HybridRetriever`)

```python
class ConvexHybridRetriever:
    def __init__(self, dense: Retriever, lexical: Retriever, alpha: float,
                 top_k: int = 10) -> None: ...
    def retrieve(self, query: str) -> List[RetrievedChunk]: ...
```

- `retrieve`: call each arm; **max-pool each arm's chunks to doc level** (contract 3)
  into a `Scores` dict, remembering each doc's text; call `fuse_one(dense_scores,
  lexical_scores, alpha)`; sort fused desc; return the top-`top_k` as
  `RetrievedChunk(doc_id, text, fused_score)` (text preferring the dense chunk,
  falling back to lexical). One chunk per doc, so the downstream `build_run` max-pool
  is a pass-through.
- **The arms must already return a deep candidate pool** (≈100 each); that is the
  caller's responsibility (the benchmark wiring builds them deep — see 4.4). The
  retriever has no pool knob of its own; it fuses whatever the arms return and cuts to
  `top_k`.
- Named arms (`dense`, `lexical`) rather than a generic sequence because α is
  asymmetric (the dense weight) — clearer than `HybridRetriever`'s symmetric list.
- Satisfies the `Retriever` Protocol, so it flows through `build_run` / `evaluate_one`
  / `--per-query-out` / `eval/significance.py` unchanged.

### 4.3 `eval/tune_alpha.py` (new) — offline α sweep + tuning

Efficiency: retrieve **once per arm**, then sweep α as pure arithmetic (no
re-retrieval per α).

- Build the dense arm and the BM25 arm by **reusing `run_benchmark._build_component`**
  for the dataset/split, with each arm's `top_k` set to `pool_depth`.
- `dense_run = build_run(dense, queries, pooling="max", top_n_docs=None)` and likewise
  `bm25_run` → raw per-query scored `Run`s (all pool docs kept).
- For each α in the grid: `fused = convex_fuse(dense_run, bm25_run, α)`;
  `evaluate_run(fused, qrels, [10])["ndcg@10"]`.
- **Tune** on the dev/train split → `best_alpha = argmax`. **Curve** on the test split
  → write `{alpha, ndcg@10}` rows.
- CLI: `--dataset`, `--tune-split` (dev/train), `--test-split` (default test),
  `--alpha-grid` (default `0.0..1.0` step `0.05`), `--pool-depth` (default 100),
  `--out` (curve CSV). Prints `best_alpha` and its test nDCG@10.

### 4.4 Wiring in `eval/run_benchmark.py` (edit)

- Add `CONVEX_HYBRID_SPECS: Dict[str, List[str]] = {"convex_hybrid_granite_bm25":
  ["granite_dense", "bm25"]}` (mirrors `HYBRID_SPECS`; component 0 = dense, 1 = lexical).
- Add `alpha: float = 0.5` to `BenchmarkConfig` + a `--alpha` CLI flag.
- In `_build_named`: if `name in CONVEX_HYBRID_SPECS`, build the two components **at
  pool depth** — `_build_component(part, ..., top_k=top_k * config.dense_fanout)`, so
  each arm returns ≈`max(k) × dense_fanout` (≈100) candidates, deep enough for fusion
  to reorder — and return `ConvexHybridRetriever(dense, lexical, alpha=config.alpha,
  top_k=top_k * config.dense_fanout)`. `build_run` then trims the fused doc list to
  `max(k)` (contract 3b), symmetric with every other retriever.
- The final test run then goes through the normal harness, e.g.:
  `python -m eval.run_benchmark --dataset fiqa --split test --retrievers
  granite_dense bm25 hybrid_granite_bm25 convex_hybrid_granite_bm25
  --alpha <tuned> --per-query-out results/fiqa_perq.csv`
  so `convex_hybrid_granite_bm25` sits in the same per-query CSV as the RRF hybrid and
  pure dense → significance (convex-vs-dense, convex-vs-RRF) via the existing
  `eval/significance.py`, paired randomization test.

## 5. Experiment protocol

| Dataset | Tune split | Test split | Notes |
|---|---|---|---|
| FiQA | dev | test | The set where dense significantly beats peers; best chance for a lexical arm to add complementary value. |
| NFCorpus | dev | test | |
| SciFact | train (no canonical dev) | test | Use train qrels for tuning. |
| Touché-2020 **or** ArguAna (stretch) | see caveat | test | Lexical-favourable: the strongest test of whether convex can *ever* win. Touché has no train/dev → tune via **transfer-α** (use another set's α) or note as a caveat. |

- Pool depth per arm: ≈100 candidates/query (= `max(k) × dense_fanout` in the
  benchmark wiring; a `--pool-depth` flag, default 100, in `tune_alpha`). Cheap; gives
  fusion room to reorder.
- **Bonus analysis (near-free):** once per-dataset dev-tuned α exist, check whether a
  single fixed α is near-optimal across sets — the convex-α-transfers claim
  (Bruch et al.). Reported as a robustness sub-result, not a separate method.
- Where to run: SciFact/NFCorpus are tiny but dense indexing hits the faiss Windows
  bug, so run on **BluePebble**; FiQA (57k) on HPC. New script
  `scripts/run_convex_hybrid.slurm` (per dataset: tune α, then the headline run).

## 6. TDD plan (tests first; existing suite 225/1 stays green)

- `tests/test_fusion.py` (new):
  - `minmax_normalize`: empty `{}`; single doc; all-equal scores → all 1.0; normal range → endpoints 0 and 1.
  - `fuse_one` / `convex_fuse`: α=1 → dense order exactly; α=0 → lexical order exactly;
    intermediate → expected blended order on a hand-checked example; doc present in
    only one arm contributes 0 for the other; union of doc_ids covered; α outside
    [0,1] raises.
- `tests/test_retrieval_hybrid.py` (extend): `ConvexHybridRetriever` `isinstance`
  `Retriever`; with `FakeRetriever` arms, α=1 reproduces the dense ranking and α=0 the
  lexical ranking; returns ≤ `top_k`; carries text.
- `tests/test_tune_alpha.py` (new): synthetic two-arm runs + qrels where the optimal α
  is known → the sweep picks it; curve CSV has one row per grid point; tuning reads
  only the split it is given (leakage-free by construction).

## 7. Artifacts / outputs

- `results/convex_alpha_curve_<dataset>.csv` — the headline α-curve figure (P7 plots).
- Convex rows at the tuned α in the fair results CSV, beside `hybrid_granite_bm25`
  (RRF) and `granite_dense`; per-query CSVs for significance.
- `scripts/run_convex_hybrid.slurm`.
- [results-summary.md](../../results-summary.md) finding #4 updated with the convex
  result + the partial 2×2 (RRF/convex × BM25), with the SPLADE column marked Phase 2.

## 8. The 2×2 this builds toward (final report)

|  | BM25 lexical arm | SPLADE lexical arm (Phase 2) |
|---|---|---|
| **RRF fusion** | done (finding #4, lost) | Phase 2 |
| **Convex fusion** | **Phase 1 (this spec)** | Phase 2 |

Plus the pure-`granite_dense` reference line. Phase 1 fills the bottom-left cell;
Phase 2 fills the right column.

## 9. Risks / caveats (do not oversell)

- On the small saturated sets (SciFact/NFCorpus) convex will **probably still not beat
  dense** — expected α-curve peak at α→1. The value there is the clean explanation, not
  a win. The realistic upside is FiQA and the lexical-favourable stretch set.
- Per-query min-max is outlier-sensitive; **TMM (theoretical-min-max)** is the noted
  fallback if curves look unstable.
- Touché/ArguAna lack a dev split → α via transfer or cross-val; flag in the writeup.
- `pool_depth=100` is a default; nDCG@10 is unlikely to be sensitive, but note it.

## 10. Definition of done

- New tests pass; full suite green (was 225/1).
- α-curve CSVs + tuned-α headline rows produced for the 3 core sets (stretch set if time).
- Significance computed for convex-vs-dense and convex-vs-RRF.
- finding #4 in `results-summary.md` updated; partial 2×2 recorded.
