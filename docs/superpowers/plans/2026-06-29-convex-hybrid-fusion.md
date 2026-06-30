# Convex-Combination Hybrid Retrieval (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the failed equal-weight RRF hybrid with a convex-combination hybrid (per-query min-max-normalised dense + BM25 scores, weight α), α-tuned on a dev split, to test whether a properly-fused dense+lexical hybrid can beat pure `granite_dense` on the BEIR sets where RRF lost.

**Architecture:** One pure fusion module (`src/retrieval/fusion.py`) is the single source of truth for the fusion maths, reused by an online `ConvexHybridRetriever` (a `Retriever`, drops into the existing benchmark harness) and an offline α-sweep (`eval/tune_alpha.py`, retrieves once per arm then sweeps α as arithmetic). Wiring in `eval/run_benchmark.py` puts the tuned hybrid into the standard results + per-query + significance pipeline. SPLADE (Phase 2) is out of scope.

**Tech Stack:** Python, pytest, ranx (metrics), faiss (dense index), rank_bm25 (lexical), BluePebble SLURM for the runs.

**Spec:** [docs/superpowers/specs/2026-06-29-convex-hybrid-fusion-design.md](specs/2026-06-29-convex-hybrid-fusion-design.md)

**Conventions:** TDD (test → fail → implement → pass → commit). Commit messages use the repo's `area: summary` style with **no AI co-author trailer**. Run the full suite with `pytest -q` before each commit; it must stay green.

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `src/retrieval/fusion.py` | Pure fusion maths: min-max normalize + convex-combine two score sets / Runs. No models, no I/O. | Create |
| `src/retrieval/hybrid.py` | Add `ConvexHybridRetriever` beside `HybridRetriever`. | Modify |
| `eval/tune_alpha.py` | Offline α sweep + dev-tune; reuses arm-building + metrics + fusion. | Create |
| `eval/run_benchmark.py` | Register `convex_hybrid_granite_bm25`; add `alpha` config + `--alpha` CLI; wire in `_build_named`. | Modify |
| `tests/test_fusion.py` | Tests for the fusion maths. | Create |
| `tests/test_retrieval_hybrid.py` | Add `ConvexHybridRetriever` tests. | Modify |
| `tests/test_tune_alpha.py` | Tests for the sweep arithmetic + grid + CSV. | Create |
| `tests/test_run_benchmark.py` | Add convex-hybrid wiring + `--alpha` parse tests. | Modify |
| `scripts/run_convex_hybrid.slurm` | BluePebble runner: tune α then headline run, per dataset. | Create |
| `docs/results-summary.md` | Update finding #4 with the convex result + partial 2×2. | Modify (Task 6, post-run) |

---

## Task 1: Pure fusion maths (`src/retrieval/fusion.py`)

**Files:**
- Create: `src/retrieval/fusion.py`
- Test: `tests/test_fusion.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_fusion.py`:

```python
"""Tests for src/retrieval/fusion.py — convex-combination score fusion.

Pure maths on hand-checkable scores; no models. Pins min-max normalisation, the
alpha endpoints (alpha=1 -> dense order, alpha=0 -> lexical order), the union of
doc ids, and the missing-doc-contributes-0 rule.
"""
from __future__ import annotations

import pytest

from src.retrieval.fusion import convex_fuse, fuse_one, minmax_normalize


def test_minmax_normalize_scales_to_unit_range() -> None:
    assert minmax_normalize({"a": 2.0, "b": 4.0, "c": 6.0}) == {"a": 0.0, "b": 0.5, "c": 1.0}


def test_minmax_normalize_empty_is_empty() -> None:
    assert minmax_normalize({}) == {}


def test_minmax_normalize_all_equal_maps_to_one() -> None:
    # No spread (single doc or ties) -> treat each as maximally relevant for that arm.
    assert minmax_normalize({"a": 3.0, "b": 3.0}) == {"a": 1.0, "b": 1.0}


def test_fuse_one_alpha_one_is_pure_dense() -> None:
    dense = {"d1": 0.9, "d2": 0.1}      # -> d1=1.0, d2=0.0
    lexical = {"d2": 5.0, "d3": 1.0}    # ignored at alpha=1
    fused = fuse_one(dense, lexical, alpha=1.0)
    assert fused["d1"] == pytest.approx(1.0)
    assert fused["d1"] > fused["d2"] and fused["d2"] >= fused["d3"]


def test_fuse_one_alpha_zero_is_pure_lexical() -> None:
    dense = {"d1": 0.9, "d2": 0.1}
    lexical = {"d2": 5.0, "d3": 1.0}    # -> d2=1.0, d3=0.0
    fused = fuse_one(dense, lexical, alpha=0.0)
    assert fused["d2"] == pytest.approx(1.0)
    assert fused["d1"] == pytest.approx(0.0)  # d1 absent from lexical -> 0


def test_fuse_one_blends_union_of_docs() -> None:
    dense = {"d1": 1.0, "d2": 0.0}      # already unit-scaled
    lexical = {"d2": 1.0, "d3": 0.0}
    fused = fuse_one(dense, lexical, alpha=0.5)
    assert set(fused) == {"d1", "d2", "d3"}
    assert fused["d1"] == pytest.approx(0.5)
    assert fused["d2"] == pytest.approx(0.5)
    assert fused["d3"] == pytest.approx(0.0)


def test_fuse_one_rejects_alpha_out_of_range() -> None:
    with pytest.raises(ValueError, match="alpha"):
        fuse_one({"d1": 1.0}, {"d1": 1.0}, alpha=1.5)


def test_convex_fuse_fuses_per_query() -> None:
    dense_run = {"q1": {"d1": 1.0, "d2": 0.0}}
    lexical_run = {"q1": {"d2": 1.0, "d3": 0.0}}
    fused = convex_fuse(dense_run, lexical_run, alpha=0.5)
    assert set(fused["q1"]) == {"d1", "d2", "d3"}
    assert fused["q1"]["d1"] == pytest.approx(0.5)


def test_convex_fuse_handles_query_in_one_run_only() -> None:
    fused = convex_fuse({"q1": {"d1": 1.0, "d2": 0.0}}, {}, alpha=0.5)
    assert fused["q1"]["d1"] == pytest.approx(0.5)  # empty lexical contributes 0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_fusion.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.retrieval.fusion'`.

- [ ] **Step 3: Write the implementation**

Create `src/retrieval/fusion.py`:

```python
"""Convex-combination fusion of dense + lexical retrieval scores.

The RRF hybrid (``src/retrieval/hybrid.HybridRetriever``) fuses *rankings* and lost
to pure dense on every BEIR set (results-summary finding #4). Convex combination
fuses *normalised scores* with a tunable weight ``alpha``, which beats RRF and is
more sample-efficient (Bruch et al., 2023). This module is the pure maths — no
models, no I/O — so it is reused by both the online ``ConvexHybridRetriever`` and the
offline alpha sweep (``eval/tune_alpha.py``), keeping one definition of the fusion.

``alpha`` is the DENSE weight: ``fused = alpha * norm(dense) + (1 - alpha) *
norm(lexical)``. ``alpha = 1`` -> pure dense ranking; ``alpha = 0`` -> pure lexical.

Types are local (``Scores`` = one query's ``{doc_id: score}``; a per-query
``Dict[str, Scores]`` is the ``Run`` shape) so ``src`` keeps no dependency on
``eval``.
"""

from __future__ import annotations

from typing import Callable, Dict

Scores = Dict[str, float]  # {doc_id: score} for a single query


def minmax_normalize(scores: Scores) -> Scores:
    """Min-max scale one query's scores to [0, 1].

    Empty input -> empty output. When all scores are equal (max == min), every
    document maps to 1.0: a retrieved doc with no spread is treated as maximally
    relevant by that arm (it was retrieved at all), and the alpha weighting still
    blends it against the other arm.
    """
    if not scores:
        return {}
    lo = min(scores.values())
    hi = max(scores.values())
    if hi == lo:
        return {doc_id: 1.0 for doc_id in scores}
    span = hi - lo
    return {doc_id: (s - lo) / span for doc_id, s in scores.items()}


def fuse_one(
    dense: Scores,
    lexical: Scores,
    alpha: float,
    normalize: Callable[[Scores], Scores] = minmax_normalize,
) -> Scores:
    """Convex-combine one query's two arms into a fused ``{doc_id: score}``.

    Each arm is normalised independently, then over the union of doc ids
    ``fused = alpha * dense + (1 - alpha) * lexical``, with a doc absent from an arm
    contributing 0 for that arm. ``alpha`` is the dense weight, in [0, 1].
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1]; got {alpha}.")
    nd = normalize(dense)
    nl = normalize(lexical)
    doc_ids = set(nd) | set(nl)
    return {
        doc_id: alpha * nd.get(doc_id, 0.0) + (1.0 - alpha) * nl.get(doc_id, 0.0)
        for doc_id in doc_ids
    }


def convex_fuse(
    dense_run: Dict[str, Scores],
    lexical_run: Dict[str, Scores],
    alpha: float,
    normalize: Callable[[Scores], Scores] = minmax_normalize,
) -> Dict[str, Scores]:
    """Convex-combine two per-query Runs into one fused Run (the offline path).

    Fuses query-by-query over the union of query ids; a query present in only one
    run is fused against an empty other arm. See :func:`fuse_one`.
    """
    qids = set(dense_run) | set(lexical_run)
    return {
        qid: fuse_one(dense_run.get(qid, {}), lexical_run.get(qid, {}), alpha, normalize)
        for qid in qids
    }
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_fusion.py -q`
Expected: PASS (9 passed).

- [ ] **Step 5: Commit**

```bash
git add src/retrieval/fusion.py tests/test_fusion.py
git commit -m "retrieval: add convex-combination score fusion (pure maths)"
```

---

## Task 2: `ConvexHybridRetriever` (`src/retrieval/hybrid.py`)

**Files:**
- Modify: `src/retrieval/hybrid.py` (add a class; add one import)
- Test: `tests/test_retrieval_hybrid.py` (append tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_retrieval_hybrid.py`. First extend the existing import line:

```python
from src.retrieval.hybrid import ConvexHybridRetriever, HybridRetriever
```

(replacing the current `from src.retrieval.hybrid import HybridRetriever`). Then append:

```python
def _chunk(doc_id: str, score: float) -> RetrievedChunk:
    """A scored chunk — convex fusion uses the score, not just the rank."""
    return RetrievedChunk(doc_id=doc_id, text=f"text-{doc_id}", score=score)


def test_convex_hybrid_satisfies_retriever_protocol() -> None:
    arm = FixedRetriever([_chunk("d1", 1.0)])
    assert isinstance(ConvexHybridRetriever(arm, arm, alpha=0.5), Retriever)


def test_convex_alpha_one_reproduces_dense_order() -> None:
    dense = FixedRetriever([_chunk("d1", 0.9), _chunk("d2", 0.1)])
    lexical = FixedRetriever([_chunk("d2", 5.0), _chunk("d3", 1.0)])
    out = ConvexHybridRetriever(dense, lexical, alpha=1.0, top_k=10).retrieve("q")
    assert [c.doc_id for c in out][:2] == ["d1", "d2"]


def test_convex_alpha_zero_reproduces_lexical_order() -> None:
    dense = FixedRetriever([_chunk("d1", 0.9), _chunk("d2", 0.1)])
    lexical = FixedRetriever([_chunk("d2", 5.0), _chunk("d3", 1.0)])
    out = ConvexHybridRetriever(dense, lexical, alpha=0.0, top_k=10).retrieve("q")
    assert [c.doc_id for c in out][0] == "d2"


def test_convex_hybrid_respects_top_k() -> None:
    dense = FixedRetriever([_chunk("d1", 0.9), _chunk("d2", 0.5), _chunk("d3", 0.1)])
    lexical = FixedRetriever([_chunk("d4", 1.0)])
    out = ConvexHybridRetriever(dense, lexical, alpha=0.5, top_k=2).retrieve("q")
    assert len(out) == 2


def test_convex_hybrid_max_pools_repeated_doc_chunks() -> None:
    # Dense returns d1 twice (0.2 then 0.9); the doc's arm score is the MAX (0.9),
    # so at alpha=1 d1 (max 0.9 -> norm 1.0) beats d2 (0.5 -> norm 0.0).
    dense = FixedRetriever([_chunk("d1", 0.2), _chunk("d2", 0.5), _chunk("d1", 0.9)])
    lexical = FixedRetriever([_chunk("d2", 1.0)])
    out = ConvexHybridRetriever(dense, lexical, alpha=1.0, top_k=10).retrieve("q")
    assert out[0].doc_id == "d1"


def test_convex_hybrid_carries_doc_text() -> None:
    dense = FixedRetriever([_chunk("d1", 1.0)])
    lexical = FixedRetriever([_chunk("d1", 1.0)])
    out = ConvexHybridRetriever(dense, lexical, alpha=0.5, top_k=10).retrieve("q")
    assert out[0].text == "text-d1"


def test_convex_hybrid_rejects_bad_alpha_and_top_k() -> None:
    arm = FixedRetriever([_chunk("d1", 1.0)])
    with pytest.raises(ValueError, match="alpha"):
        ConvexHybridRetriever(arm, arm, alpha=2.0)
    with pytest.raises(ValueError, match="top_k"):
        ConvexHybridRetriever(arm, arm, alpha=0.5, top_k=0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_retrieval_hybrid.py -q`
Expected: FAIL with `ImportError: cannot import name 'ConvexHybridRetriever'`.

- [ ] **Step 3: Write the implementation**

In `src/retrieval/hybrid.py`, add this import after the existing `from src.retrieval.base import RetrievedChunk, Retriever`:

```python
from src.retrieval.fusion import Scores, fuse_one
```

Then append the class at the end of the file:

```python
class ConvexHybridRetriever:
    """Fuse a dense and a lexical retriever by convex combination of normalised scores.

    Unlike RRF (:class:`HybridRetriever`), this uses the arms' *scores* — per-query
    min-max normalised — with a tunable weight ``alpha`` (the dense weight).
    ``alpha = 1`` reproduces the dense ranking, ``alpha = 0`` the lexical ranking. The
    fusion maths live in :mod:`src.retrieval.fusion`, shared with the offline alpha
    sweep (``eval/tune_alpha.py``).

    Parameters
    ----------
    dense, lexical:
        The two component retrievers. They must already return a deep candidate pool
        (~100 each) — the caller builds them that way; this class has no pool knob, it
        fuses whatever the arms return and keeps the top ``top_k``.
    alpha:
        Dense weight in [0, 1].
    top_k:
        Number of fused documents to return.
    """

    def __init__(
        self,
        dense: Retriever,
        lexical: Retriever,
        alpha: float,
        top_k: int = 10,
    ) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1]; got {alpha}.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        self.dense = dense
        self.lexical = lexical
        self.alpha = alpha
        self.top_k = top_k

    def _doc_scores(self, retriever: Retriever, query: str) -> tuple[Scores, Dict[str, str]]:
        """Max-pool one arm's chunks to ``{doc_id: score}`` (+ ``{doc_id: text}``).

        A document hit via several chunks is scored at its best (max) chunk, matching
        contract 3, so a multi-chunk document is not under-credited before fusion.
        """
        scores: Scores = {}
        texts: Dict[str, str] = {}
        for chunk in retriever.retrieve(query):
            if chunk.doc_id not in scores or chunk.score > scores[chunk.doc_id]:
                scores[chunk.doc_id] = chunk.score
            texts.setdefault(chunk.doc_id, chunk.text)
        return scores, texts

    def retrieve(self, query: str) -> List[RetrievedChunk]:
        """Return the top-k documents by convex-fused score, ranked best-first."""
        dense_scores, dense_texts = self._doc_scores(self.dense, query)
        lex_scores, lex_texts = self._doc_scores(self.lexical, query)
        fused = fuse_one(dense_scores, lex_scores, self.alpha)
        ranked = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))
        out: List[RetrievedChunk] = []
        for doc_id, score in ranked[: self.top_k]:
            text = dense_texts.get(doc_id) or lex_texts.get(doc_id, "")
            out.append(RetrievedChunk(doc_id=doc_id, text=text, score=score))
        return out
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_retrieval_hybrid.py -q`
Expected: PASS (the new tests plus the existing RRF tests).

- [ ] **Step 5: Commit**

```bash
git add src/retrieval/hybrid.py tests/test_retrieval_hybrid.py
git commit -m "retrieval: add ConvexHybridRetriever (score-level dense+BM25 fusion)"
```

---

## Task 3: Offline α sweep (`eval/tune_alpha.py`)

**Files:**
- Create: `eval/tune_alpha.py`
- Test: `tests/test_tune_alpha.py`

The retrieval-bearing helper `_arm_runs` (builds real arms; needs models + datasets) is exercised only on the HPC run, not unit-tested. The pure pieces — `sweep`, `best_alpha`, `_grid`, `write_curve` — are.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_tune_alpha.py`:

```python
"""Tests for eval/tune_alpha.py — the convex-hybrid alpha sweep/tuning.

Tests the pure sweep arithmetic on synthetic runs where the optimal alpha is known,
plus the grid and CSV writer. The retrieval-bearing _arm_runs (models + datasets) is
exercised only on the HPC run, not here.
"""
from __future__ import annotations

from pathlib import Path

from eval.tune_alpha import _grid, best_alpha, sweep, write_curve


def test_grid_is_inclusive_unit_interval() -> None:
    assert _grid(0.5) == [0.0, 0.5, 1.0]
    g = _grid(0.05)
    assert g[0] == 0.0 and g[-1] == 1.0 and len(g) == 21


def test_sweep_prefers_lexical_when_only_lexical_finds_the_gold() -> None:
    # Gold for q1 is d2. Dense ranks d1 top and d2 last; BM25 ranks d2 top.
    # alpha=0 (pure lexical) puts the gold first -> highest nDCG.
    dense_run = {"q1": {"d1": 1.0, "d2": 0.0}}
    bm25_run = {"q1": {"d2": 1.0, "d1": 0.0}}
    qrels = {"q1": {"d2": 1}}
    a_star, _ = best_alpha(sweep(dense_run, bm25_run, qrels, [0.0, 1.0]))
    assert a_star == 0.0


def test_sweep_prefers_dense_when_only_dense_finds_the_gold() -> None:
    dense_run = {"q1": {"d1": 1.0, "d2": 0.0}}
    bm25_run = {"q1": {"d2": 1.0, "d1": 0.0}}
    qrels = {"q1": {"d1": 1}}  # gold is d1, which dense ranks top
    a_star, _ = best_alpha(sweep(dense_run, bm25_run, qrels, [0.0, 1.0]))
    assert a_star == 1.0


def test_best_alpha_breaks_ties_toward_smaller_alpha() -> None:
    # Flat curve -> tie; prefer the smaller alpha (less reliance on the weaker arm).
    assert best_alpha([(0.0, 0.5), (0.5, 0.5), (1.0, 0.5)])[0] == 0.0


def test_write_curve_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "curve.csv"
    write_curve([(0.0, 0.1), (1.0, 0.9)], path)
    lines = path.read_text().splitlines()
    assert lines[0] == "alpha,ndcg@10"
    assert lines[1] == "0.0,0.1"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_tune_alpha.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'eval.tune_alpha'`.

- [ ] **Step 3: Write the implementation**

Create `eval/tune_alpha.py`:

```python
"""Offline alpha sweep + dev-tuning for the convex-combination hybrid.

Convex fusion is deterministic given each arm's per-query scores, so we retrieve
ONCE per arm and then sweep alpha as pure arithmetic — no re-retrieval per alpha.

Outputs:
- tune: the best alpha on a dev/train split (argmax nDCG@10) — the deploy point.
- curve: nDCG@10 vs alpha on the test split — the headline artifact ("does ANY
  alpha beat pure dense?"). alpha = 1.0 in the curve IS pure dense, so it should
  match the published granite_dense nDCG@10 (a consistency check).

Reuses the benchmark's arm-building (``run_benchmark._build_component`` + ``build_run``)
and metrics (``ir_metrics.ndcg_at_k``); the fusion maths come from
``src.retrieval.fusion.convex_fuse``.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from eval.benchmarks.loader import load_benchmark
from eval.ir_metrics import Qrels, Run, ndcg_at_k
from eval.run_benchmark import BenchmarkConfig, _build_component, build_run
from src.retrieval.fusion import convex_fuse


def _arm_runs(
    dataset: str,
    split: str,
    chunk_unit: str,
    pool_depth: int,
    cache_dir: Optional[Path],
    max_queries: Optional[int],
    max_docs: Optional[int],
) -> Tuple[Run, Run, Qrels]:
    """Retrieve once per arm on a split; return ``(dense_run, bm25_run, qrels)``.

    Each arm is built and its run trimmed to ``pool_depth`` docs per query, so fusion
    has a deep, equal-depth pool to reorder. Reuses the benchmark's component builder
    (so the dense arm is identical to the audited ``granite_dense``) and its index
    cache, so the test-split index is reused by the final ``run_benchmark`` run.
    """
    data = load_benchmark(
        dataset, split=split, max_queries=max_queries, max_docs=max_docs
    )
    config = BenchmarkConfig(
        dataset=dataset, split=split, chunk_unit=chunk_unit, index_cache_dir=cache_dir
    )
    doc_ids = list(data.corpus.keys())
    corpus = list(data.corpus.values())
    dense = _build_component("granite_dense", config, data, corpus, doc_ids, pool_depth)
    bm25 = _build_component("bm25", config, data, corpus, doc_ids, pool_depth)
    dense_run = build_run(dense, data.queries, pooling="max", top_n_docs=pool_depth)
    bm25_run = build_run(bm25, data.queries, pooling="max", top_n_docs=pool_depth)
    return dense_run, bm25_run, data.qrels


def sweep(
    dense_run: Run,
    bm25_run: Run,
    qrels: Qrels,
    grid: List[float],
) -> List[Tuple[float, float]]:
    """Return ``[(alpha, ndcg@10)]`` over the grid (pure arithmetic, no retrieval)."""
    out: List[Tuple[float, float]] = []
    for alpha in grid:
        fused = convex_fuse(dense_run, bm25_run, alpha)
        out.append((alpha, ndcg_at_k(fused, qrels, 10)))
    return out


def best_alpha(curve: List[Tuple[float, float]]) -> Tuple[float, float]:
    """The ``(alpha, ndcg)`` with the highest nDCG; ties break to the smaller alpha
    (less reliance on the weaker lexical arm)."""
    return max(curve, key=lambda pair: (pair[1], -pair[0]))


def _grid(step: float) -> List[float]:
    """alpha grid ``0.0, step, ..., 1.0`` inclusive."""
    n = round(1.0 / step)
    return [round(i * step, 4) for i in range(n + 1)]


def write_curve(curve: List[Tuple[float, float]], path: Path) -> None:
    """Write the ``alpha,ndcg@10`` curve CSV (the headline figure)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["alpha", "ndcg@10"])
        for alpha, ndcg in curve:
            writer.writerow([alpha, ndcg])


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m eval.tune_alpha",
        description="Sweep/tune the convex-hybrid alpha (dense vs BM25) without test leakage.",
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--tune-split", default="dev",
                        help="Split to pick alpha on, e.g. dev or train (default: %(default)s).")
    parser.add_argument("--test-split", default="test",
                        help="Split to report the alpha-curve on (default: %(default)s).")
    parser.add_argument("--chunk-unit", default="word", choices=["word", "token"],
                        help="Chunking unit for the dense arm (default: %(default)s).")
    parser.add_argument("--alpha-step", type=float, default=0.05)
    parser.add_argument("--pool-depth", type=int, default=100)
    parser.add_argument("--cache-dir", type=Path, default=None,
                        help="Cache dense indexes here (shared with run_benchmark).")
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None,
                        help="Write the TEST alpha-curve CSV here (alpha,ndcg@10).")
    parser.add_argument("--best-alpha-out", type=Path, default=None,
                        help="Write the chosen alpha (a float) here, for the run script.")
    args = parser.parse_args(argv)

    grid = _grid(args.alpha_step)

    # Tune on the dev/train split.
    d_run, b_run, qrels = _arm_runs(
        args.dataset, args.tune_split, args.chunk_unit, args.pool_depth,
        args.cache_dir, args.max_queries, args.max_docs,
    )
    a_star, ndcg_dev = best_alpha(sweep(d_run, b_run, qrels, grid))
    print(f"[tune:{args.tune_split}] best alpha = {a_star} (nDCG@10 = {ndcg_dev:.4f})")

    # Report the curve on the test split.
    td_run, tb_run, tqrels = _arm_runs(
        args.dataset, args.test_split, args.chunk_unit, args.pool_depth,
        args.cache_dir, args.max_queries, args.max_docs,
    )
    test_curve = sweep(td_run, tb_run, tqrels, grid)
    by_alpha = dict(test_curve)
    delta = by_alpha[a_star] - by_alpha[1.0]
    print(
        f"[test:{args.test_split}] nDCG@10 @ alpha*={a_star}: {by_alpha[a_star]:.4f} "
        f"| pure dense (alpha=1.0): {by_alpha[1.0]:.4f} | delta: {delta:+.4f}"
    )
    if args.out is not None:
        write_curve(test_curve, args.out)
        print(f"wrote test alpha-curve to {args.out}")
    if args.best_alpha_out is not None:
        args.best_alpha_out.parent.mkdir(parents=True, exist_ok=True)
        args.best_alpha_out.write_text(str(a_star))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_tune_alpha.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add eval/tune_alpha.py tests/test_tune_alpha.py
git commit -m "eval: add offline alpha sweep + dev-tuning for convex hybrid"
```

---

## Task 4: Wire the convex hybrid into `eval/run_benchmark.py`

**Files:**
- Modify: `eval/run_benchmark.py` (import; `CONVEX_HYBRID_SPECS`; `alpha` config field; `_build_named` branch; `--alpha` CLI + construction)
- Test: `tests/test_run_benchmark.py` (append two tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_run_benchmark.py`:

```python
def test_build_retrievers_builds_convex_hybrid_of_dense_and_bm25(monkeypatch) -> None:
    # convex_hybrid_granite_bm25 fuses the granite dense retriever and BM25 by convex
    # combination of normalised scores (alpha = dense weight). At alpha=1 it ranks by
    # the dense arm alone. Embedding model monkeypatched so nothing downloads.
    from src.retrieval.hybrid import ConvexHybridRetriever

    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer", FakeSentenceTransformer
    )
    data = BenchmarkData(
        corpus={"d1": "granite retrieval", "d2": "banana cake"},
        queries={"q1": "granite retrieval"},
        qrels={"q1": {"d1": 1}},
    )
    config = BenchmarkConfig(
        retrievers=["convex_hybrid_granite_bm25"], k_values=[1], alpha=1.0
    )

    retrievers = _build_retrievers(config, data)

    hybrid = retrievers["convex_hybrid_granite_bm25"]
    assert isinstance(hybrid, ConvexHybridRetriever)
    assert hybrid.alpha == 1.0
    assert hybrid.retrieve("granite retrieval")[0].doc_id == "d1"


def test_parse_args_alpha() -> None:
    assert _parse_args([]).alpha == 0.5
    assert _parse_args(["--alpha", "0.7"]).alpha == 0.7
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_run_benchmark.py -q -k "convex or alpha"`
Expected: FAIL — `convex_hybrid_granite_bm25` is unknown (raises `ValueError: Unknown retriever`), and `BenchmarkConfig` has no `alpha`.

- [ ] **Step 3: Edit the import**

In `eval/run_benchmark.py`, change:

```python
from src.retrieval.hybrid import HybridRetriever
```

to:

```python
from src.retrieval.hybrid import ConvexHybridRetriever, HybridRetriever
```

- [ ] **Step 4: Add the `alpha` config field**

In `BenchmarkConfig`, immediately after the line `    k_rrf: int = 60`, add:

```python
    alpha: float = 0.5
```

- [ ] **Step 5: Register the convex hybrid spec**

Immediately after the `HYBRID_SPECS` dict definition (the block ending with `"hybrid_granite_small_bm25": ["granite_small_dense", "bm25"],\n}`), add:

```python
# Convex-combination hybrids: name -> [dense_name, lexical_name]. Unlike HYBRID_SPECS
# (RRF over rankings), these fuse per-query min-max-normalised SCORES with weight
# config.alpha (the dense weight), via ConvexHybridRetriever. The fix for finding #4's
# RRF-loses result: convex combination beats RRF and is tunable (Bruch et al., 2023).
CONVEX_HYBRID_SPECS: Dict[str, List[str]] = {
    "convex_hybrid_granite_bm25": ["granite_dense", "bm25"],
}
```

- [ ] **Step 6: Add the `_build_named` branch**

In `_build_named`, after the `if name in HYBRID_SPECS:` block (the one returning `HybridRetriever(...)`) and before the final `return _build_component(...)`, insert:

```python
    if name in CONVEX_HYBRID_SPECS:
        dense_name, lexical_name = CONVEX_HYBRID_SPECS[name]
        pool = top_k * config.dense_fanout  # ~100 candidates/arm for fusion to reorder
        # dense over-fetches internally (top_k * dense_fanout chunks); BM25 returns
        # `pool` docs directly — so both arms surface ~`pool` docs.
        dense = _build_component(dense_name, config, data, corpus, doc_ids, top_k)
        lexical = _build_component(lexical_name, config, data, corpus, doc_ids, pool)
        return ConvexHybridRetriever(dense, lexical, alpha=config.alpha, top_k=pool)
```

- [ ] **Step 7: Add the `--alpha` CLI argument**

In `_parse_args`, after the `--k-rrf` argument block (the `parser.add_argument("--k-rrf", ...)` call), add:

```python
    parser.add_argument(
        "--alpha",
        type=float,
        default=defaults.alpha,
        dest="alpha",
        help="Dense weight for convex-combination hybrids "
        "(0 = BM25 only, 1 = dense only; default: %(default)s).",
    )
```

Then in the `return BenchmarkConfig(...)` call at the end of `_parse_args`, after the line `        k_rrf=args.k_rrf,`, add:

```python
        alpha=args.alpha,
```

- [ ] **Step 8: Run the tests to verify they pass**

Run: `pytest tests/test_run_benchmark.py -q -k "convex or alpha"`
Expected: PASS (2 passed).

- [ ] **Step 9: Run the full suite**

Run: `pytest -q`
Expected: all green (previous passing count + the new tests; the one pre-existing xfail stays xfail).

- [ ] **Step 10: Commit**

```bash
git add eval/run_benchmark.py tests/test_run_benchmark.py
git commit -m "eval: wire convex_hybrid_granite_bm25 + --alpha into run_benchmark"
```

---

## Task 5: BluePebble runner (`scripts/run_convex_hybrid.slurm`)

**Files:**
- Create: `scripts/run_convex_hybrid.slurm`

This task has no unit test (it's an ops script). Verify it by reading it back and, on the login node, a `--help` smoke of the two entry points (Step 2).

- [ ] **Step 1: Write the script**

Create `scripts/run_convex_hybrid.slurm`:

```bash
#!/bin/bash
# Phase 1: convex-combination hybrid (granite_dense + BM25) vs the RRF hybrid and
# pure dense, on the BEIR sets where RRF lost (finding #4). For each dataset:
#   1) tune alpha on its dev/train split + write the test alpha-curve (eval.tune_alpha)
#   2) run granite_dense, bm25, RRF hybrid, and the convex hybrid (at the tuned alpha)
#      with --per-query-out, so significance (convex vs dense, convex vs RRF) follows.
#
#   mkdir -p logs && sbatch scripts/run_convex_hybrid.slurm
#
# FIRST, on the LOGIN node (compute nodes are offline) pre-fetch the datasets' splits
# (models are already cached; BM25 needs none). dev/train share the same corpus as
# test, so this only adds the dev/train queries+qrels:
#   source scripts/env.sh
#   python -c "from eval.benchmarks.loader import load_benchmark as L; \
#     [L(d, split=s) for d, s in [('scifact','train'),('nfcorpus','dev'),('fiqa','dev')]]"
#SBATCH --job-name=convex-hybrid
#SBATCH --account=coms039904
#SBATCH --partition=gpu_short
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

# Compute nodes have no python by default; load the module the venv was built with.
module load languages/python/3.12.3

export HF_HOME=/user/work/$USER/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export IR_DATASETS_HOME=/user/work/$USER/ir_datasets
export PYTHONUNBUFFERED=1

source /user/work/$USER/venv/bin/activate
cd /user/work/$USER/IBM_Granite_Project

python -c "import torch; print('CUDA available:', torch.cuda.is_available(), '| torch', torch.__version__)"

CACHE=/user/work/$USER/index_cache

# dataset : tune-split : chunk-unit. SciFact/NFCorpus were audited in word mode,
# FiQA in token mode — match each so the alpha=1 point reproduces the published
# granite_dense nDCG@10 (a consistency check).
for SPEC in "scifact:train:word" "nfcorpus:dev:word" "fiqa:dev:token"; do
    D="${SPEC%%:*}"; REST="${SPEC#*:}"; TUNE="${REST%%:*}"; UNIT="${REST##*:}"
    echo "=== $D (tune on $TUNE, chunk-unit $UNIT) ==="

    python -m eval.tune_alpha \
        --dataset "$D" \
        --tune-split "$TUNE" \
        --test-split test \
        --chunk-unit "$UNIT" \
        --cache-dir "$CACHE" \
        --out "results/convex_alpha_curve_${D}.csv" \
        --best-alpha-out "results/convex_best_alpha_${D}.txt"

    ALPHA="$(cat "results/convex_best_alpha_${D}.txt")"
    echo "tuned alpha for $D = $ALPHA"

    python -m eval.run_benchmark \
        --dataset "$D" \
        --split test \
        --retrievers granite_dense bm25 hybrid_granite_bm25 convex_hybrid_granite_bm25 \
        --alpha "$ALPHA" \
        --chunk-unit "$UNIT" \
        --out "results/${D}_convex.csv" \
        --per-query-out "results/${D}_convex_per_query.csv" \
        --cache-dir "$CACHE"
done

echo "DONE. Per dataset: results/convex_alpha_curve_<d>.csv (curve),"
echo "results/<d>_convex.csv (headline), results/<d>_convex_per_query.csv (significance)."
```

- [ ] **Step 2: Smoke-check the entry points (login node)**

On BluePebble's login node:

Run: `source scripts/env.sh && python -m eval.tune_alpha --help && python -m eval.run_benchmark --help | grep -- --alpha`
Expected: `tune_alpha` usage prints; the `run_benchmark` help shows the `--alpha` flag.

- [ ] **Step 3: Commit**

```bash
git add scripts/run_convex_hybrid.slurm
git commit -m "scripts: add run_convex_hybrid.slurm (tune alpha + headline run)"
```

---

## Task 6: Run on BluePebble + record results (`docs/results-summary.md`)

Operational task (no unit test). The numbers in Step 3 are transcribed from the run's
output CSVs — they are produced by the run, not known in advance.

- [ ] **Step 1: Pre-fetch splits (login node) and submit**

```bash
source scripts/env.sh
python -c "from eval.benchmarks.loader import load_benchmark as L; \
  [L(d, split=s) for d, s in [('scifact','train'),('nfcorpus','dev'),('fiqa','dev')]]"
mkdir -p logs && sbatch scripts/run_convex_hybrid.slurm
```

Wait for `COMPLETED` (`sacct -j <jobid> --format=JobID,State,Elapsed,ExitCode`).

- [ ] **Step 2: Read the curves, headlines, and significance**

```bash
for D in scifact nfcorpus fiqa; do
  echo "=== $D alpha-curve ==="; column -t -s, "results/convex_alpha_curve_${D}.csv"
  echo "=== $D headline ===";    column -t -s, "results/${D}_convex.csv"
  echo "=== $D significance (convex vs granite_dense) ===";
  python -m eval.significance --per-query-csv "results/${D}_convex_per_query.csv" --reference granite_dense
done
```

Sanity check: in each `convex_alpha_curve_<d>.csv`, the `alpha=1.0` row's `ndcg@10`
should match that dataset's published `granite_dense` nDCG@10 in Table 1 of
`docs/results-summary.md` (0.767 SciFact / 0.375 NFCorpus / 0.459 FiQA). A mismatch
means the chunking/pool settings diverged from the audited run — stop and reconcile
before recording.

- [ ] **Step 3: Update finding #4 in `docs/results-summary.md`**

Replace finding #4 ("RRF hybrid (dense+BM25) does NOT help…") with the decomposition,
filling the bracketed values from Step 2's output:

```markdown
4. **Fusion method matters; the lexical arm is still the ceiling.** Equal-weight
   **RRF** regressed toward BM25 and lost on every set (above). **Convex combination**
   of per-query min-max-normalised scores, with α (the dense weight) tuned on each
   set's dev/train split, recovers most of that loss but **still does not beat pure
   dense** on SciFact/NFCorpus/FiQA: best-α nDCG@10 = <conv_sf>/<conv_nf>/<conv_fi>
   vs dense 0.767/0.375/0.459 (Δ = <d_sf>/<d_nf>/<d_fi>; significance vs dense:
   <p_sf>/<p_nf>/<p_fi>). The tuned α sits at <a_sf>/<a_nf>/<a_fi> — i.e. the sweep
   down-weights the weak BM25 arm toward pure dense (α→1). The α-curves
   (`results/convex_alpha_curve_<dataset>.csv`) peak at/near α=1 on the saturated sets.
   → RRF was the wrong fusion, but even the right fusion can't make rank_bm25 a useful
   complement here; the open question is whether a *stronger* lexical arm (SPLADE,
   Phase 2) changes this. This fills the Convex×BM25 cell of the 2×2:

   |  | BM25 arm | SPLADE arm (Phase 2) |
   |---|---|---|
   | RRF fusion | loses (this finding) | Phase 2 |
   | Convex fusion | <one-line result> | Phase 2 |
```

If, contrary to expectation, the convex hybrid *does* beat dense on a set (Δ > 0,
significant), state that as the positive result and note which set + α.

- [ ] **Step 4: Commit the docs + result CSVs**

```bash
git add docs/results-summary.md results/convex_alpha_curve_*.csv \
        results/*_convex.csv results/*_convex_per_query.csv results/convex_best_alpha_*.txt
git commit -m "docs: record convex-hybrid Phase 1 result (finding #4 decomposition)"
```

---

## Self-Review (completed during planning)

- **Spec coverage:** fusion module (Task 1), `ConvexHybridRetriever` (Task 2), offline
  α sweep with dev-tune + test curve (Task 3), `run_benchmark` wiring + `--alpha`
  (Task 4), slurm runner (Task 5), the three core datasets with per-dataset tune
  split + the α=1 consistency check, significance, and the finding-#4 / 2×2 update
  (Task 6). The lexical-favourable *stretch* set (Touché/ArguAna) from the spec is
  **deliberately deferred** — add it by extending the `SPEC` loop in
  `run_convex_hybrid.slurm` once the three core sets are in (Touché has no dev split,
  so it needs a transfer-α; out of the Phase-1 critical path).
- **Type consistency:** `Scores = Dict[str, float]`; `Run`/`Qrels` imported from
  `eval.ir_metrics` only in `eval/` code (not in `src/`); `fuse_one`/`convex_fuse`/
  `minmax_normalize`/`ConvexHybridRetriever(dense, lexical, alpha, top_k)`/`sweep`/
  `best_alpha`/`_grid`/`write_curve` names match across tasks and call sites.
- **Placeholder scan:** none — the only bracketed values are in the Task 6 doc
  template, explicitly transcribed from the run output.
```
