# Gated dynamic-alpha Corroboration (offline) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A single offline module `eval/gate_corroboration.py` that, from the certified runs dump, (1) re-certifies finding 15 under a dev/test protocol and (2) tests whether a per-query gated corroboration blend beats the global blend — pure arithmetic, no LLM/GPU.

**Architecture:** New module of small pure functions (signals → gate mask → gated fuse → sweep → dev selection → test certification → flip table) composed by a thin CLI `main`. Reuses `tune_corroboration.load_runs/per_query_hits`, `fusion.fuse_one/convex_fuse/minmax_normalize`, `tune_alpha._grid`, `run_benchmark.write_per_query_csv` (lazy import). Nothing existing is modified. Spec: `docs/superpowers/specs/2026-07-06-gated-corroboration-design.md`.

**Tech Stack:** Python stdlib (argparse/csv/random/pathlib/NamedTuple), pytest. No new dependencies.

---

## File structure

- Create: `eval/gate_corroboration.py` — all new logic (one file, one responsibility: offline gated analysis from a runs dump).
- Create: `tests/test_gate_corroboration.py` — all tests (mirrors `tests/test_tune_corroboration.py` style: pure-function tests + tmp_path end-to-end).
- Not modified: `eval/tune_corroboration.py`, `src/retrieval/fusion.py`, `eval/tune_alpha.py`, `eval/run_benchmark.py` (import-only reuse).

Run tests with: `python -m pytest tests/test_gate_corroboration.py -v` (Windows PowerShell and bash identical).

**Type vocabulary used throughout** (matches existing code): a `Run` is `Dict[qid, Dict[doc_id, float]]` (`eval.ir_metrics.Run`); `needles` is `Dict[qid, needle_doc_id]`; per-query hits are `Dict[qid, 1.0|0.0]`.

---

### Task 1: Module skeleton + gate signals

**Files:**
- Create: `eval/gate_corroboration.py`
- Create: `tests/test_gate_corroboration.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_gate_corroboration.py`:

```python
"""Tests for the offline gated dynamic-alpha corroboration analysis."""
from eval.gate_corroboration import margin_signal, max_votes_signal


def test_max_votes_signal_takes_the_per_query_max():
    corr = {"q1": {"a": 0.0, "b": 2.0, "c": 1.0}, "q2": {"a": 0.0, "b": 0.0}}
    assert max_votes_signal(corr) == {"q1": 2.0, "q2": 0.0}


def test_max_votes_signal_empty_query_scores_zero():
    assert max_votes_signal({"q1": {}}) == {"q1": 0.0}


def test_margin_signal_is_top1_minus_top2_after_minmax():
    # exactly-representable scores: minmax -> 1.0/0.5/0.0 -> margin 0.5 (no float dust)
    rel = {"q1": {"a": 1.0, "b": 0.5, "c": 0.0}}
    assert margin_signal(rel) == {"q1": 0.5}


def test_margin_signal_degenerate_cases_are_zero():
    # all-equal scores minmax to all-1.0 (margin 0); a single doc has no top2.
    rel = {"q1": {"a": 0.7, "b": 0.7}, "q2": {"a": 0.9}}
    assert margin_signal(rel) == {"q1": 0.0, "q2": 0.0}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_gate_corroboration.py -v`
Expected: FAIL (ModuleNotFoundError / ImportError: `eval.gate_corroboration`)

- [ ] **Step 3: Write minimal implementation**

Create `eval/gate_corroboration.py`:

```python
"""Offline gated dynamic-alpha analysis for the corroboration reranker.

Everything here is pure arithmetic on a runs dump produced by
``eval.tune_corroboration --dump-runs`` (no LLM, no GPU, no index): dev/test split,
per-query gate signals, gated convex blend, dev-only selection, test-only
certification, and the descriptive flip table. Two deliverables: re-certify
finding 15 under an honest dev/test protocol, and test whether a per-query gate
(blend only where consensus exists / where the first stage is unsure) beats the
global blend. Design spec: docs/superpowers/specs/2026-07-06-gated-corroboration-design.md.

A structural fact this leans on: ``minmax_normalize`` maps an all-equal score set to
all-1.0, and a convex blend with a constant is order-preserving -- so zero-consensus
queries are untouched by the global blend already; the gate's room to act is the
weak-consensus region (votes gate tau=1 must therefore reproduce the global blend).
"""
from __future__ import annotations

from typing import Dict

from eval.ir_metrics import Run
from src.retrieval.fusion import minmax_normalize


def max_votes_signal(corroboration_run: Run) -> Dict[str, float]:
    """Per-query strongest consensus: max raw vote count (0.0 for an empty pool)."""
    return {
        qid: (max(scores.values()) if scores else 0.0)
        for qid, scores in corroboration_run.items()
    }


def margin_signal(relevance_run: Run) -> Dict[str, float]:
    """Per-query first-stage confidence: top1 - top2 of the min-max-normalised
    relevance scores (0.0 when fewer than 2 docs; all-equal scores normalise to
    all-1.0 so the margin is 0.0 and a margin gate fires -- harmless, the blend
    then decides)."""
    out: Dict[str, float] = {}
    for qid, scores in relevance_run.items():
        norm = sorted(minmax_normalize(scores).values(), reverse=True)
        out[qid] = norm[0] - norm[1] if len(norm) >= 2 else 0.0
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_gate_corroboration.py -v`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add eval/gate_corroboration.py tests/test_gate_corroboration.py
git commit -m "feat(eval): per-query gate signals (max votes, relevance margin) for gated corroboration"
```

---

### Task 2: Gate mask + gated fuse

**Files:**
- Modify: `eval/gate_corroboration.py` (append)
- Modify: `tests/test_gate_corroboration.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_gate_corroboration.py` (and extend the import to include `gate_mask`, `gated_fuse`):

```python
import pytest

from eval.gate_corroboration import gate_mask, gated_fuse
from src.retrieval.fusion import fuse_one


def test_gate_mask_global_always_fires():
    votes, margins = {"q1": 0.0, "q2": 3.0}, {"q1": 0.5, "q2": 0.01}
    assert gate_mask("global", None, votes, margins) == {"q1": True, "q2": True}


def test_gate_mask_votes_fires_at_or_above_tau():
    votes, margins = {"q1": 1.0, "q2": 2.0, "q3": 0.0}, {}
    assert gate_mask("votes", 2.0, votes, margins) == {"q1": False, "q2": True, "q3": False}


def test_gate_mask_margin_fires_below_threshold():
    votes, margins = {}, {"q1": 0.04, "q2": 0.5}
    assert gate_mask("margin", 0.05, votes, margins) == {"q1": True, "q2": False}


def test_gate_mask_rejects_unknown_family():
    with pytest.raises(ValueError):
        gate_mask("nonsense", 1.0, {}, {})


def test_gated_fuse_gate_off_keeps_pure_relevance_order():
    rel = {"q1": {"cf": 0.99, "n1": 0.5}}
    cor = {"q1": {"cf": 0.0, "n1": 5.0}}
    fused = gated_fuse(rel, cor, alpha=0.5, gate={"q1": False})
    assert fused["q1"] == rel["q1"]  # untouched: first-stage scores pass through


def test_gated_fuse_gate_on_matches_fuse_one():
    rel = {"q1": {"cf": 0.99, "n1": 0.5}}
    cor = {"q1": {"cf": 0.0, "n1": 5.0}}
    fused = gated_fuse(rel, cor, alpha=0.5, gate={"q1": True})
    assert fused["q1"] == fuse_one(rel["q1"], cor["q1"], 0.5)


def test_gated_fuse_mixed_gate():
    rel = {"q1": {"a": 0.9, "b": 0.1}, "q2": {"a": 0.9, "b": 0.1}}
    cor = {"q1": {"a": 0.0, "b": 3.0}, "q2": {"a": 0.0, "b": 3.0}}
    fused = gated_fuse(rel, cor, alpha=0.0, gate={"q1": True, "q2": False})
    assert fused["q1"] == fuse_one(rel["q1"], cor["q1"], 0.0)
    assert fused["q2"] == rel["q2"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_gate_corroboration.py -v`
Expected: ImportError (`gate_mask`)

- [ ] **Step 3: Write minimal implementation**

Append to `eval/gate_corroboration.py` (extend the fusion import line to `from src.retrieval.fusion import fuse_one, minmax_normalize`; add `Optional` to the typing import):

```python
def gate_mask(
    family: str,
    param: Optional[float],
    votes: Dict[str, float],
    margins: Dict[str, float],
) -> Dict[str, bool]:
    """Per-query blend/keep decision. ``global`` always blends; ``votes`` blends iff
    ``max_votes >= param``; ``margin`` blends iff ``margin < param`` (blend only where
    the first stage is unsure)."""
    if family == "global":
        return {qid: True for qid in votes}
    if family == "votes":
        return {qid: v >= param for qid, v in votes.items()}
    if family == "margin":
        return {qid: m < param for qid, m in margins.items()}
    raise ValueError(f"unknown gate family: {family!r}")


def gated_fuse(
    relevance_run: Run, corroboration_run: Run, alpha: float, gate: Dict[str, bool]
) -> Run:
    """Blend gated-on queries with ``fuse_one``; gated-off queries keep their raw
    first-stage scores (identical ranking to alpha=1 for that query)."""
    fused: Run = {}
    for qid, rel in relevance_run.items():
        if gate.get(qid, False):
            fused[qid] = fuse_one(rel, corroboration_run.get(qid, {}), alpha)
        else:
            fused[qid] = dict(rel)
    return fused
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_gate_corroboration.py -v`
Expected: 11 PASS

- [ ] **Step 5: Commit**

```bash
git add eval/gate_corroboration.py tests/test_gate_corroboration.py
git commit -m "feat(eval): gate mask + gated convex fuse (gated-off queries keep first-stage order)"
```

---

### Task 3: Deterministic dev/test split

**Files:**
- Modify: `eval/gate_corroboration.py` (append)
- Modify: `tests/test_gate_corroboration.py` (append)

- [ ] **Step 1: Write the failing tests**

Append (extend import with `split_queries`):

```python
from eval.gate_corroboration import split_queries


def test_split_queries_is_deterministic_disjoint_and_covers_all():
    qids = [f"q{i}" for i in range(300)]
    dev1, test1 = split_queries(qids, seed=0)
    dev2, test2 = split_queries(list(reversed(qids)), seed=0)  # input order irrelevant
    assert (dev1, test1) == (dev2, test2)
    assert len(dev1) == len(test1) == 150
    assert set(dev1).isdisjoint(test1)
    assert set(dev1) | set(test1) == set(qids)


def test_split_queries_seed_changes_the_split():
    qids = [f"q{i}" for i in range(300)]
    assert split_queries(qids, seed=0) != split_queries(qids, seed=1)


def test_split_queries_dev_fraction():
    dev, test = split_queries([f"q{i}" for i in range(10)], seed=0, dev_fraction=0.3)
    assert len(dev) == 3 and len(test) == 7
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_gate_corroboration.py -v`
Expected: ImportError (`split_queries`)

- [ ] **Step 3: Write minimal implementation**

Append (add `import random` and `List`, `Tuple` to imports):

```python
def split_queries(
    qids: List[str], seed: int = 0, dev_fraction: float = 0.5
) -> Tuple[List[str], List[str]]:
    """Deterministic dev/test split: sort, seeded shuffle, cut at ``dev_fraction``.

    Sorting first makes the split a function of (qid set, seed) alone -- input
    order (dict iteration, file order) cannot change who lands in test.
    """
    ordered = sorted(qids)
    random.Random(seed).shuffle(ordered)
    n_dev = round(len(ordered) * dev_fraction)
    return ordered[:n_dev], ordered[n_dev:]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_gate_corroboration.py -v`
Expected: 14 PASS

- [ ] **Step 5: Commit**

```bash
git add eval/gate_corroboration.py tests/test_gate_corroboration.py
git commit -m "feat(eval): deterministic seeded dev/test query split"
```

---

### Task 4: Evaluate one config (per-query hits on a qid subset)

**Files:**
- Modify: `eval/gate_corroboration.py` (append)
- Modify: `tests/test_gate_corroboration.py` (append)

- [ ] **Step 1: Write the failing tests**

Append (extend import with `evaluate_config`):

```python
from eval.gate_corroboration import evaluate_config

# One fixable query: relevance ranks the counterfactual first; the needle holds a
# 2-vote consensus. And one zero-consensus query: all votes 0.
_REL = {"q1": {"cf1": 0.99, "n1": 0.5}, "q2": {"cf2": 0.99, "n2": 0.5}}
_COR = {"q1": {"cf1": 0.0, "n1": 2.0}, "q2": {"cf2": 0.0, "n2": 0.0}}
_NEEDLES = {"q1": "n1", "q2": "n2"}


def test_evaluate_config_alpha_one_is_pure_first_stage():
    hits = evaluate_config(_REL, _COR, _NEEDLES, ["q1", "q2"], "global", None, 1.0, k=1)
    assert hits == {"q1": 0.0, "q2": 0.0}  # counterfactual on top -> both missed


def test_evaluate_config_votes_gate_fixes_only_the_consensus_query():
    # tau=2: q1 (max votes 2) blends -> needle wins at alpha=0.2; q2 (no consensus)
    # stays pure relevance -> still missed.
    hits = evaluate_config(_REL, _COR, _NEEDLES, ["q1", "q2"], "votes", 2.0, 0.2, k=1)
    assert hits == {"q1": 1.0, "q2": 0.0}


def test_evaluate_config_restricts_to_the_given_qids():
    hits = evaluate_config(_REL, _COR, _NEEDLES, ["q1"], "global", None, 1.0, k=1)
    assert set(hits) == {"q1"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_gate_corroboration.py -v`
Expected: ImportError (`evaluate_config`)

- [ ] **Step 3: Write minimal implementation**

Append (add to the existing tune_corroboration import — this task introduces it: `from eval.tune_corroboration import per_query_hits`):

```python
def evaluate_config(
    relevance_run: Run,
    corroboration_run: Run,
    needles: Dict[str, str],
    qids: List[str],
    family: str,
    param: Optional[float],
    alpha: float,
    k: int,
) -> Dict[str, float]:
    """Per-query needle-found hits for ONE (family, param, alpha) config, restricted
    to ``qids`` (the dev or test half). The single definition of "score a config" --
    the sweep, the selection, and the test arms all go through here."""
    votes = max_votes_signal(corroboration_run)
    margins = margin_signal(relevance_run)
    mask = gate_mask(family, param, votes, margins)
    rel = {q: relevance_run[q] for q in qids}
    cor = {q: corroboration_run.get(q, {}) for q in qids}
    nee = {q: needles[q] for q in qids}
    return per_query_hits(gated_fuse(rel, cor, alpha, mask), nee, k)


def _mean(hits: Dict[str, float]) -> float:
    return sum(hits.values()) / len(hits) if hits else 0.0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_gate_corroboration.py -v`
Expected: 17 PASS

- [ ] **Step 5: Commit**

```bash
git add eval/gate_corroboration.py tests/test_gate_corroboration.py
git commit -m "feat(eval): evaluate_config - per-query hits for one gate config on a query subset"
```

---

### Task 5: Sweep all configs + structural sanity equivalences

**Files:**
- Modify: `eval/gate_corroboration.py` (append)
- Modify: `tests/test_gate_corroboration.py` (append)

- [ ] **Step 1: Write the failing tests**

Append (extend import with `CurveRow`, `sweep_gated`, `VOTE_TAUS`, `MARGIN_THRESHOLDS`):

```python
from eval.gate_corroboration import (
    MARGIN_THRESHOLDS,
    VOTE_TAUS,
    CurveRow,
    sweep_gated,
)


def _rows_by(rows, family, param=None):
    return {r.alpha: r for r in rows if r.family == family and r.param == param}


def test_sweep_covers_all_families_and_params():
    rows = sweep_gated(_REL, _COR, _NEEDLES, ["q1", "q2"], [0.0, 0.5, 1.0], k=1)
    families = {(r.family, r.param) for r in rows}
    assert ("global", None) in families
    assert all(("votes", t) in families for t in VOTE_TAUS)
    assert all(("margin", m) in families for m in MARGIN_THRESHOLDS)
    assert len(rows) == (1 + len(VOTE_TAUS) + len(MARGIN_THRESHOLDS)) * 3


def test_sweep_votes_tau1_equals_global_blend():
    # Zero-consensus queries have all-equal votes -> constant after minmax -> the
    # global blend is order-preserving on them, so gating them off (tau=1) changes
    # nothing: identical needle-found at every alpha (the spec's structural fact).
    rows = sweep_gated(_REL, _COR, _NEEDLES, ["q1", "q2"], [0.0, 0.2, 0.5, 1.0], k=1)
    global_curve = {a: r.score for a, r in _rows_by(rows, "global").items()}
    tau1_curve = {a: r.score for a, r in _rows_by(rows, "votes", 1.0).items()}
    assert tau1_curve == global_curve


def test_sweep_alpha_one_scores_equal_pure_first_stage_everywhere():
    rows = sweep_gated(_REL, _COR, _NEEDLES, ["q1", "q2"], [1.0], k=1)
    assert {r.score for r in rows} == {0.0}  # every family at alpha=1 = q2d = both missed


def test_sweep_n_gated_counts_gated_queries():
    rows = sweep_gated(_REL, _COR, _NEEDLES, ["q1", "q2"], [0.5], k=1)
    assert _rows_by(rows, "global")[0.5].n_gated == 2
    assert _rows_by(rows, "votes", 2.0)[0.5].n_gated == 1  # only q1 has votes >= 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_gate_corroboration.py -v`
Expected: ImportError (`CurveRow`)

- [ ] **Step 3: Write minimal implementation**

Append (add `NamedTuple` to the typing import):

```python
# Gate grids (spec section 4). tau=1 reproduces the global blend (structural fact,
# asserted in tests); the real hypothesis space is tau >= 2.
VOTE_TAUS = [1.0, 2.0, 3.0, 4.0]
MARGIN_THRESHOLDS = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30]


class CurveRow(NamedTuple):
    """One point of the dev sensitivity surface."""

    family: str
    param: Optional[float]
    alpha: float
    score: float
    n_gated: int


def sweep_gated(
    relevance_run: Run,
    corroboration_run: Run,
    needles: Dict[str, str],
    qids: List[str],
    alpha_grid: List[float],
    k: int,
) -> List[CurveRow]:
    """Score every (family, param, alpha) config on ``qids`` (the dev half).

    Pure arithmetic; publish the WHOLE surface (sensitivity artifact, no
    cherry-picking), selection happens separately in :func:`select_on_dev`.
    """
    votes = max_votes_signal(corroboration_run)
    margins = margin_signal(relevance_run)
    rows: List[CurveRow] = []
    for family, params in (
        ("global", [None]),
        ("votes", VOTE_TAUS),
        ("margin", MARGIN_THRESHOLDS),
    ):
        for param in params:
            mask = gate_mask(family, param, votes, margins)
            n_gated = sum(1 for q in qids if mask.get(q, False))
            for alpha in alpha_grid:
                hits = evaluate_config(
                    relevance_run, corroboration_run, needles, qids, family, param, alpha, k
                )
                rows.append(CurveRow(family, param, alpha, _mean(hits), n_gated))
    return rows
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_gate_corroboration.py -v`
Expected: 21 PASS

- [ ] **Step 5: Commit**

```bash
git add eval/gate_corroboration.py tests/test_gate_corroboration.py
git commit -m "feat(eval): full gate/alpha dev sweep with tau=1==global structural sanity check"
```

---

### Task 6: Dev selection with spec tie-breaks

**Files:**
- Modify: `eval/gate_corroboration.py` (append)
- Modify: `tests/test_gate_corroboration.py` (append)

- [ ] **Step 1: Write the failing tests**

Append (extend import with `select_on_dev`):

```python
from eval.gate_corroboration import select_on_dev


def test_select_prefers_higher_score_then_larger_alpha_then_stricter_gate():
    rows = [
        CurveRow("votes", 1.0, 0.2, 0.8, 10),
        CurveRow("votes", 1.0, 0.5, 0.8, 10),  # same score, larger alpha -> preferred
        CurveRow("votes", 2.0, 0.5, 0.8, 5),   # same score+alpha, stricter tau -> preferred
        CurveRow("votes", 3.0, 0.1, 0.7, 2),   # lower score -> ignored
        CurveRow("global", None, 0.6, 0.75, 20),
        CurveRow("margin", 0.05, 0.3, 0.8, 4),
        CurveRow("margin", 0.10, 0.3, 0.8, 8),  # same score+alpha, smaller m stricter -> 0.05 wins
    ]
    best, winner, best_gated = select_on_dev(rows)
    assert best["votes"] == CurveRow("votes", 2.0, 0.5, 0.8, 5)
    assert best["margin"] == CurveRow("margin", 0.05, 0.3, 0.8, 4)
    assert best["global"] == CurveRow("global", None, 0.6, 0.75, 20)
    # Across families: 0.8 beats global's 0.75; votes beats margin on the family order.
    assert winner == CurveRow("votes", 2.0, 0.5, 0.8, 5)
    assert best_gated == CurveRow("votes", 2.0, 0.5, 0.8, 5)


def test_select_across_family_tie_prefers_global_then_votes():
    rows = [
        CurveRow("global", None, 0.5, 0.8, 20),
        CurveRow("votes", 2.0, 0.5, 0.8, 5),
        CurveRow("margin", 0.05, 0.5, 0.8, 4),
    ]
    best, winner, best_gated = select_on_dev(rows)
    assert winner.family == "global"        # tie -> simpler wins
    assert best_gated.family == "votes"     # best GATED config still reported (spec 4)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_gate_corroboration.py -v`
Expected: ImportError (`select_on_dev`)

- [ ] **Step 3: Write minimal implementation**

Append:

```python
_FAMILY_ORDER = {"global": 0, "votes": 1, "margin": 2}  # ties -> simpler wins


def _strictness(row: CurveRow) -> float:
    """Larger = gate fires less often (spec tie-break: prefer the stricter gate)."""
    if row.family == "votes":
        return row.param
    if row.family == "margin":
        return -row.param
    return 0.0


def select_on_dev(
    rows: List[CurveRow],
) -> Tuple[Dict[str, CurveRow], CurveRow, CurveRow]:
    """``(best_per_family, overall_winner, best_gated)`` under the spec tie-breaks.

    Within a family: max score, ties to larger alpha (closer to pure relevance,
    matching ``tune_corroboration.best_alpha``), then to the stricter gate. Across
    families: max score, ties to the simpler family (global > votes > margin).
    ``best_gated`` is the winner among the votes/margin families only -- always
    certified on test so "gated vs global" is measured even when global wins dev.
    """
    best: Dict[str, CurveRow] = {}
    for family in _FAMILY_ORDER:
        candidates = [r for r in rows if r.family == family]
        best[family] = max(candidates, key=lambda r: (r.score, r.alpha, _strictness(r)))
    winner = max(
        best.values(), key=lambda r: (r.score, -_FAMILY_ORDER[r.family], r.alpha)
    )
    best_gated = max(
        (best["votes"], best["margin"]),
        key=lambda r: (r.score, -_FAMILY_ORDER[r.family], r.alpha),
    )
    return best, winner, best_gated
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_gate_corroboration.py -v`
Expected: 23 PASS

- [ ] **Step 5: Commit**

```bash
git add eval/gate_corroboration.py tests/test_gate_corroboration.py
git commit -m "feat(eval): dev-only selection with conservative tie-breaks (alpha, strictness, family)"
```

---

### Task 7: Flip table (descriptive, full set)

**Files:**
- Modify: `eval/gate_corroboration.py` (append)
- Modify: `tests/test_gate_corroboration.py` (append)

- [ ] **Step 1: Write the failing tests**

Append (extend import with `flip_status`, `flip_table`, `margin_bucket`, `vote_bucket`):

```python
from eval.gate_corroboration import flip_status, flip_table, margin_bucket, vote_bucket


def test_flip_status_four_way():
    assert flip_status(0.0, 1.0) == "fixed"
    assert flip_status(1.0, 0.0) == "broken"
    assert flip_status(1.0, 1.0) == "unchanged_hit"
    assert flip_status(0.0, 0.0) == "unchanged_miss"


def test_vote_bucket_edges():
    assert [vote_bucket(v) for v in (0.0, 1.0, 3.0, 4.0, 9.0)] == ["0", "1", "3", "4+", "4+"]


def test_margin_bucket_edges():
    assert [margin_bucket(m) for m in (0.0, 0.049, 0.05, 0.1, 0.19, 0.2, 0.9)] == [
        "<0.05", "<0.05", "0.05-0.1", "0.1-0.2", "0.1-0.2", ">=0.2", ">=0.2",
    ]


def test_flip_table_counts_one_fixed_one_broken_one_unchanged():
    # q1: blend fixes it (needle 2-vote consensus). q2: blend breaks it (needle on
    # top by relevance, counterfactual holds the consensus). q3: unchanged miss
    # (no consensus -> constant blend preserves order).
    rel = {
        "q1": {"cf1": 0.99, "n1": 0.5},
        "q2": {"n2": 0.99, "cf2": 0.5},
        "q3": {"cf3": 0.99, "n3": 0.5},
    }
    cor = {
        "q1": {"cf1": 0.0, "n1": 2.0},
        "q2": {"n2": 0.0, "cf2": 2.0},
        "q3": {"cf3": 0.0, "n3": 0.0},
    }
    needles = {"q1": "n1", "q2": "n2", "q3": "n3"}

    rows = flip_table(rel, cor, needles, alpha=0.2, k=1)

    votes_rows = {r["bucket"]: r for r in rows if r["signal"] == "max_votes"}
    assert votes_rows["2"]["fixed"] == 1 and votes_rows["2"]["broken"] == 1
    assert votes_rows["0"]["unchanged_miss"] == 1
    # every query lands in exactly one bucket per signal
    for signal in ("max_votes", "margin"):
        sig_rows = [r for r in rows if r["signal"] == signal]
        assert sum(
            r["fixed"] + r["broken"] + r["unchanged_hit"] + r["unchanged_miss"]
            for r in sig_rows
        ) == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_gate_corroboration.py -v`
Expected: ImportError (`flip_status`)

- [ ] **Step 3: Write minimal implementation**

Append (extend the tune_corroboration import to `from eval.tune_corroboration import load_runs, per_query_hits` — `load_runs` is used by `main` in Task 9; extend the fusion import with `convex_fuse`):

```python
_VOTE_BUCKETS = ["0", "1", "2", "3", "4+"]
_MARGIN_BUCKETS = ["<0.05", "0.05-0.1", "0.1-0.2", ">=0.2"]
_STATUSES = ["fixed", "broken", "unchanged_hit", "unchanged_miss"]


def vote_bucket(votes: float) -> str:
    return "4+" if votes >= 4 else str(int(votes))


def margin_bucket(margin: float) -> str:
    if margin < 0.05:
        return "<0.05"
    if margin < 0.1:
        return "0.05-0.1"
    if margin < 0.2:
        return "0.1-0.2"
    return ">=0.2"


def flip_status(base_hit: float, blended_hit: float) -> str:
    """fixed (miss->hit) / broken (hit->miss) / unchanged_hit / unchanged_miss."""
    if blended_hit and not base_hit:
        return "fixed"
    if base_hit and not blended_hit:
        return "broken"
    return "unchanged_hit" if base_hit else "unchanged_miss"


def flip_table(
    relevance_run: Run,
    corroboration_run: Run,
    needles: Dict[str, str],
    alpha: float,
    k: int,
) -> List[Dict[str, object]]:
    """Descriptive flip analysis on the FULL query set at one alpha (no tuning):
    who does the global blend fix/break, bucketed by each gate signal. The
    evidence for/against gating's premise; goes in the report either way."""
    votes = max_votes_signal(corroboration_run)
    margins = margin_signal(relevance_run)
    base = per_query_hits(relevance_run, needles, k)
    blended = per_query_hits(convex_fuse(relevance_run, corroboration_run, alpha), needles, k)
    statuses = {qid: flip_status(base[qid], blended[qid]) for qid in needles}
    rows: List[Dict[str, object]] = []
    for signal, sig_map, buckets, bucket_fn in (
        ("max_votes", votes, _VOTE_BUCKETS, vote_bucket),
        ("margin", margins, _MARGIN_BUCKETS, margin_bucket),
    ):
        counts = {b: {s: 0 for s in _STATUSES} for b in buckets}
        for qid in needles:
            counts[bucket_fn(sig_map[qid])][statuses[qid]] += 1
        for b in buckets:
            rows.append({"signal": signal, "bucket": b, **counts[b]})
    return rows
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_gate_corroboration.py -v`
Expected: 27 PASS

- [ ] **Step 5: Commit**

```bash
git add eval/gate_corroboration.py tests/test_gate_corroboration.py
git commit -m "feat(eval): descriptive flip table bucketed by gate signals"
```

---

### Task 8: CSV writers + CLI arg parsing

**Files:**
- Modify: `eval/gate_corroboration.py` (append)
- Modify: `tests/test_gate_corroboration.py` (append)

- [ ] **Step 1: Write the failing tests**

Append (extend import with `_parse_args`, `write_dev_curves`, `write_flip_table`):

```python
import csv

from eval.gate_corroboration import _parse_args, write_dev_curves, write_flip_table


def test_write_dev_curves_csv(tmp_path):
    rows = [CurveRow("global", None, 0.6, 0.58, 150), CurveRow("votes", 2.0, 0.5, 0.6, 80)]
    path = tmp_path / "curves.csv"
    write_dev_curves(rows, path, k=10)
    with open(path, newline="") as f:
        got = list(csv.reader(f))
    assert got[0] == ["family", "param", "alpha", "needle_found@10", "n_gated"]
    assert got[1] == ["global", "", "0.6", "0.58", "150"]  # global param is empty (spec 6)
    assert got[2] == ["votes", "2.0", "0.5", "0.6", "80"]


def test_write_flip_table_csv(tmp_path):
    rows = [
        {"signal": "max_votes", "bucket": "0", "fixed": 0, "broken": 1,
         "unchanged_hit": 2, "unchanged_miss": 3},
    ]
    path = tmp_path / "flips.csv"
    write_flip_table(rows, path)
    with open(path, newline="") as f:
        got = list(csv.reader(f))
    assert got[0] == ["signal", "bucket", "fixed", "broken", "unchanged_hit", "unchanged_miss"]
    assert got[1] == ["max_votes", "0", "0", "1", "2", "3"]


def test_parse_args_defaults_and_required_from_runs():
    args = _parse_args(["--from-runs", "results/runs.json"])
    assert args.from_runs.name == "runs.json"
    assert args.k == 10 and args.seed == 0
    assert args.alpha_step == 0.1 and args.flip_alpha == 0.6
    assert args.out_dir.name == "results"


def test_parse_args_requires_from_runs():
    with pytest.raises(SystemExit):
        _parse_args([])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_gate_corroboration.py -v`
Expected: ImportError (`_parse_args`)

- [ ] **Step 3: Write minimal implementation**

Append (add `import argparse`, `import csv`, `from pathlib import Path` to imports):

```python
def write_dev_curves(rows: List[CurveRow], path: Path, k: int) -> None:
    """The dev sensitivity surface CSV (spec 6): whole curves, no cherry-picking."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["family", "param", "alpha", f"needle_found@{k}", "n_gated"])
        for r in rows:
            param = "" if r.param is None else r.param
            writer.writerow([r.family, param, r.alpha, round(r.score, 4), r.n_gated])


def write_flip_table(rows: List[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["signal", "bucket", *_STATUSES])
        writer.writeheader()
        writer.writerows(rows)


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m eval.gate_corroboration",
        description="Offline gated dynamic-alpha analysis of the corroboration "
        "reranker from a runs dump: dev/test re-certification of the global blend "
        "+ gated-vs-global comparison + flip table. Pure arithmetic, no GPU.",
    )
    p.add_argument("--from-runs", type=Path, required=True, dest="from_runs",
                   help="Runs dump JSON from eval.tune_corroboration --dump-runs.")
    p.add_argument("--k", type=int, default=10, help="needle-found cut-off (default: %(default)s).")
    p.add_argument("--seed", type=int, default=0, help="dev/test split seed (default: %(default)s).")
    p.add_argument("--alpha-step", type=float, default=0.1, dest="alpha_step")
    p.add_argument("--flip-alpha", type=float, default=0.6, dest="flip_alpha",
                   help="Alpha for the descriptive flip table (default: the certified "
                        "alpha*=%(default)s).")
    p.add_argument("--out-dir", type=Path, default=Path("results"), dest="out_dir")
    return p.parse_args(argv)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_gate_corroboration.py -v`
Expected: 31 PASS

- [ ] **Step 5: Commit**

```bash
git add eval/gate_corroboration.py tests/test_gate_corroboration.py
git commit -m "feat(eval): CSV writers + CLI args for the offline gated analysis"
```

---

### Task 9: `main` — end-to-end from a synthetic dump

**Files:**
- Modify: `eval/gate_corroboration.py` (append)
- Modify: `tests/test_gate_corroboration.py` (append)

- [ ] **Step 1: Write the failing test**

Append (extend import with `main`; add `from eval.tune_corroboration import dump_runs` to the test imports):

```python
from eval.gate_corroboration import main
from eval.tune_corroboration import dump_runs


def _synthetic_dump(tmp_path):
    """4 identical fixable queries (any 2/2 split behaves the same): the lone
    counterfactual tops relevance; the needle holds a 2-vote consensus."""
    rel, cor, needles = {}, {}, {}
    for i in range(4):
        qid = f"q{i}"
        rel[qid] = {f"cf{i}": 0.99, f"n{i}": 0.5}
        cor[qid] = {f"cf{i}": 0.0, f"n{i}": 2.0}
        needles[qid] = f"n{i}"
    path = tmp_path / "runs.json"
    dump_runs(rel, cor, needles, path)
    return path


def test_main_end_to_end_writes_all_artifacts(tmp_path, capsys):
    runs = _synthetic_dump(tmp_path)
    out = tmp_path / "out"

    main(["--from-runs", str(runs), "--k", "1", "--out-dir", str(out)])

    # 1. all three artifacts exist
    per_query = out / "corroboration_gate_test_per_query.csv"
    assert (out / "corroboration_flip_table.csv").exists()
    assert (out / "corroboration_gate_dev_curves.csv").exists()
    assert per_query.exists()

    # 2. test arms: q2d misses everywhere (cf on top); blend + gated blend fix it.
    with open(per_query, newline="") as f:
        rows = list(csv.DictReader(f))
    assert set(rows[0]) == {"qid", "q2d", "global_corroborate", "gated_corroborate"}
    assert len(rows) == 2  # the test half of 4 queries
    assert all(r["q2d"] == "0.0" for r in rows)
    assert all(r["global_corroborate"] == "1.0" for r in rows)
    assert all(r["gated_corroborate"] == "1.0" for r in rows)

    # 3. summary names the winner and prints both significance commands
    printed = capsys.readouterr().out
    assert "winner" in printed and "eval.significance" in printed
    assert "--reference q2d" in printed and "--reference global_corroborate" in printed
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_gate_corroboration.py -v`
Expected: ImportError (`main`)

- [ ] **Step 3: Write minimal implementation**

Append (uses `load_runs` from the tune_corroboration import extended in Task 7; `write_per_query_csv` imported lazily inside `main`, the established pattern for keeping heavy `run_benchmark` imports out of module import time):

```python
def main(argv: Optional[List[str]] = None) -> None:
    from eval.run_benchmark import write_per_query_csv  # lazy: run_benchmark is heavy

    args = _parse_args(argv)
    relevance_run, corroboration_run, needles = load_runs(args.from_runs)

    # 1. Descriptive flip table on the FULL set (no tuning -- spec section 4).
    flips = flip_table(relevance_run, corroboration_run, needles, args.flip_alpha, args.k)
    write_flip_table(flips, args.out_dir / "corroboration_flip_table.csv")

    # 2. Split, sweep on dev, select on dev.
    dev_qids, test_qids = split_queries(list(needles), args.seed)
    rows = sweep_gated(
        relevance_run, corroboration_run, needles, dev_qids, _grid(args.alpha_step), args.k
    )
    write_dev_curves(rows, args.out_dir / "corroboration_gate_dev_curves.csv", args.k)
    best, winner, best_gated = select_on_dev(rows)

    # 3. Certify exactly three arms on the held-out test half (params frozen).
    arms = {
        "q2d": ("global", None, 1.0),
        "global_corroborate": ("global", None, best["global"].alpha),
        "gated_corroborate": (best_gated.family, best_gated.param, best_gated.alpha),
    }
    per_query = {
        name: evaluate_config(
            relevance_run, corroboration_run, needles, test_qids, fam, par, al, args.k
        )
        for name, (fam, par, al) in arms.items()
    }
    test_csv = args.out_dir / "corroboration_gate_test_per_query.csv"
    write_per_query_csv(per_query, test_csv)

    print(f"seed={args.seed}  dev={len(dev_qids)}  test={len(test_qids)}  k={args.k}")
    for family, row in best.items():
        print(f"  dev best [{family}]: param={row.param} alpha={row.alpha} "
              f"needle_found@{args.k}={row.score:.4f} n_gated={row.n_gated}")
    print(f"dev winner: {winner.family} (param={winner.param}, alpha={winner.alpha})")
    for name, (fam, par, al) in arms.items():
        print(f"  test {name}: needle_found@{args.k}={_mean(per_query[name]):.4f} "
              f"(family={fam}, param={par}, alpha={al})")
    print(f"wrote {test_csv} -> significance:")
    print(f"  python -m eval.significance --per-query-csv {test_csv} --reference q2d")
    print(f"  python -m eval.significance --per-query-csv {test_csv} --reference global_corroborate")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_gate_corroboration.py -v`
Expected: 32 PASS

Trace for the assertion values (why global/gated both land at 1.0): each query's
minmax gives cf=1.0/needle=0.0 on relevance and needle=1.0/cf=0.0 on votes, so the
blend scores are cf=alpha vs needle=1-alpha and the needle wins iff alpha<0.5; the
best dev rows are alpha in {0.0..0.4} at score 1.0, ties break to alpha=0.4. Global
and votes tie at 1.0 -> winner=global (family order), best_gated=votes with tau=2
(strictest tau that still fires: max_votes=2). The margin signal is 1.0 for every
query (cf=1.0, needle=0.0) so no margin gate ever fires -> margin family scores 0.

- [ ] **Step 5: Run the FULL suite to check nothing broke**

Run: `python -m pytest`
Expected: everything passes (427 passed + 32 new = 459 passed, 1 skipped — the
pre-existing skip; no new failures)

- [ ] **Step 6: Commit**

```bash
git add eval/gate_corroboration.py tests/test_gate_corroboration.py
git commit -m "feat(eval): gate_corroboration CLI - offline dev/test certification + gated-vs-global from the runs dump"
```

---

## After implementation (user-run, not part of this plan)

The real analysis needs `corroboration_runs_nq300cert.json` (job 18025180 artifact, on
the HPC). Either scp it to `results/` locally or run on the HPC login node (CPU,
seconds):

```bash
python -m eval.gate_corroboration --from-runs results/corroboration_runs_nq300cert.json
python -m eval.significance --per-query-csv results/corroboration_gate_test_per_query.csv --reference q2d
python -m eval.significance --per-query-csv results/corroboration_gate_test_per_query.csv --reference global_corroborate
```

Then paste the three test-arm numbers + p-values into `docs/results-summary.md`
(finding 15 update: de-caveated re-certification + the gating verdict).
