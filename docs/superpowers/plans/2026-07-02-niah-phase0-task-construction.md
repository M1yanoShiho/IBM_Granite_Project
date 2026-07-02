# NIAH Phase 0 — Task Construction & Scale Probe — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Phase-0 critical-path deliverables for the NIAH reorientation — a literature-grounded *distractor construction pipeline* (spec §5.1) that turns an existing IR/QA benchmark into a hard "needle-in-a-haystack" task, plus an HPC *scale-feasibility probe* — so Phases 1–5 have a non-saturated task and an honest corpus-size ceiling to build on.

**Architecture:** A new `src/niah/` package of small, single-responsibility, injectable-model units (three distractor *sources* → two *filters* → *assembly* → *hardness gate*), composed by an `eval/build_niah_task.py` CLI into a persisted `NiahTask`. Everything reuses the existing contracts (`Retriever`, `RetrievedChunk`, `LLMClient`, `load_benchmark`) so the built task drops into the existing eval harness unchanged. Pure logic is TDD'd with hand-checked values; model-dependent logic is TDD'd with injected fakes (the established pattern — cf. `FakeSentenceTransformer`, `test_fusion.py`).

**Tech Stack:** Python 3.12, pytest, `ir_datasets` (via `eval/benchmarks/loader.py`), `transformers` (`LLMClient`), the project's dense/sparse retrievers + Granite reranker, Slurm (BluePebble HPC).

**Scope note (required by writing-plans scope check).** Phases 0–5 of the spec are **not one subsystem**, and Phases 1–5 consume Phase 0's *outputs* (the constructed task, the probed ceiling). Writing bite-sized steps for them now would require inventing test inputs that do not yet exist. Therefore **this plan details Phase 0 only**; Phases 1–5 appear as a milestone roadmap (bottom) with entry criteria, each to be expanded into its own plan once Phase 0 lands.

---

## File structure (Phase 0)

Create (new `src/niah/` package + flat tests, matching the repo's `tests/test_*.py` convention):

| File | Responsibility |
|---|---|
| `src/niah/__init__.py` | package marker |
| `src/niah/types.py` | dataclasses: `Distractor`, `NiahExample`, `NiahTask` |
| `src/niah/counterfactual.py` | **Source A** — type-consistent entity swap (pure `swap_entity` + LLM `make_counterfactual`) |
| `src/niah/generative.py` | **Source B** — SyNeg-style LLM plausible non-answer (`make_generative_distractor`) |
| `src/niah/mining.py` | **Source C** — topical hard negatives via dense∪sparse top-k (`mine_topical`) |
| `src/niah/filters.py` | **Filter 1** answerability + positive-anchor (`passes_margin`, `answers_query`); **Filter 2** dual-retriever hardness (`is_hard`); `keep_distractor` compose |
| `src/niah/assembly.py` | inject distractors + gold-preserving corpus cap (`inject`, `cap_haystack`) |
| `src/niah/hardness_gate.py` | pure `recall_at_k`, `mean_recall`, `is_saturated`; `gate_report` |
| `eval/build_niah_task.py` | CLI: loader → sources → filters → assembly → gate → persist `NiahTask` JSON |
| `scripts/run_scale_probe.slurm` | HPC feasibility probe: index build-time + peak RSS at `max_docs` ∈ {1M,5M,21M} |
| `docs/niah-task-definition.md` | the written needle/haystack/hay/metrics definition + supervisor-align checklist |
| `tests/test_niah_types.py`, `tests/test_niah_counterfactual.py`, `tests/test_niah_generative.py`, `tests/test_niah_mining.py`, `tests/test_niah_filters.py`, `tests/test_niah_assembly.py`, `tests/test_niah_hardness_gate.py`, `tests/test_build_niah_task.py` | tests |

**Task order & shippable slices.** Tasks 1→6 are the **MVP task builder** (Source A + both filters + assembly + gate + CLI) — independently produces a hard task and unblocks Phase 1. Tasks 7–8 add Sources B & C (richness). Task 9 is the scale probe (parallelisable — no code dependency on 1–8). Task 10 is the written definition + supervisor sign-off (the non-code Phase-0 gate).

**Conventions.**
- Run one test: `python -m pytest tests/test_niah_<x>.py -v`
- Run the suite: `python -m pytest -q`
- A "fake LLM" is any object with `generate(prompt: str) -> str`; a "fake retriever" is any object with `retrieve(query: str) -> List[RetrievedChunk]`. Inject them; never load a real model in a unit test.

---

## Task 1: Shared types

**Files:**
- Create: `src/niah/__init__.py` (empty)
- Create: `src/niah/types.py`
- Test: `tests/test_niah_types.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_niah_types.py
"""Tests for src/niah/types.py — the NIAH task dataclasses."""
from __future__ import annotations

from src.niah.types import Distractor, NiahExample, NiahTask


def test_distractor_carries_source_and_parent() -> None:
    d = Distractor(doc_id="q7__cf0", text="...", source="counterfactual", parent_needle_id="d42")
    assert d.source == "counterfactual"
    assert d.parent_needle_id == "d42"


def test_niahexample_holds_needles_and_distractors() -> None:
    d = Distractor(doc_id="q7__cf0", text="wrong", source="counterfactual", parent_needle_id="d42")
    ex = NiahExample(query_id="q7", query="who?", needle_ids=["d42"], distractors=[d])
    assert ex.needle_ids == ["d42"]
    assert ex.distractors[0].doc_id == "q7__cf0"


def test_niahtask_qrels_mark_only_needles_relevant() -> None:
    task = NiahTask(
        corpus={"d42": "gold", "q7__cf0": "wrong"},
        queries={"q7": "who?"},
        qrels={"q7": {"d42": 1}},
        examples=[],
    )
    # distractor ids must NOT appear in qrels (they are non-relevant by construction)
    assert "q7__cf0" not in task.qrels["q7"]
    assert task.qrels["q7"]["d42"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_niah_types.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.niah'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/niah/__init__.py
```
```python
# src/niah/types.py
"""Dataclasses for a constructed Needle-In-A-Haystack retrieval task.

A NIAH task is an ordinary (corpus, queries, qrels) benchmark — so it runs on the
existing eval harness unchanged — plus the provenance of the injected distractors.
Invariant: only *needles* (the gold docs from the source qrels) are relevant;
every injected ``Distractor`` is non-relevant by construction and must never be
written into ``qrels`` (see src/niah/filters.py for how that is enforced).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

# Distractor provenance labels (also the three §5.1 sources).
SOURCE_COUNTERFACTUAL = "counterfactual"
SOURCE_GENERATIVE = "generative"
SOURCE_MINED = "mined"


@dataclass(frozen=True)
class Distractor:
    """One injected non-relevant passage and where it came from."""

    doc_id: str
    text: str
    source: str  # one of SOURCE_*
    parent_needle_id: str


@dataclass
class NiahExample:
    """One query with its needle(s) and the distractors built for it."""

    query_id: str
    query: str
    needle_ids: List[str]
    distractors: List[Distractor] = field(default_factory=list)


@dataclass
class NiahTask:
    """A built task: a runnable benchmark + distractor provenance."""

    corpus: Dict[str, str]
    queries: Dict[str, str]
    qrels: Dict[str, Dict[str, int]]
    examples: List[NiahExample] = field(default_factory=list)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_niah_types.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/niah/__init__.py src/niah/types.py tests/test_niah_types.py
git commit -m "feat(niah): task dataclasses (Distractor/NiahExample/NiahTask)"
```

---

## Task 2: Source A — counterfactual entity swap

**Files:**
- Create: `src/niah/counterfactual.py`
- Test: `tests/test_niah_counterfactual.py`

Source A is the sharpest dual-adversarial distractor: a type-consistent but factually wrong entity swap keeps lexical overlap near-maximal (fools BM25/SPLADE) and the embedding near-identical (fools dense) while changing the answer. Split into a **pure** swap (fully testable) and an **LLM** wrong-entity proposer (testable with a fake).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_niah_counterfactual.py
"""Tests for src/niah/counterfactual.py — Source A (entity substitution)."""
from __future__ import annotations

import pytest

from src.niah.counterfactual import make_counterfactual, propose_wrong_entity, swap_entity


def test_swap_entity_replaces_all_occurrences() -> None:
    text = "Linda Davis won in 1994. Linda Davis was the artist."
    assert swap_entity(text, "Linda Davis", "Mary Jones") == (
        "Mary Jones won in 1994. Mary Jones was the artist."
    )


def test_swap_entity_is_case_sensitive_exact() -> None:
    # Only exact-surface occurrences are swapped (avoids corrupting substrings).
    assert swap_entity("Paris and paris", "Paris", "Rome") == "Rome and paris"


def test_swap_entity_no_match_returns_unchanged() -> None:
    assert swap_entity("no entity here", "Xyz", "Abc") == "no entity here"


class _FakeLLM:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.reply


def test_propose_wrong_entity_parses_llm_reply() -> None:
    llm = _FakeLLM("  Mary Jones\n")
    assert propose_wrong_entity("Linda Davis", llm) == "Mary Jones"
    # the answer is shown to the LLM so it can pick a *type-consistent* alternative
    assert "Linda Davis" in llm.prompts[0]


def test_propose_wrong_entity_rejects_echo() -> None:
    # If the model returns the same entity, that is not a counterfactual -> error,
    # so the caller can skip this needle rather than inject a false negative.
    llm = _FakeLLM("Linda Davis")
    with pytest.raises(ValueError, match="same entity"):
        propose_wrong_entity("Linda Davis", llm)


def test_make_counterfactual_swaps_answer_in_passage() -> None:
    llm = _FakeLLM("Mary Jones")
    out = make_counterfactual("Linda Davis won the 1994 award.", "Linda Davis", llm)
    assert out == "Mary Jones won the 1994 award."
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_niah_counterfactual.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.niah.counterfactual'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/niah/counterfactual.py
"""Source A — counterfactual entity substitution (spec §5.1).

Given a needle passage and its answer entity, ask an LLM for a *type-consistent
but different* entity, then swap it in. The result reads coherently, shares
almost all tokens with the needle (adversarial to BM25/SPLADE) and embeds almost
identically (adversarial to dense), yet no longer answers the query.

Method lineage: entity-based knowledge conflicts (Longpre et al., 2021) /
Faithfulness-QA / WikiContradict.
"""
from __future__ import annotations

_WRONG_ENTITY_PROMPT = (
    "Replace the following answer with a DIFFERENT but same-type, equally plausible "
    "entity (same category: person/place/date/number/organisation). "
    "Reply with ONLY the replacement, nothing else.\n"
    "Answer: {answer}\n"
    "Replacement:"
)


def swap_entity(text: str, old: str, new: str) -> str:
    """Replace every exact-surface occurrence of ``old`` with ``new`` (pure)."""
    return text.replace(old, new)


def propose_wrong_entity(answer: str, llm) -> str:
    """Ask ``llm`` for a type-consistent but different entity than ``answer``."""
    reply = llm.generate(_WRONG_ENTITY_PROMPT.format(answer=answer)).strip()
    if not reply:
        raise ValueError("LLM returned an empty replacement entity.")
    if reply.lower() == answer.strip().lower():
        raise ValueError("LLM returned the same entity; not a counterfactual.")
    return reply


def make_counterfactual(needle_text: str, answer: str, llm) -> str:
    """Build a counterfactual distractor passage from a needle + its answer."""
    wrong = propose_wrong_entity(answer, llm)
    return swap_entity(needle_text, answer, wrong)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_niah_counterfactual.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add src/niah/counterfactual.py tests/test_niah_counterfactual.py
git commit -m "feat(niah): Source A counterfactual entity substitution"
```

---

## Task 3: Filters — answerability + positive-anchor + dual-retriever hardness

**Files:**
- Create: `src/niah/filters.py`
- Test: `tests/test_niah_filters.py`

The two mandatory filters (spec §5.1). Filter 1 protects **eval validity** (a distractor that actually answers the query is a false negative that corrupts recall — NV-Retriever finds ~70% of naive negatives are false). Filter 2 keeps only distractors hard to *both* retriever families.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_niah_filters.py
"""Tests for src/niah/filters.py — Filter 1 (answerability/positive-anchor) + Filter 2 (hardness)."""
from __future__ import annotations

from src.niah.filters import answers_query, is_hard, keep_distractor, passes_margin


def test_passes_margin_keeps_clearly_below_positive() -> None:
    # candidate must be < positive - margin to be a safe (true) negative
    assert passes_margin(cand_score=0.40, positive_score=0.90, margin=0.05) is True


def test_passes_margin_rejects_near_positive() -> None:
    assert passes_margin(cand_score=0.88, positive_score=0.90, margin=0.05) is False


class _FakeJudge:
    def __init__(self, verdict: str) -> None:
        self.verdict = verdict

    def generate(self, prompt: str) -> str:
        return self.verdict


def test_answers_query_true_when_judge_says_yes() -> None:
    assert answers_query("Paris is the capital.", "capital of France?", _FakeJudge("YES")) is True


def test_answers_query_false_when_judge_says_no() -> None:
    assert answers_query("Rome is the capital.", "capital of France?", _FakeJudge("NO")) is False


def test_is_hard_requires_top_rank_in_both_runs() -> None:
    dense_rank = {"cf0": 2, "cf1": 50}
    sparse_rank = {"cf0": 3, "cf1": 1}
    assert is_hard("cf0", dense_rank, sparse_rank, rank_threshold=10) is True   # top-10 in both
    assert is_hard("cf1", dense_rank, sparse_rank, rank_threshold=10) is False  # 50 in dense


def test_is_hard_false_when_absent_from_a_run() -> None:
    assert is_hard("cf9", {"cf9": 1}, {}, rank_threshold=10) is False


def test_keep_distractor_applies_both_filters() -> None:
    # safe negative (passes margin, judge says NO) AND hard in both -> keep
    assert keep_distractor(
        cand_score=0.4, positive_score=0.9, margin=0.05,
        cand_text="Rome is the capital.", query="capital of France?", judge=_FakeJudge("NO"),
        cand_id="cf0", dense_rank={"cf0": 1}, sparse_rank={"cf0": 2}, rank_threshold=10,
    ) is True


def test_keep_distractor_drops_answer_leaking_candidate() -> None:
    # judge says YES (it answers the query) -> dropped even though it is hard
    assert keep_distractor(
        cand_score=0.4, positive_score=0.9, margin=0.05,
        cand_text="Paris is the capital.", query="capital of France?", judge=_FakeJudge("YES"),
        cand_id="cf0", dense_rank={"cf0": 1}, sparse_rank={"cf0": 2}, rank_threshold=10,
    ) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_niah_filters.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.niah.filters'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/niah/filters.py
"""The two mandatory distractor filters (spec §5.1).

Filter 1 — false-negative / answerability: a distractor must NOT answer the query.
  * positive-anchor score threshold (NV-Retriever TopK-MarginPos): keep only
    candidates scored clearly below the true needle (``passes_margin``);
  * an explicit answerability judge (cross-encoder or LLM) as the semantic backstop
    (``answers_query``).
Filter 2 — dual-retriever hardness: keep only candidates ranked highly by BOTH a
  dense and a sparse retriever (``is_hard``).
"""
from __future__ import annotations

from typing import Dict

_ANSWERABILITY_PROMPT = (
    "Does the passage directly answer the question? Reply ONLY 'YES' or 'NO'.\n"
    "Question: {query}\n"
    "Passage: {passage}\n"
    "Answer:"
)


def passes_margin(cand_score: float, positive_score: float, margin: float) -> bool:
    """True if the candidate scores far enough below the gold to be a safe negative."""
    return cand_score < positive_score - margin


def answers_query(passage: str, query: str, judge) -> bool:
    """True if ``judge`` (any ``generate``-able) says the passage answers the query."""
    verdict = judge.generate(_ANSWERABILITY_PROMPT.format(query=query, passage=passage))
    return verdict.strip().upper().startswith("YES")


def is_hard(
    cand_id: str,
    dense_rank: Dict[str, int],
    sparse_rank: Dict[str, int],
    rank_threshold: int,
) -> bool:
    """True if ``cand_id`` is within top-``rank_threshold`` in BOTH rank maps."""
    d = dense_rank.get(cand_id)
    s = sparse_rank.get(cand_id)
    if d is None or s is None:
        return False
    return d <= rank_threshold and s <= rank_threshold


def keep_distractor(
    *,
    cand_score: float,
    positive_score: float,
    margin: float,
    cand_text: str,
    query: str,
    judge,
    cand_id: str,
    dense_rank: Dict[str, int],
    sparse_rank: Dict[str, int],
    rank_threshold: int,
) -> bool:
    """Apply Filter 1 (margin AND not-answering) then Filter 2 (hard in both)."""
    if not passes_margin(cand_score, positive_score, margin):
        return False
    if answers_query(cand_text, query, judge):
        return False
    return is_hard(cand_id, dense_rank, sparse_rank, rank_threshold)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_niah_filters.py -v`
Expected: PASS (8 passed)

- [ ] **Step 5: Commit**

```bash
git add src/niah/filters.py tests/test_niah_filters.py
git commit -m "feat(niah): distractor filters (answerability + positive-anchor + dual-retriever hardness)"
```

---

## Task 4: Assembly — inject distractors + gold-preserving corpus cap

**Files:**
- Create: `src/niah/assembly.py`
- Test: `tests/test_niah_assembly.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_niah_assembly.py
"""Tests for src/niah/assembly.py — haystack assembly + scale capping."""
from __future__ import annotations

from src.niah.assembly import cap_haystack, inject
from src.niah.types import Distractor


def test_inject_adds_distractor_docs() -> None:
    corpus = {"d1": "gold"}
    ds = [Distractor(doc_id="d1__cf0", text="wrong", source="counterfactual", parent_needle_id="d1")]
    out = inject(corpus, ds)
    assert out == {"d1": "gold", "d1__cf0": "wrong"}
    assert corpus == {"d1": "gold"}  # pure: input not mutated


def test_cap_haystack_always_keeps_needles_and_distractors() -> None:
    corpus = {f"bg{i}": "hay" for i in range(100)}
    corpus["gold1"] = "needle"
    corpus["gold1__cf0"] = "distractor"
    out = cap_haystack(
        corpus, keep_ids={"gold1", "gold1__cf0"}, max_docs=10, seed=0
    )
    assert "gold1" in out and "gold1__cf0" in out
    assert len(out) == 12  # 10 background + the 2 kept


def test_cap_haystack_is_deterministic_under_seed() -> None:
    corpus = {f"bg{i}": "hay" for i in range(100)}
    a = cap_haystack(corpus, keep_ids=set(), max_docs=5, seed=42)
    b = cap_haystack(corpus, keep_ids=set(), max_docs=5, seed=42)
    assert a == b


def test_cap_haystack_none_keeps_everything() -> None:
    corpus = {f"bg{i}": "hay" for i in range(10)}
    assert cap_haystack(corpus, keep_ids=set(), max_docs=None, seed=0) == corpus
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_niah_assembly.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.niah.assembly'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/niah/assembly.py
"""Assemble the haystack: inject distractors and cap corpus size for the scale axis.

``cap_haystack`` mirrors ``eval.benchmarks.loader.load_benchmark(max_docs=...)``:
needles and injected distractors are ALWAYS kept; the remainder is filled with a
deterministic random sample of background docs, so recall stays well-defined while
the corpus size is swept (Phase 1).
"""
from __future__ import annotations

import random
from typing import Dict, Iterable, Optional, Set

from src.niah.types import Distractor


def inject(corpus: Dict[str, str], distractors: Iterable[Distractor]) -> Dict[str, str]:
    """Return a new corpus with the distractor docs added (input not mutated)."""
    out = dict(corpus)
    for d in distractors:
        out[d.doc_id] = d.text
    return out


def cap_haystack(
    corpus: Dict[str, str],
    keep_ids: Set[str],
    max_docs: Optional[int],
    seed: int,
) -> Dict[str, str]:
    """Keep all ``keep_ids`` + up to ``max_docs`` deterministically-sampled others."""
    if max_docs is None:
        return dict(corpus)
    kept = {doc_id: corpus[doc_id] for doc_id in keep_ids if doc_id in corpus}
    background = sorted(doc_id for doc_id in corpus if doc_id not in keep_ids)
    rng = random.Random(seed)
    rng.shuffle(background)
    for doc_id in background[:max_docs]:
        kept[doc_id] = corpus[doc_id]
    return kept
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_niah_assembly.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/niah/assembly.py tests/test_niah_assembly.py
git commit -m "feat(niah): haystack assembly + gold-preserving corpus cap"
```

---

## Task 5: Hardness gate — non-saturation check

**Files:**
- Create: `src/niah/hardness_gate.py`
- Test: `tests/test_niah_hardness_gate.py`

The task-level gate: after injection, baseline recall must be well below saturation, else the task measures nothing. Pure recall maths (self-contained so the gate has no hidden dependency; if `eval/ir_metrics.py` exposes an identical `recall_at_k`, a later refactor may DRY them).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_niah_hardness_gate.py
"""Tests for src/niah/hardness_gate.py — the non-saturation gate."""
from __future__ import annotations

from src.niah.hardness_gate import gate_report, is_saturated, mean_recall, recall_at_k


def test_recall_at_k_counts_found_gold_in_top_k() -> None:
    ranked = ["d3", "d1", "d9"]          # retriever output, best-first
    assert recall_at_k(ranked, {"d1"}, k=3) == 1.0
    assert recall_at_k(ranked, {"d1"}, k=1) == 0.0     # d1 is at rank 2
    assert recall_at_k(ranked, {"d1", "d9"}, k=3) == 1.0


def test_recall_at_k_no_gold_is_zero() -> None:
    assert recall_at_k(["d1", "d2"], set(), k=3) == 0.0


def test_mean_recall_averages_per_query() -> None:
    per_q = {"q1": 1.0, "q2": 0.0}
    assert mean_recall(per_q) == 0.5


def test_is_saturated_true_above_threshold() -> None:
    assert is_saturated(0.97, threshold=0.95) is True
    assert is_saturated(0.80, threshold=0.95) is False


def test_gate_report_flags_pass_or_fail() -> None:
    rep = gate_report({"q1": 0.6, "q2": 0.4}, threshold=0.95)
    assert rep["mean_recall"] == 0.5
    assert rep["saturated"] is False
    assert rep["passes_gate"] is True   # NOT saturated -> the task is hard enough
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_niah_hardness_gate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.niah.hardness_gate'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/niah/hardness_gate.py
"""Task-level hardness gate (spec §5.1): baselines must NOT saturate.

If a baseline retriever already finds the needle almost always (mean recall above
``threshold``), the constructed task is too easy and measures nothing — raise the
distractor ratio/hardness and rebuild.
"""
from __future__ import annotations

from typing import Dict, List, Set


def recall_at_k(ranked_ids: List[str], gold_ids: Set[str], k: int) -> float:
    """Fraction of gold ids present in the top-``k`` retrieved ids."""
    if not gold_ids:
        return 0.0
    topk = set(ranked_ids[:k])
    return len(topk & gold_ids) / len(gold_ids)


def mean_recall(per_query: Dict[str, float]) -> float:
    """Mean of per-query recall values (0.0 if empty)."""
    return sum(per_query.values()) / len(per_query) if per_query else 0.0


def is_saturated(mean_recall_value: float, threshold: float = 0.95) -> bool:
    """True if mean recall is at/above the saturation ``threshold``."""
    return mean_recall_value >= threshold


def gate_report(per_query: Dict[str, float], threshold: float = 0.95) -> dict:
    """Summarise the gate: mean recall, saturated?, and pass (= not saturated)."""
    mr = mean_recall(per_query)
    saturated = is_saturated(mr, threshold)
    return {"mean_recall": mr, "saturated": saturated, "passes_gate": not saturated}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_niah_hardness_gate.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add src/niah/hardness_gate.py tests/test_niah_hardness_gate.py
git commit -m "feat(niah): task-level non-saturation hardness gate"
```

---

## Task 6: CLI — compose Source A + filters + assembly into a persisted task

**Files:**
- Create: `eval/build_niah_task.py`
- Test: `tests/test_build_niah_task.py`

Composes the MVP builder: for each query, take gold docs as needles, build a counterfactual distractor per needle (Source A), keep those that pass both filters, inject, and emit a `NiahTask` + JSON. Retrievers/LLM/judge are **injected** so the wiring is unit-testable with fakes; `main()` wires the real ones.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_build_niah_task.py
"""Tests for eval/build_niah_task.py — task-builder wiring (with fakes)."""
from __future__ import annotations

from src.niah.types import NiahTask
from eval.build_niah_task import build_task


class _FakeLLM:
    def generate(self, prompt: str) -> str:
        # wrong-entity proposer: return a fixed alternative; answerability judge: 'NO'
        return "NO" if prompt.strip().endswith("Answer:") else "Mary Jones"


def _rank(ids):
    return {doc_id: i + 1 for i, doc_id in enumerate(ids)}


def test_build_task_makes_counterfactual_distractor_and_keeps_qrels_clean() -> None:
    corpus = {"d1": "Linda Davis won the 1994 award."}
    queries = {"q1": "who won the 1994 award?"}
    qrels = {"q1": {"d1": 1}}
    answers = {"q1": ["Linda Davis"]}
    # both retrievers rank the distractor highly (hard); scores below the gold
    dense_rank = {"q1": _rank(["d1", "q1__d1__cf0"])}
    sparse_rank = {"q1": _rank(["d1", "q1__d1__cf0"])}
    scores = {"q1": {"q1__d1__cf0": 0.4}}

    task = build_task(
        corpus=corpus, queries=queries, qrels=qrels, answers=answers,
        llm=_FakeLLM(), judge=_FakeLLM(),
        dense_rank=dense_rank, sparse_rank=sparse_rank, cand_scores=scores,
        positive_score=0.9, margin=0.05, rank_threshold=10,
    )

    assert isinstance(task, NiahTask)
    assert "q1__d1__cf0" in task.corpus                 # distractor injected
    assert task.corpus["q1__d1__cf0"] == "Mary Jones won the 1994 award."
    assert "q1__d1__cf0" not in task.qrels["q1"]        # distractor NOT relevant
    assert task.qrels["q1"] == {"d1": 1}                # needle still the only gold
    assert task.examples[0].distractors[0].source == "counterfactual"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_build_niah_task.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'eval.build_niah_task'`

- [ ] **Step 3: Write minimal implementation**

```python
# eval/build_niah_task.py
"""Build a hard NIAH task from a source benchmark (spec §5.1, MVP = Source A).

For each query: gold docs = needles; make one counterfactual distractor per needle
(Source A), keep those passing Filter 1 + Filter 2, inject them, and return a
``NiahTask`` whose qrels still mark ONLY the needles relevant.

``build_task`` takes precomputed ranks/scores + injected models so it is unit
testable; ``main`` (below) computes them from the real retrievers.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from src.niah.assembly import inject
from src.niah.counterfactual import make_counterfactual
from src.niah.filters import keep_distractor
from src.niah.types import SOURCE_COUNTERFACTUAL, Distractor, NiahExample, NiahTask


def build_task(
    *,
    corpus: Dict[str, str],
    queries: Dict[str, str],
    qrels: Dict[str, Dict[str, int]],
    answers: Dict[str, List[str]],
    llm,
    judge,
    dense_rank: Dict[str, Dict[str, int]],
    sparse_rank: Dict[str, Dict[str, int]],
    cand_scores: Dict[str, Dict[str, float]],
    positive_score: float,
    margin: float,
    rank_threshold: int,
) -> NiahTask:
    examples: List[NiahExample] = []
    all_distractors: List[Distractor] = []

    for qid, query in queries.items():
        needle_ids = list(qrels.get(qid, {}))
        gold_answer = (answers.get(qid) or [None])[0]
        ex = NiahExample(query_id=qid, query=query, needle_ids=needle_ids)
        if gold_answer:
            for nid in needle_ids:
                cand_id = f"{qid}__{nid}__cf0"
                try:
                    text = make_counterfactual(corpus[nid], gold_answer, llm)
                except (ValueError, KeyError):
                    continue  # skip needles we cannot safely counterfactual
                score = cand_scores.get(qid, {}).get(cand_id, 0.0)
                if keep_distractor(
                    cand_score=score, positive_score=positive_score, margin=margin,
                    cand_text=text, query=query, judge=judge,
                    cand_id=cand_id,
                    dense_rank=dense_rank.get(qid, {}),
                    sparse_rank=sparse_rank.get(qid, {}),
                    rank_threshold=rank_threshold,
                ):
                    d = Distractor(cand_id, text, SOURCE_COUNTERFACTUAL, nid)
                    ex.distractors.append(d)
                    all_distractors.append(d)
        examples.append(ex)

    return NiahTask(
        corpus=inject(corpus, all_distractors),
        queries=dict(queries),
        qrels={q: dict(r) for q, r in qrels.items()},
        examples=examples,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_build_niah_task.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Run the whole suite (nothing regressed)**

Run: `python -m pytest -q`
Expected: PASS — the previously-green suite plus the new NIAH tests.

- [ ] **Step 6: Commit**

```bash
git add eval/build_niah_task.py tests/test_build_niah_task.py
git commit -m "feat(niah): build_task CLI wiring (Source A + filters + assembly)"
```

> **MVP slice complete.** Tasks 1–6 produce a hard NIAH task from Source A alone. Tasks 7–8 add Sources B & C; Task 9 (probe) and Task 10 (definition/sign-off) are parallel.

---

## Task 7: Source B — LLM plausible non-answer (SyNeg-style)

**Files:**
- Create: `src/niah/generative.py`
- Test: `tests/test_niah_generative.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_niah_generative.py
"""Tests for src/niah/generative.py — Source B (LLM plausible non-answer)."""
from __future__ import annotations

import pytest

from src.niah.generative import make_generative_distractor


class _FakeLLM:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.reply


def test_make_generative_distractor_returns_passage() -> None:
    llm = _FakeLLM("The 1994 ceremony was held in Nashville and drew a large crowd.")
    out = make_generative_distractor("who won the 1994 award?", "Linda Davis won it.", llm)
    assert "Nashville" in out
    # multi-attribute prompt shows the query terms + the instruction to NOT answer
    assert "who won the 1994 award?" in llm.prompts[0]
    assert "not answer" in llm.prompts[0].lower()


def test_make_generative_distractor_rejects_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        make_generative_distractor("q?", "needle", _FakeLLM("   "))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_niah_generative.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.niah.generative'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/niah/generative.py
"""Source B — LLM-generated plausible non-answers (spec §5.1).

Multi-attribute self-reflection prompting (SyNeg, 2024) + MCQ-distractor design:
write an on-topic passage that shares the query's terms/entities and is highly
plausible, but does NOT answer the query. Filtered downstream by the same
answerability check (src/niah/filters.py) as Source A.
"""
from __future__ import annotations

_GENERATIVE_PROMPT = (
    "Write a short, fluent passage that is on the same topic as the query and "
    "shares its key terms and entities, BUT does not answer the query (it should "
    "discuss adjacent facts only). Reply with ONLY the passage.\n"
    "Query: {query}\n"
    "A real answer passage (for style reference, do NOT reuse its answer):\n{needle}\n"
    "Passage:"
)


def make_generative_distractor(query: str, needle_text: str, llm) -> str:
    """Generate a plausible, on-topic passage that does not answer ``query``."""
    out = llm.generate(_GENERATIVE_PROMPT.format(query=query, needle=needle_text)).strip()
    if not out:
        raise ValueError("LLM returned an empty distractor passage.")
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_niah_generative.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/niah/generative.py tests/test_niah_generative.py
git commit -m "feat(niah): Source B LLM plausible non-answer distractor"
```

> **Wire-in step (same commit or follow-up):** in `eval/build_niah_task.py`, after the Source-A block, add an analogous block calling `make_generative_distractor` with `cand_id = f"{qid}__{nid}__gen0"` and `source=SOURCE_GENERATIVE`, gated by the same `keep_distractor`. Add a test mirroring Task 6 asserting a `gen0` distractor appears and stays out of qrels.

---

## Task 8: Source C — topical hard-negative mining (dense ∪ sparse)

**Files:**
- Create: `src/niah/mining.py`
- Test: `tests/test_niah_mining.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_niah_mining.py
"""Tests for src/niah/mining.py — Source C (topical hard-negative mining)."""
from __future__ import annotations

from src.retrieval.base import RetrievedChunk
from src.niah.mining import mine_topical


class _FakeRetriever:
    def __init__(self, ids):
        self._ids = ids

    def retrieve(self, query: str):
        return [RetrievedChunk(doc_id=i, text=f"t{i}", score=1.0) for i in self._ids]


def test_mine_topical_unions_both_retrievers_minus_needles() -> None:
    dense = _FakeRetriever(["d1", "n1", "d2"])   # n1 is a needle -> excluded
    sparse = _FakeRetriever(["d2", "d3"])
    out = mine_topical("q", dense, sparse, k=3, exclude_ids={"n1"})
    assert set(out) == {"d1", "d2", "d3"}


def test_mine_topical_respects_k_per_retriever() -> None:
    dense = _FakeRetriever(["d1", "d2", "d3", "d4"])
    sparse = _FakeRetriever([])
    out = mine_topical("q", dense, sparse, k=2, exclude_ids=set())
    assert set(out) == {"d1", "d2"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_niah_mining.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.niah.mining'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/niah/mining.py
"""Source C — topical hard-negative mining (spec §5.1).

Union of the top-``k`` doc ids from a dense and a sparse retriever (covering both
retriever families), minus the needle ids. These "natural" distractors still pass
through Filter 1 (answerability) before injection, since a mined neighbour may in
fact be relevant (NV-Retriever: ~70% of naive top negatives are false negatives).
"""
from __future__ import annotations

from typing import List, Set


def mine_topical(query: str, dense, sparse, k: int, exclude_ids: Set[str]) -> List[str]:
    """Top-``k`` doc ids from dense ∪ sparse, excluding needles, order-preserving."""
    seen: Set[str] = set()
    out: List[str] = []
    for retriever in (dense, sparse):
        for chunk in retriever.retrieve(query)[:k]:
            if chunk.doc_id in exclude_ids or chunk.doc_id in seen:
                continue
            seen.add(chunk.doc_id)
            out.append(chunk.doc_id)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_niah_mining.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/niah/mining.py tests/test_niah_mining.py
git commit -m "feat(niah): Source C topical hard-negative mining"
```

> **Wire-in step:** in `eval/build_niah_task.py`, add a Source-C block turning each mined id into a `Distractor(doc_id=mined_id, source=SOURCE_MINED, ...)` (text = `corpus[mined_id]`), gated by Filter 1 (mined docs already exist in the corpus, so Filter 2's rank check is implicit). Mirror-test it.

---

## Task 9: Scale-feasibility probe (HPC) — build time + peak memory vs corpus size

**Files:**
- Create: `scripts/run_scale_probe.slurm`
- (Reuses `eval/run_benchmark.py --max-docs` + `eval/benchmarks/loader.py`; no new Python unless `--max-docs` is absent — see Step 1.)

This answers the spec §7 "scale ceiling = TBD" question: at what corpus size does index build become infeasible on the allocation? Complements the existing `scripts/run_scale_demo.slurm` (which measures recall vs latency) by measuring **build time + peak RSS** at `max_docs` ∈ {1M, 5M, 21M}.

- [ ] **Step 1: Verify the harness accepts a corpus cap**

Run: `python -m eval.run_benchmark --help`
Expected: a `--max-docs` (or equivalent) flag exists and is forwarded to `load_benchmark(max_docs=...)`.
- If present: proceed to Step 2.
- If absent: add `--max-docs` to `eval/run_benchmark.py`'s arg parser and pass it into `load_benchmark` (one-line change; add a parser test in `tests/test_run_benchmark.py` mirroring the existing parse tests), commit, then proceed.

- [ ] **Step 2: Write the probe script**

```bash
# scripts/run_scale_probe.slurm
#!/bin/bash
# SCALE-FEASIBILITY probe: how index build time + peak memory grow with corpus
# size, to set the honest "scale ceiling" for the NIAH headline (spec §7).
# Sweeps max_docs and records /usr/bin/time -v (Elapsed + Maximum resident set).
#
#   mkdir -p logs && sbatch scripts/run_scale_probe.slurm nq dev
# Pre-fetch model + dataset on the login node first (see run_scale_demo.slurm header).
#SBATCH --job-name=niah-scale-probe
#SBATCH --account=coms039904
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
DATASET="${1:?usage: sbatch scripts/run_scale_probe.slurm <dataset> [split]}"
SPLIT="${2:-test}"

module load languages/python/3.12.3
export HF_HOME=/user/work/$USER/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export IR_DATASETS_HOME=/user/work/$USER/ir_datasets PYTHONUNBUFFERED=1
source /user/work/$USER/venv/bin/activate
cd /user/work/$USER/IBM_Granite_Project

for N in 1000000 5000000 21000000; do
  echo "=================== max_docs=$N ==================="
  /usr/bin/time -v python -m eval.run_benchmark \
      --dataset "$DATASET" --split "$SPLIT" \
      --retrievers granite_dense \
      --index-type hnsw --chunk-unit token --max-docs "$N" \
      --out "results/scale_probe_${DATASET}_${N}.csv" \
      --cache-dir /user/work/$USER/index_cache \
    2>&1 | grep -E "Elapsed|Maximum resident|max_docs" || true
done
echo "Record Elapsed + Maximum resident set per max_docs -> the scale ceiling."
```

- [ ] **Step 3: Shell-lint the script**

Run: `bash -n scripts/run_scale_probe.slurm`
Expected: no output (syntax OK). Do NOT run the sbatch locally — it is HPC-only.

- [ ] **Step 4: Commit**

```bash
git add scripts/run_scale_probe.slurm
git commit -m "feat(niah): HPC scale-feasibility probe (build time + peak RSS vs corpus size)"
```

- [ ] **Step 5 (manual, on HPC): run and record.** Submit on BluePebble; record Elapsed + peak RSS per `max_docs` into `docs/niah-task-definition.md` §Scale-ceiling. The largest size that builds within the allocation is the honest ceiling for the headline.

---

## Task 10: Written task definition + supervisor-alignment gate (non-code)

**Files:**
- Create: `docs/niah-task-definition.md`

This is the Phase-0 gate the spec calls the critical path: the definition must be agreed with the supervisor before Phases 1–5 invest.

- [ ] **Step 1: Write `docs/niah-task-definition.md`** covering, concretely (fill with the choices made during Tasks 1–9):
  - **Needle** — the source benchmark(s) and query subset (e.g. NQ/TriviaQA dev on dpr-w100; the rare-entity subset criteria); what counts as gold (source qrels).
  - **Haystack** — corpus + how it scales (`cap_haystack` / `--max-docs`, the sweep sizes from Task 9); the injection ratio (distractors per needle).
  - **Misleading hay** — the three sources (Tasks 2/7/8) + the two filters (Task 3) + parameters actually used (`margin`, `rank_threshold`, judge model).
  - **Metrics** — recall@k / nDCG on needles; needle-found rate; `context_precision`; (Phase 3) citation-correctness + abstention; significance via `eval/significance.py`.
  - **Hardness-gate result** — the measured baseline mean-recall on the built task (Task 5 `gate_report`), demonstrating non-saturation.
  - **Scale ceiling** — the Task-9 numbers and the chosen honest maximum.
- [ ] **Step 2: Supervisor sign-off (manual gate).** Confirm the re-anchor + this definition with the supervisor (changes the Meeting-1 locked headline). Record the decision date in the doc.
- [ ] **Step 3: Commit**

```bash
git add docs/niah-task-definition.md
git commit -m "docs(niah): task definition + supervisor-alignment gate"
```

---

## Self-review (writing-plans checklist)

**Spec coverage (§5.1):** Source A → Task 2; Source B → Task 7; Source C → Task 8; Filter 1 (answerability + positive-anchor) → Task 3; Filter 2 (dual-retriever hardness) → Task 3; assembly/injection + scale cap → Task 4; task-level hardness gate → Task 5; CLI compose → Task 6; scale-feasibility probe (§7 ceiling) → Task 9; written definition + supervisor gate (§6.0 Phase 0) → Task 10. RGB-style counterfactual-robustness reuse is a Phase-3 item (roadmap below), correctly out of Phase 0.

**Placeholder scan:** no "TBD/TODO" in code steps; every code step shows complete code; the only deferred content is Task 7/8 "wire-in" notes (explicit, with ids/sources named) and Task 10 (an inherently prose deliverable, itemised).

**Type consistency:** `Distractor(doc_id, text, source, parent_needle_id)`, `NiahExample(query_id, query, needle_ids, distractors)`, `NiahTask(corpus, queries, qrels, examples)`, `SOURCE_*` constants, and function signatures (`make_counterfactual(needle_text, answer, llm)`, `keep_distractor(**kw)`, `mine_topical(query, dense, sparse, k, exclude_ids)`, `cap_haystack(corpus, keep_ids, max_docs, seed)`, `gate_report(per_query, threshold)`, `build_task(**kw)`) are used identically across Task 6 and the tests.

---

## Phases 1–5 — milestone roadmap (each becomes its own plan after Phase 0)

Not yet task-level: each consumes Phase 0 outputs and will be expanded once they exist.

- **Phase 1 — Scale axis / star (Weeks 2–3).** *Entry:* built `NiahTask` + scale ceiling (Task 9). Sweep corpus size with `cap_haystack` → recall/precision **degradation curve** (headline figure) + rare-needle recall. Reuses `eval/run_benchmark.py`, `eval/significance.py`, `scripts/run_scale_demo.slurm` (flat vs HNSW).
- **Phase 2 — Rarity + distractors (Weeks 4–5).** *Entry:* Phase 1 curve. Vary injection ratio/source mix; re-evaluate **reranking** (`Reranker`, `LLMListwiseReranker`) + **SPLADE/BM25** under distractor pressure → do the prior "hybrid/rerank don't help" nulls flip? Significance vs the no-distractor baseline.
- **Phase 3 — Trust + adaptivity + RAG-vs-long-context (Weeks 6–7).** *Entry:* a distractor-hardened task. Citation/attribution (`src/explainability/citations.py`); **recalibrate** `CorrectiveRAGPipeline._confidence` (spec §9 degeneracy) + abstention on unanswerable/counterfactual contexts (RGB-style, reuses Source A); retrieve-vs-stuff **crossover** via `eval/niah_runner.py` + `src/data_processing.py`.
- **Phase 4 — System packaging + efficiency (Week 8).** *Entry:* results across Phases 1–3. Package the demo-able needle-finder (`src/rag_app.py`); efficiency numbers (granite-small, latency/memory at the scale ceiling).
- **Phase 5 — Report + buffer (Week 9).** Write-up, figures, slippage buffer.
