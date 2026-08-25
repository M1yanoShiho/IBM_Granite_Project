# Corroboration Reranking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a training-free reranker that ranks a candidate pool by cross-source answer *corroboration* (not query relevance), so a lone factually-wrong counterfactual distractor is demoted below the true needle.

**Architecture:** A pure scoring module (`corroboration.py`, no LLM/IO) + a `CorroborationReranker` (in `reranker.py`) that extracts each candidate's answer with the injected Granite LLM, scores corroboration, min-max blends it with first-stage relevance (`final = alpha·rel + (1-alpha)·corrob`, reusing `fusion.minmax_normalize`), and reorders. Registered as `q2d_corroborate` in `run_benchmark`, so it drops into `run_niah` unchanged. Design spec: `docs/superpowers/specs/2026-07-05-corroboration-reranking-design.md`.

**Tech Stack:** Python 3.12, pytest, existing `src/retrieval/{reranker,fusion,base}.py`, `eval/run_benchmark.py`. Reranker is duck-typed (same `rerank()` contract as `Reranker`/`LLMListwiseReranker`), reuses the shared `LLMClient` (`generate(prompt)->str`).

**Scope note:** This plan is the CORE method + registration only, runnable for the ablation `q2d_corroborate` vs cross-encoder vs listwise on `run_niah`. The mandatory diagnostics (tie-rate, counterfactual-demotion-rate) and the conditional entailment tie-breaker (§5 of the spec) are explicit follow-ups, out of scope here (see "After this plan").

---

## File Structure

- **Create** `src/retrieval/corroboration.py` — pure scoring: `normalize_answer`, `is_valid_answer`, `corroboration_scores`. No LLM, no IO (mirrors `fusion.py`).
- **Create** `tests/test_corroboration.py` — unit tests for the pure logic.
- **Modify** `src/retrieval/reranker.py` — add `CorroborationReranker` (LLM extraction + parametric + blend), beside `Reranker`/`LLMListwiseReranker`.
- **Create** `tests/test_corroboration_reranker.py` — reranker tests with a fake LLM.
- **Modify** `eval/run_benchmark.py` — `CORROBORATION_SPECS`, `retrievers_need_llm`, `_build_retrievers` branch, top import.
- **Modify** `tests/test_run_benchmark.py` — registration + need-llm tests.

---

## Task 1: Pure corroboration scoring

**Files:**
- Create: `src/retrieval/corroboration.py`
- Test: `tests/test_corroboration.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_corroboration.py`:

```python
"""Tests for the pure corroboration scoring (no LLM)."""
from src.retrieval.corroboration import (
    corroboration_scores,
    is_valid_answer,
    normalize_answer,
)


def test_normalize_strips_article_case_and_punctuation():
    assert normalize_answer("  The Origin. ") == "origin"
    assert normalize_answer("Paris") == "paris"
    assert normalize_answer("A Charles Darwin") == "charles darwin"


def test_is_valid_answer_rejects_degenerate_extractions():
    assert not is_valid_answer("the")      # stopword
    assert not is_valid_answer("NONE")     # sentinel non-answer
    assert not is_valid_answer("")         # empty
    assert not is_valid_answer("ab")       # too short, non-numeric
    assert is_valid_answer("12")           # short but numeric -> kept
    assert is_valid_answer("Charles Darwin")


def test_lone_answers_tie_at_zero():
    # needle X and counterfactual Y each appear once, rest NONE -> both 0 (the tie case)
    assert corroboration_scores(["Darwin", "Lamarck", "NONE"]) == [0.0, 0.0, 0.0]


def test_corroborated_answer_beats_lone_counterfactual():
    # X answered by two passages, Y by one -> each X gets 1 vote, Y gets 0
    assert corroboration_scores(["Darwin", "Darwin", "Lamarck"]) == [1.0, 1.0, 0.0]


def test_parametric_vote_breaks_a_tie_toward_its_answer():
    assert corroboration_scores(["Darwin", "Lamarck"], parametric="Darwin") == [1.0, 0.0]


def test_degenerate_extraction_does_not_corroborate():
    # a failed "the" extraction must not count or match; the two real X's still corroborate
    assert corroboration_scores(["the", "Darwin", "Darwin"]) == [0.0, 1.0, 1.0]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_corroboration.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.retrieval.corroboration'`.

- [ ] **Step 3: Write the minimal implementation**

Create `src/retrieval/corroboration.py`:

```python
"""Corroboration scoring for the corroboration reranker (pure — no LLM, no IO).

Scores a candidate by how many OTHER retrieved passages (plus the model's own
parametric answer) independently answer the query with the SAME entity. A lone
factually-wrong passage (a Source-A counterfactual: a relevant-but-wrong entity) is
corroborated by nobody, so it is demoted even though it is semantically relevant --
the signal a relevance reranker cannot provide. Design spec:
docs/superpowers/specs/2026-07-05-corroboration-reranking-design.md.
"""
from __future__ import annotations

import re
from typing import List, Optional

# A too-weak extraction (e.g. a failed "the") must not corroborate everything. Non-numeric
# answers shorter than this (after normalisation) are ignored; short numbers are kept.
_MIN_ANSWER_LEN = 3
_STOPWORDS = {"the", "a", "an", "none", "n/a", "unknown", "it", "yes", "no"}


def normalize_answer(answer: str) -> str:
    """Lowercase, strip a leading article, and strip surrounding punctuation/space."""
    s = answer.strip().lower()
    s = re.sub(r"^(the|a|an)\s+", "", s)
    return s.strip(" \t\n.,;:!?\"'()[]")


def is_valid_answer(answer: str) -> bool:
    """True if ``answer`` counts as an entity vote (not empty/stopword/too-short)."""
    s = normalize_answer(answer)
    if not s or s in _STOPWORDS:
        return False
    if len(s) < _MIN_ANSWER_LEN and not s.isdigit():
        return False
    return True


def corroboration_scores(
    answers: List[str], parametric: Optional[str] = None
) -> List[float]:
    """Raw count of OTHER sources that answer with the same entity, per candidate.

    ``answers[i]`` is candidate ``i``'s extracted answer (or a non-answer like "NONE").
    ``parametric`` is the model's own answer (one extra voter) or ``None``. Each valid
    ``answers[i]`` scores the number of *other* candidates with an equal normalised
    answer, +1 if ``parametric`` matches. Invalid answers score 0. Raw counts — the
    reranker min-max normalises them before blending with relevance.
    """
    norms = [normalize_answer(a) if is_valid_answer(a) else None for a in answers]
    p = normalize_answer(parametric) if (parametric and is_valid_answer(parametric)) else None
    scores: List[float] = []
    for i, ni in enumerate(norms):
        if ni is None:
            scores.append(0.0)
            continue
        votes = sum(1 for j, nj in enumerate(norms) if j != i and nj == ni)
        if p is not None and p == ni:
            votes += 1
        scores.append(float(votes))
    return scores
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_corroboration.py -q`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add src/retrieval/corroboration.py tests/test_corroboration.py
git commit -m "feat(retrieval): corroboration_scores - pure cross-source answer-voting for the corroboration reranker (degenerate-answer guard, parametric voter)"
```

---

## Task 2: CorroborationReranker

**Files:**
- Modify: `src/retrieval/reranker.py` (add class + two prompt constants + imports)
- Test: `tests/test_corroboration_reranker.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_corroboration_reranker.py`:

```python
"""Tests for CorroborationReranker (LLM answer-extraction + corroboration blend)."""
from src.retrieval.base import RetrievedChunk
from src.retrieval.reranker import CorroborationReranker


class FakeAnswerLLM:
    """Extracts an answer per passage via a text->answer map; parametric via a fixed
    reply. The extraction prompt contains 'Passage:'; the parametric prompt does not."""

    def __init__(self, answer_by_text, parametric="NONE"):
        self.answer_by_text = answer_by_text
        self.parametric = parametric

    def generate(self, prompt: str) -> str:
        if "Passage:" not in prompt:            # parametric elicitation
            return self.parametric
        for text, ans in self.answer_by_text.items():
            if text in prompt:
                return ans
        return "NONE"


def test_corroborated_needle_outranks_lone_counterfactual():
    # The counterfactual has the HIGHEST relevance but a lone (wrong) answer; the needle
    # and another gold share the correct answer, so corroboration lifts them above it.
    cand = [
        RetrievedChunk("cf", "counterfactual passage", 0.99),   # answer Berlin (lone)
        RetrievedChunk("needle", "needle passage", 0.90),       # answer Paris
        RetrievedChunk("gold2", "second gold passage", 0.80),   # answer Paris (corroborates)
    ]
    llm = FakeAnswerLLM({
        "counterfactual passage": "Berlin",
        "needle passage": "Paris",
        "second gold passage": "Paris",
    })
    rr = CorroborationReranker(llm, top_n=3, alpha=0.5, use_parametric=False)

    out = [c.doc_id for c in rr.rerank("capital of France?", cand, top_k=3)]

    assert out[0] in {"needle", "gold2"}         # a corroborated (correct) passage leads
    assert out.index("needle") < out.index("cf")  # needle beat its counterfactual twin


def test_alpha_one_is_pure_first_stage_order():
    cand = [RetrievedChunk("a", "alpha passage", 0.9), RetrievedChunk("b", "bravo passage", 0.5)]
    llm = FakeAnswerLLM({"alpha passage": "Paris", "bravo passage": "Paris"})
    rr = CorroborationReranker(llm, top_n=2, alpha=1.0, use_parametric=False)

    out = [c.doc_id for c in rr.rerank("q", cand, top_k=2)]

    assert out == ["a", "b"]                     # alpha=1 ignores corroboration entirely


def test_top_n_caps_extraction_and_keeps_tail_below():
    cand = [
        RetrievedChunk("a", "alpha passage", 0.9),
        RetrievedChunk("b", "bravo passage", 0.8),
        RetrievedChunk("c", "charlie passage", 0.7),   # beyond top_n -> tail
    ]
    llm = FakeAnswerLLM({"alpha passage": "Paris", "bravo passage": "NONE"})
    rr = CorroborationReranker(llm, top_n=2, alpha=0.0, use_parametric=False)

    out = [c.doc_id for c in rr.rerank("q", cand, top_k=3)]

    assert out[-1] == "c"                        # the un-scored tail stays below the window


def test_empty_pool_returns_empty():
    rr = CorroborationReranker(FakeAnswerLLM({}), top_n=5)
    assert rr.rerank("q", [], top_k=3) == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_corroboration_reranker.py -q`
Expected: FAIL — `ImportError: cannot import name 'CorroborationReranker'`.

- [ ] **Step 3: Add the imports and prompt constants**

In `src/retrieval/reranker.py`, the current top imports are:

```python
from src.retrieval.base import RetrievedChunk, Retriever
```

Replace that line with:

```python
from src.retrieval.base import RetrievedChunk, Retriever
from src.retrieval.corroboration import corroboration_scores
from src.retrieval.fusion import minmax_normalize
```

Also change the typing import at the top of the file from:

```python
from typing import List, Sequence
```
to:
```python
from typing import List, Optional, Sequence
```

- [ ] **Step 4: Append the `CorroborationReranker` class**

Add to the END of `src/retrieval/reranker.py`:

```python
_EXTRACT_PROMPT = (
    "Using ONLY the passage below, answer the question with the shortest exact answer "
    "(a name, place, date, or number). If the passage does not answer it, reply NONE.\n"
    "Question: {question}\n"
    "Passage: {passage}\n"
    "Answer:"
)
_PARAMETRIC_PROMPT = (
    "Answer the question with the shortest exact answer from your own knowledge. "
    "If you are not sure, reply NONE.\n"
    "Question: {question}\n"
    "Answer:"
)


class CorroborationReranker:
    """Rerank by cross-source answer corroboration instead of query relevance.

    Extracts each candidate's answer with the LLM, scores how many OTHER candidates
    (plus the model's own parametric answer) give the same answer
    (:func:`~src.retrieval.corroboration.corroboration_scores`), then blends that --
    min-max normalised -- with the min-max normalised first-stage relevance:
    ``final = alpha * relevance + (1 - alpha) * corroboration`` (``alpha`` is the
    relevance weight, like the hybrid's dense weight; ``alpha = 1`` -> plain first
    stage). A lone counterfactual is relevant but uncorroborated, so it falls below the
    corroborated needle -- the separation a relevance reranker cannot make.

    Same ``rerank(query, candidates, top_k)`` contract as :class:`Reranker`, so
    :class:`TwoStageRetriever` wraps it unchanged. Cost = ``top_n`` (+1 parametric) LLM
    calls/query; only the first ``top_n`` candidates are re-scored, the rest keep their
    first-stage order below. Design: 2026-07-05-corroboration-reranking-design.md.
    """

    def __init__(
        self,
        llm,
        top_n: int = 20,
        alpha: float = 0.5,
        use_parametric: bool = True,
        passage_chars: int = 600,
    ) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1]; got {alpha}.")
        if top_n < 1:
            raise ValueError("top_n must be at least 1.")
        self.llm = llm
        self.top_n = top_n
        self.alpha = alpha
        self.use_parametric = use_parametric
        self.passage_chars = passage_chars

    def _extract_answer(self, query: str, passage: str) -> str:
        prompt = _EXTRACT_PROMPT.format(question=query, passage=passage[: self.passage_chars])
        return self.llm.generate(prompt).strip()

    def _parametric_answer(self, query: str) -> Optional[str]:
        if not self.use_parametric:
            return None
        return self.llm.generate(_PARAMETRIC_PROMPT.format(question=query)).strip()

    def rerank(
        self, query: str, candidates: List[RetrievedChunk], top_k: int
    ) -> List[RetrievedChunk]:
        """Re-score the first ``top_n`` candidates by relevance+corroboration; the rest
        keep their order below. Returns the top ``top_k`` chunks (blended score)."""
        if not candidates:
            return []
        window = candidates[: self.top_n]
        answers = [self._extract_answer(query, c.text) for c in window]
        corr = corroboration_scores(answers, self._parametric_answer(query))
        rel_norm = minmax_normalize({i: c.score for i, c in enumerate(window)})
        cor_norm = minmax_normalize({i: corr[i] for i in range(len(window))})
        final = [
            self.alpha * rel_norm[i] + (1.0 - self.alpha) * cor_norm[i]
            for i in range(len(window))
        ]
        # Stable sort: ties keep first-stage order (Python sort is stable under reverse).
        order = sorted(range(len(window)), key=lambda i: final[i], reverse=True)
        reranked = [
            RetrievedChunk(window[i].doc_id, window[i].text, float(final[i]))
            for i in order
        ]
        return (reranked + candidates[self.top_n :])[:top_k]
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/test_corroboration_reranker.py -q`
Expected: PASS (4 passed).

- [ ] **Step 6: Commit**

```bash
git add src/retrieval/reranker.py tests/test_corroboration_reranker.py
git commit -m "feat(retrieval): CorroborationReranker - rerank by relevance+corroboration blend (minmax convex, alpha=relevance weight), reusing fusion.minmax_normalize"
```

---

## Task 3: Register `q2d_corroborate` in run_benchmark

**Files:**
- Modify: `eval/run_benchmark.py`
- Test: `tests/test_run_benchmark.py`

- [ ] **Step 1: Write the failing tests**

Add to the END of `tests/test_run_benchmark.py`:

```python
def test_build_retrievers_builds_q2d_corroborate_reranker(monkeypatch) -> None:
    # q2d_corroborate = q2d query-transform first stage + CorroborationReranker, as one
    # TwoStageRetriever, reusing the injected LLM (no second model load).
    from src.retrieval.query_transform import TransformingRetriever
    from src.retrieval.reranker import CorroborationReranker, TwoStageRetriever

    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer", FakeSentenceTransformer
    )

    class InjectedLLM:
        def generate(self, prompt):
            return "granite retrieval passage"

    injected = InjectedLLM()
    data = BenchmarkData(
        corpus={"d1": "granite retrieval", "d2": "banana cake"},
        queries={"q1": "granite"},
        qrels={"q1": {"d1": 1}},
    )
    config = BenchmarkConfig(retrievers=["q2d_corroborate"], k_values=[1])

    retrievers = _build_retrievers(config, data, llm=injected)

    rr = retrievers["q2d_corroborate"]
    assert isinstance(rr, TwoStageRetriever)
    assert isinstance(rr.reranker, CorroborationReranker)   # the corroboration reranker
    assert isinstance(rr.retriever, TransformingRetriever)  # q2d first stage
    assert rr.reranker.llm is injected                      # reuse, not a second load
    assert rr.retrieve("granite")[0].doc_id == "d1"


def test_retrievers_need_llm_flags_corroborate() -> None:
    from eval.run_benchmark import retrievers_need_llm

    assert retrievers_need_llm(["q2d_corroborate"])
    assert not retrievers_need_llm(["granite_dense", "splade"])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_run_benchmark.py -k "corroborate" -q`
Expected: FAIL — `KeyError: 'q2d_corroborate'` (unknown retriever) / assertion on `retrievers_need_llm`.

- [ ] **Step 3: Add `CorroborationReranker` to the reranker import**

In `eval/run_benchmark.py` the current reranker import is:

```python
from src.retrieval.reranker import (
    DEFAULT_RERANKER_MODEL_ID,
    LLMListwiseReranker,
    Reranker,
    TwoStageRetriever,
)
```

Add `CorroborationReranker` to it:

```python
from src.retrieval.reranker import (
    CorroborationReranker,
    DEFAULT_RERANKER_MODEL_ID,
    LLMListwiseReranker,
    Reranker,
    TwoStageRetriever,
)
```

- [ ] **Step 4: Add the spec dict and extend `retrievers_need_llm`**

In `eval/run_benchmark.py`, immediately AFTER the `LLM_RERANK_SPECS = { ... }` block and BEFORE `def retrievers_need_llm`, insert:

```python
# Corroboration reranker: name -> first-stage retriever name. A
# :class:`~src.retrieval.reranker.CorroborationReranker` reranks the first stage's pool
# by cross-source answer corroboration (not relevance), to demote lone counterfactual
# distractors. Reuses the injected LLM.
CORROBORATION_SPECS: Dict[str, str] = {
    "q2d_corroborate": "q2d_granite",
}
```

Then in `def retrievers_need_llm`, change the first condition from:

```python
        if name in HYDE_SPECS or name in LLM_RERANK_SPECS:
            return True
```
to:
```python
        if name in HYDE_SPECS or name in LLM_RERANK_SPECS or name in CORROBORATION_SPECS:
            return True
```

- [ ] **Step 5: Add the build branch in `_build_retrievers`**

In `eval/run_benchmark.py`, in `_build_retrievers`, the loop has an `elif name in LLM_RERANK_SPECS:` branch that ends by assigning `retrievers[name] = TwoStageRetriever(first, LLMListwiseReranker(client), ...)`. Immediately AFTER that `elif` block (and before the final `else:`), insert:

```python
        elif name in CORROBORATION_SPECS:
            first = _build_named(
                CORROBORATION_SPECS[name], config, data, corpus, doc_ids, top_k, llm=llm
            )
            from src.llm_client import LLMClient

            client = llm if llm is not None else LLMClient()
            retrievers[name] = TwoStageRetriever(
                first,
                CorroborationReranker(client),
                top_k=pool,
                candidates=config.rerank_pool or pool,
            )
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `python -m pytest tests/test_run_benchmark.py -k "corroborate" -q`
Expected: PASS (2 passed).

- [ ] **Step 7: Run the full affected suites to check for regressions**

Run: `python -m pytest tests/test_run_benchmark.py tests/test_corroboration.py tests/test_corroboration_reranker.py -q`
Expected: PASS (no failures).

- [ ] **Step 8: Commit**

```bash
git add eval/run_benchmark.py tests/test_run_benchmark.py
git commit -m "feat(niah): register q2d_corroborate - CorroborationReranker over the q2d first stage; retrievers_need_llm gate + shared-LLM wiring so it drops into run_niah"
```

---

## After this plan (out of scope — separate work)

1. **Run the ablation (HPC)** — on the frozen 300q task, add `q2d_corroborate` beside the existing arms:
   `sbatch scripts/run_niah.slurm results/niah_nq300_frozen.json nq_corrob 10 granite_dense q2d_granite q2d_corroborate granite_rerank granite_listrank`
   → the "corroboration beats relevance-reranking" table (needle-found@10 + MRR + significance).
2. **λ (alpha) tuning** — sweep `alpha` offline via the `eval/tune_alpha.py` pattern; pick on dev, report the curve.
3. **Mandatory diagnostics (spec §7)** — tie-rate + counterfactual-demotion-rate, computed from the per-query output + the task's `parent_needle_id`.
4. **Conditional entailment tie-breaker (spec §5)** — only if diagnostics show answer-conflicting ties dominate.
5. **Source-A hardening (spec §9)** — a strictly separate PR, sequenced after the core result (variable control).

---

## Self-Review

**Spec coverage:**
- §3–§4 method (extract → corroboration → normalise+convex blend) → Tasks 1–2. ✓
- §4 degenerate-answer guard → Task 1 `is_valid_answer` + `test_degenerate_extraction_does_not_corroborate`. ✓
- §3 scale-fix (min-max both signals, `alpha` convex) → Task 2 (`minmax_normalize` + blend) + `test_alpha_one_is_pure_first_stage_order`. ✓
- §6 integration (`CorroborationReranker`, `TwoStageRetriever`, `q2d_corroborate` spec, `retrievers_need_llm`) → Tasks 2–3. ✓
- §5 tie-breaker, §7 diagnostics, §9 hardening → explicitly deferred ("After this plan"). ✓ (in-scope core is self-contained)

**Placeholder scan:** No TBD/TODO; every code step shows full code and exact commands. ✓

**Type consistency:** `corroboration_scores(answers, parametric)` defined in Task 1 and called with that signature in Task 2. `CorroborationReranker(llm, top_n, alpha, use_parametric, passage_chars)` defined in Task 2 and constructed with the injected `client` (defaults) in Task 3. `rerank(query, candidates, top_k)` matches the `Reranker` contract `TwoStageRetriever` calls. `minmax_normalize` used with an index-keyed dict (valid — it is doc-id-agnostic). ✓
