# Complementarity-Aware Coverage Selection Implementation Plan (A2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Implement the coverage layer from `docs/superpowers/specs/2026-07-21-complementarity-coverage-selection-design.md`: deterministic salient-feature extraction, greedy set-cover selection, a `GatedCoverageSelector` that packs the gate's survivors by coverage, and config registration.

**Architecture:** The drop gate (gated.py) is unchanged and still governs removal. This layer replaces the survivors' truncation with pin-then-greedy-coverage packing over deterministic query+winning-cluster features. No NLI, no absolute threshold. Contract unchanged (variable-count SelectionResult already legal).

**Tech Stack:** Python 3.11, pydantic v2, pytest, mypy strict, ruff. No new deps. Run tests with `PYTHONPATH=src python -m pytest`.

---

## File Structure

| File | Responsibility |
|---|---|
| Create `src/evidence_rag/selector/coverage.py` | `salient_features()` + `coverage_select()` |
| Modify `src/evidence_rag/selector/gated.py` | extract `_gate()` helper returning survivors + metadata; add `GatedCoverageSelector` |
| Modify `src/evidence_rag/composition.py` | register `gated-coverage-corroboration` |
| Tests | `tests/selector/test_coverage.py`, `tests/selector/test_gated_coverage.py`, `tests/pipeline/test_selector_registration.py` (extend) |

---

### Task 1: salient_features

**Files:** Create `src/evidence_rag/selector/coverage.py`; Test `tests/selector/test_coverage.py`

- [ ] **Step 1.1: Write failing tests**

```python
from evidence_rag.selector.coverage import salient_features


def test_numbers_are_canonicalized_features() -> None:
    feats = salient_features("Revenue was $1.2B in 2025.")
    assert "num:1200000000" in feats
    assert "num:2025" in feats


def test_number_aliases_collapse_to_same_feature() -> None:
    assert salient_features("1,200 million") == salient_features("$1.2B")


def test_content_words_become_features_stopwords_excluded() -> None:
    feats = salient_features("The operating margin of IBM")
    assert "tok:operating" in feats
    assert "tok:margin" in feats
    assert "tok:ibm" in feats
    assert "tok:the" not in feats


def test_pure_digits_do_not_duplicate_as_tokens() -> None:
    feats = salient_features("18%")
    assert feats == frozenset({"num:18"})


def test_deterministic() -> None:
    text = "IBM 2025 operating margin was 18 percent"
    assert salient_features(text) == salient_features(text)
```

- [ ] **Step 1.2: Run to verify failure**

Run: `$env:PYTHONPATH='src'; python -m pytest tests/selector/test_coverage.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'evidence_rag.selector.coverage'`

- [ ] **Step 1.3: Implement salient_features**

```python
"""Set-level coverage selection (A2 spec 5-7).

Complementarity is a property of the selected set, so selection moves from
pointwise-score-and-truncate to greedy set cover over deterministic query+answer
features. No NLI, no absolute threshold; the drop gate still governs removal — this
module only packs survivors. Feature extraction is a deterministic heuristic whose
error the E1-support component eval measures (spec 9).
"""

import re
from collections.abc import Mapping, Sequence

from evidence_rag.contracts.models import EvidenceCandidate
from evidence_rag.selector.answer_norm import STOPWORDS, canonicalize_answer

_NUMBER_TOKEN = re.compile(
    r"(?:us\$|\$|£|€)?\d[\d,]*(?:\.\d+)?\s?"
    r"(?:%|percent|k|m|b|thousand|million|billion|trillion)?",
    re.IGNORECASE,
)
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9&.\-]{2,}")


def salient_features(text: str) -> frozenset[str]:
    features: set[str] = set()
    for match in _NUMBER_TOKEN.finditer(text):
        token = match.group(0).strip()
        if not any(char.isdigit() for char in token):
            continue
        canonical = canonicalize_answer(token)
        if canonical:
            features.add(f"num:{canonical}")
    for match in _WORD.finditer(text):
        token = match.group(0).strip(".").lower()
        if token and token not in STOPWORDS and not token.isdigit():
            features.add(f"tok:{token}")
    return frozenset(features)
```

- [ ] **Step 1.4: Run to verify pass**

Run: `$env:PYTHONPATH='src'; python -m pytest tests/selector/test_coverage.py -q`
Expected: PASS (5 tests)

- [ ] **Step 1.5: Commit**

```bash
git add src/evidence_rag/selector/coverage.py tests/selector/test_coverage.py
git commit -m "feat(selector): deterministic salient-feature extraction for coverage"
```

### Task 2: coverage_select

**Files:** Modify `src/evidence_rag/selector/coverage.py`; extend `tests/selector/test_coverage.py`

- [ ] **Step 2.1: Write failing tests**

```python
from evidence_rag.contracts.models import EvidenceCandidate
from evidence_rag.selector.coverage import coverage_select


def cand(evidence_id: str, text: str, rank: int, document_id: str | None = None) -> EvidenceCandidate:
    return EvidenceCandidate(
        evidence_id=evidence_id,
        document_id=document_id or f"doc-{evidence_id}",
        chunk_id=f"chunk-{evidence_id}",
        text=text,
        source_uri=f"fixture://{evidence_id}",
        retrieval_score=1.0,
        retrieval_rank=rank,
    )


def test_answer_cluster_representative_is_pinned() -> None:
    survivors = (
        cand("ans", "IBM 2025 operating margin was 18 percent", 1),
        cand("noise", "Unrelated weather report about rain", 2),
    )
    out = coverage_select(
        survivors,
        blended_by_id={"ans": 0.4, "noise": 0.9},
        answer_by_id={"ans": "18", "noise": None},
        query_text="IBM 2025 operating margin",
        max_selected=1,
    )
    assert tuple(c.evidence_id for c in out) == ("ans",)


def test_complementary_support_beats_irrelevant_after_pin() -> None:
    survivors = (
        cand("ans", "IBM 2025 operating margin was 18 percent", 1),
        cand("comp", "IBM 2025 revenue 1.2B and operating income", 2),
        cand("noise", "A recipe for chocolate cake", 3),
    )
    out = coverage_select(
        survivors,
        blended_by_id={"ans": 0.9, "comp": 0.5, "noise": 0.6},
        answer_by_id={"ans": "18", "comp": None, "noise": None},
        query_text="IBM 2025 operating margin",
        max_selected=2,
    )
    ids = tuple(c.evidence_id for c in out)
    assert ids[0] == "ans"
    assert "comp" in ids and "noise" not in ids


def test_redundant_duplicate_is_demoted() -> None:
    survivors = (
        cand("ans", "IBM 2025 operating margin was 18 percent", 1),
        cand("dup", "IBM 2025 operating margin was 18 percent", 2, document_id="doc-ans"),
        cand("comp", "IBM 2025 revenue 1.2B operating income detail", 3),
    )
    out = coverage_select(
        survivors,
        blended_by_id={"ans": 0.9, "dup": 0.8, "comp": 0.1},
        answer_by_id={"ans": "18", "dup": "18", "comp": None},
        query_text="IBM 2025 operating margin",
        max_selected=2,
    )
    ids = tuple(c.evidence_id for c in out)
    assert "ans" in ids and "comp" in ids and "dup" not in ids


def test_two_answer_clusters_each_get_a_pinned_rep() -> None:
    survivors = (
        cand("a", "IBM margin was 18 percent", 1),
        cand("b", "IBM margin was 16 percent", 2),
        cand("c", "IBM margin context filler", 3),
    )
    out = coverage_select(
        survivors,
        blended_by_id={"a": 0.9, "b": 0.8, "c": 0.7},
        answer_by_id={"a": "18", "b": "16", "c": None},
        query_text="IBM margin",
        max_selected=2,
    )
    ids = set(c.evidence_id for c in out)
    assert ids == {"a", "b"}


def test_shortfall_not_padded_and_deterministic() -> None:
    survivors = (cand("a", "IBM margin 18 percent", 1),)
    args = dict(
        blended_by_id={"a": 0.5},
        answer_by_id={"a": "18"},
        query_text="IBM margin",
        max_selected=5,
    )
    out1 = coverage_select(survivors, **args)
    out2 = coverage_select(survivors, **args)
    assert tuple(c.evidence_id for c in out1) == ("a",)
    assert out1 == out2


def test_empty_universe_falls_back_to_blended_after_pin() -> None:
    survivors = (
        cand("x", "zzz", 1),
        cand("y", "qqq", 2),
    )
    out = coverage_select(
        survivors,
        blended_by_id={"x": 0.2, "y": 0.9},
        answer_by_id={"x": None, "y": None},
        query_text="nothing matches",
        max_selected=2,
    )
    assert tuple(c.evidence_id for c in out) == ("y", "x")
```

- [ ] **Step 2.2: Run to verify failure**

Run: `$env:PYTHONPATH='src'; python -m pytest tests/selector/test_coverage.py -q`
Expected: FAIL — `ImportError: cannot import name 'coverage_select'`

- [ ] **Step 2.3: Implement coverage_select (append to coverage.py)**

```python
def coverage_select(
    survivors: Sequence[EvidenceCandidate],
    *,
    blended_by_id: Mapping[str, float],
    answer_by_id: Mapping[str, str | None],
    query_text: str,
    max_selected: int,
) -> tuple[EvidenceCandidate, ...]:
    if max_selected <= 0 or not survivors:
        return ()

    clusters: dict[str, list[EvidenceCandidate]] = {}
    for candidate in survivors:
        answer = answer_by_id.get(candidate.evidence_id)
        if answer is not None:
            clusters.setdefault(answer, []).append(candidate)

    def support(answer: str) -> int:
        return len({member.document_id for member in clusters[answer]})

    ordered_clusters = sorted(clusters, key=lambda answer: (-support(answer), answer))

    universe: set[str] = set(salient_features(query_text))
    if ordered_clusters:
        for member in clusters[ordered_clusters[0]]:
            universe |= salient_features(member.text)

    def features(candidate: EvidenceCandidate) -> frozenset[str]:
        return salient_features(candidate.text) & universe

    def blended(candidate: EvidenceCandidate) -> float:
        return blended_by_id.get(candidate.evidence_id, candidate.retrieval_score)

    output: list[EvidenceCandidate] = []
    pinned: set[str] = set()
    covered: set[str] = set()
    for answer in ordered_clusters:
        if len(output) >= max_selected:
            break
        representative = min(
            clusters[answer],
            key=lambda c: (-blended(c), c.retrieval_rank, c.evidence_id),
        )
        output.append(representative)
        pinned.add(representative.evidence_id)
        covered |= features(representative)

    remaining = [c for c in survivors if c.evidence_id not in pinned]
    while len(output) < max_selected and remaining:
        remaining.sort(
            key=lambda c: (
                -len(features(c) - covered),
                -blended(c),
                c.retrieval_rank,
                c.evidence_id,
            )
        )
        chosen = remaining.pop(0)
        output.append(chosen)
        covered |= features(chosen)

    return tuple(output)
```

- [ ] **Step 2.4: Run to verify pass**

Run: `$env:PYTHONPATH='src'; python -m pytest tests/selector/test_coverage.py -q`
Expected: PASS (11 tests total)

- [ ] **Step 2.5: Commit**

```bash
git add src/evidence_rag/selector/coverage.py tests/selector/test_coverage.py
git commit -m "feat(selector): greedy set-cover selection with answer-cluster pinning"
```

### Task 3: extract `_gate` helper in gated.py

**Files:** Modify `src/evidence_rag/selector/gated.py` (refactor only — existing `tests/selector/test_gated.py` must stay green)

- [ ] **Step 3.1: Add a `GateResult` dataclass and `_gate` method**

Add near the top imports: `from dataclasses import dataclass` is already imported. Add:

```python
@dataclass(frozen=True)
class GateResult:
    survivors: tuple[EvidenceCandidate, ...]
    blended_by_id: dict[str, float]
    answer_by_id: dict[str, str | None]
```

Add `from evidence_rag.contracts.models import EvidenceCandidate` and
`from evidence_rag.selector.answer_norm import canonicalize_answer, is_valid_answer`
(is_valid_answer already imported). Then add a method on `GatedCorroborationSelector`:

```python
    def _gate(self, query: Query, candidates: CandidateSet) -> GateResult:
        ranked = tuple(sorted(candidates.candidates, key=lambda item: item.retrieval_rank))
        window = ranked[: self.top_n]
        tail = ranked[self.top_n :]
        extracted = self._engine.extract(query, window)
        corroboration = minmax(corroboration_scores(extracted.answers, extracted.parametric))
        relevance = minmax(tuple(candidate.retrieval_score for candidate in window))
        blended = {
            window[index].evidence_id: (
                self.alpha * relevance[index] + (1.0 - self.alpha) * corroboration[index]
            )
            for index in range(len(window))
        }
        clusters = build_clusters(window, extracted.answers)
        cluster_by_member = {
            member_id: cluster for cluster in clusters for member_id in cluster.member_ids
        }
        dropped: set[str] = set()
        for index, candidate in enumerate(window):
            decision = self._decide(
                candidate.evidence_id, extracted.answers[index], clusters, cluster_by_member
            )
            if decision.action == "drop":
                dropped.add(candidate.evidence_id)
            if self.on_gate_decision is not None:
                self.on_gate_decision(decision)
        answer_by_id: dict[str, str | None] = {}
        for index, candidate in enumerate(window):
            raw = extracted.answers[index]
            answer_by_id[candidate.evidence_id] = (
                canonicalize_answer(raw) if is_valid_answer(raw) else None
            )
        survivors = tuple(
            candidate
            for candidate in window
            if candidate.evidence_id not in dropped
        ) + tail
        for candidate in tail:
            blended.setdefault(candidate.evidence_id, candidate.retrieval_score)
            answer_by_id.setdefault(candidate.evidence_id, None)
        return GateResult(survivors=survivors, blended_by_id=blended, answer_by_id=answer_by_id)
```

- [ ] **Step 3.2: Rewrite `GatedCorroborationSelector.select` to use `_gate`**

Replace the body of `select` (after the guard clauses) so it calls `_gate` then reproduces the current ordering (blended desc within survivors, preserving window-before-tail via blended then rank):

```python
    def select(self, query, candidates, max_selected):
        if query.query_id != candidates.query_id:
            raise ValueError("query and candidates query IDs differ")
        if max_selected <= 0:
            raise ValueError("max_selected must be positive")
        if not candidates.candidates:
            return SelectionResult(query_id=query.query_id, items=())
        result = self._gate(query, candidates)
        ordered = sorted(
            result.survivors,
            key=lambda c: (-result.blended_by_id[c.evidence_id], c.retrieval_rank, c.evidence_id),
        )
        output = ordered[:max_selected]
        return _to_selection(query, output, result.blended_by_id)
```

Add a module-level helper reused by both selectors:

```python
def _to_selection(query, output, blended_by_id):
    return SelectionResult(
        query_id=query.query_id,
        items=tuple(
            SelectionItem(
                evidence_id=candidate.evidence_id,
                selection_score=blended_by_id.get(candidate.evidence_id, candidate.retrieval_score),
                selection_rank=rank,
            )
            for rank, candidate in enumerate(output, start=1)
        ),
    )
```

Note: the original ordered by blended already (tail kept its retrieval order after window). Sorting all survivors by (blended desc, rank, id) is equivalent for the window and keeps tail after window because tail blended = retrieval_score which is typically below reranked window values; if a regression appears in `test_matches_plain_corroboration_when_gate_never_fires`, restore the original `reranked_window + tail` ordering inside `select` instead and keep `_gate` only for the coverage selector.

- [ ] **Step 3.3: Run existing gate tests**

Run: `$env:PYTHONPATH='src'; python -m pytest tests/selector/test_gated.py tests/selector/test_corroboration.py -q`
Expected: PASS (all existing). If `test_matches_plain_corroboration_when_gate_never_fires` regresses, apply the note in Step 3.2.

- [ ] **Step 3.4: Commit**

```bash
git add src/evidence_rag/selector/gated.py
git commit -m "refactor(selector): extract reusable gate helper for coverage layer"
```

### Task 4: GatedCoverageSelector

**Files:** Modify `src/evidence_rag/selector/gated.py`; Test `tests/selector/test_gated_coverage.py`

- [ ] **Step 4.1: Write failing tests**

```python
from evidence_rag.contracts.models import CandidateSet, EvidenceCandidate, Query
from evidence_rag.selector.gated import GatedCoverageSelector


class MappedExtractor:
    def __init__(self, mapping: dict[str, str]) -> None:
        self.mapping = mapping

    def generate(self, prompt: str) -> str:
        if "own knowledge" in prompt:
            return "NONE"
        for text, answer in self.mapping.items():
            if f"Passage: {text}" in prompt:
                return answer
        return "NONE"


def cand(evidence_id, text, score, rank):
    return EvidenceCandidate(
        evidence_id=evidence_id, document_id=f"doc-{evidence_id}", chunk_id=f"c-{evidence_id}",
        text=text, source_uri=f"fixture://{evidence_id}", retrieval_score=score, retrieval_rank=rank,
    )


QUERY = Query(query_id="q", text="IBM 2025 operating margin")


def test_coverage_keeps_answer_and_complementary_drops_noise() -> None:
    pool = CandidateSet(query_id="q", candidates=(
        cand("ans", "IBM 2025 operating margin was 18 percent", 0.9, 1),
        cand("comp", "IBM 2025 revenue 1.2B and operating income detail", 0.5, 2),
        cand("noise", "A recipe for chocolate cake with sugar", 0.6, 3),
    ))
    extractor = MappedExtractor({
        "IBM 2025 operating margin was 18 percent": "18 percent",
    })
    selector = GatedCoverageSelector(extractor, use_parametric=False)
    ids = tuple(i.evidence_id for i in selector.select(QUERY, pool, 2).items)
    assert ids[0] == "ans" and "comp" in ids and "noise" not in ids


def test_coverage_selector_respects_shortfall() -> None:
    pool = CandidateSet(query_id="q", candidates=(
        cand("ans", "IBM 2025 operating margin was 18 percent", 0.9, 1),
    ))
    extractor = MappedExtractor({"IBM 2025 operating margin was 18 percent": "18 percent"})
    selector = GatedCoverageSelector(extractor, use_parametric=False)
    assert len(selector.select(QUERY, pool, 5).items) == 1
```

- [ ] **Step 4.2: Run to verify failure**

Run: `$env:PYTHONPATH='src'; python -m pytest tests/selector/test_gated_coverage.py -q`
Expected: FAIL — `ImportError: cannot import name 'GatedCoverageSelector'`

- [ ] **Step 4.3: Implement GatedCoverageSelector (append to gated.py)**

Add import: `from evidence_rag.selector.coverage import coverage_select`. Then:

```python
class GatedCoverageSelector(GatedCorroborationSelector):
    """Gate (drop) followed by set-level coverage packing of survivors (A2 spec)."""

    def select(
        self,
        query: Query,
        candidates: CandidateSet,
        max_selected: int,
    ) -> SelectionResult:
        if query.query_id != candidates.query_id:
            raise ValueError("query and candidates query IDs differ")
        if max_selected <= 0:
            raise ValueError("max_selected must be positive")
        if not candidates.candidates:
            return SelectionResult(query_id=query.query_id, items=())
        result = self._gate(query, candidates)
        output = coverage_select(
            result.survivors,
            blended_by_id=result.blended_by_id,
            answer_by_id=result.answer_by_id,
            query_text=query.text,
            max_selected=max_selected,
        )
        return _to_selection(query, output, result.blended_by_id)
```

- [ ] **Step 4.4: Run to verify pass + full selector suite**

Run: `$env:PYTHONPATH='src'; python -m pytest tests/selector -q`
Expected: PASS (all selector tests including the new coverage selector)

- [ ] **Step 4.5: Commit**

```bash
git add src/evidence_rag/selector/gated.py tests/selector/test_gated_coverage.py
git commit -m "feat(selector): GatedCoverageSelector composing gate and coverage packing"
```

### Task 5: composition registration

**Files:** Modify `src/evidence_rag/composition.py`; extend `tests/pipeline/test_selector_registration.py`

- [ ] **Step 5.1: Write failing test**

```python
def test_gated_coverage_builds_with_parameters() -> None:
    from evidence_rag.selector.gated import GatedCoverageSelector
    selector = build_selector(
        ModuleConfig(name="gated-coverage-corroboration", parameters={"margin": 3, "support_cap": 2}),
        llm=FakeLLM(),
    )
    assert isinstance(selector, GatedCoverageSelector)
    assert selector.margin == 3
    assert selector.support_cap == 2
```

- [ ] **Step 5.2: Run to verify failure**

Run: `$env:PYTHONPATH='src'; python -m pytest tests/pipeline/test_selector_registration.py -q`
Expected: FAIL — `unknown selector: gated-coverage-corroboration`

- [ ] **Step 5.3: Implement in composition.py**

Add import `from evidence_rag.selector.gated import GatedCoverageSelector` and a branch in `build_selector` mirroring the `gated-corroboration` branch:

```python
    if config.name == "gated-coverage-corroboration":
        parameters = _selector_parameters(
            config, frozenset({"alpha", "margin", "support_cap", "top_n"})
        )
        client = llm if llm is not None else GraniteLLMClient()
        return GatedCoverageSelector(
            client,
            alpha=float(parameters.get("alpha", 0.6)),
            margin=int(parameters.get("margin", 2)),
            support_cap=int(parameters.get("support_cap", 1)),
            top_n=int(parameters.get("top_n", 20)),
        )
```

- [ ] **Step 5.4: Run to verify pass, full suite, typecheck, ruff**

Run:
```
$env:PYTHONPATH='src'; python -m pytest -q
$env:MYPYPATH='src'; python -m mypy
python -m ruff check src/evidence_rag/selector src/evidence_rag/composition.py tests/selector
```
Expected: pytest 0 new failures (only the known Windows-fixture set if still present — confirm parity), mypy clean, ruff clean.

- [ ] **Step 5.5: Commit**

```bash
git add src/evidence_rag/composition.py tests/pipeline/test_selector_registration.py
git commit -m "feat(composition): register gated-coverage-corroboration selector"
```

---

## Experiment marking (running-hpc-experiments)

- **E1-support** (support-edge precision/recall) and **E3** (coverage-on vs coverage-off, grounded F1/faithfulness) are **HPC + BLOCKED** on the A2 evaluation data (spec §9 §13; couples to the §14 dataset decision — needs complementarity-stressing data). Do NOT build their harness in this plan. Add pre-registered BEFORE entries to `docs/hpc-run-log.md` when the data decision lands; the E3 slurm reuses the `run_selector_gate.slurm` triple pattern with a third `coverage` config arm.

## Self-review notes

- Spec coverage: §5→Task 1; §6 support-edge (no-answer + feature overlap) realized by `answer_by_id[None]` + universe intersection in Task 2; §7 pin+greedy→Task 2; §7.1 recall-pin + shortfall + determinism → Task 2/4 tests; §11 files → Tasks 3-5; §9 evals → experiment-marking (blocked). 
- No placeholders: all code complete; the entity heuristic is pinned to the two regexes in Task 1 (spec §5's "impl detail" is now concrete).
- Type consistency: `coverage_select` keyword-only signature matches its call in `GatedCoverageSelector`; `_gate`/`GateResult`/`_to_selection` names consistent across Tasks 3-4.
- Risk flagged inline: Step 3.2 note covers the one behavior-parity risk (plain-gate ordering) with a concrete fallback.
