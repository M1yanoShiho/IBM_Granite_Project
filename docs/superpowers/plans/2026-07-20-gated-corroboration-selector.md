# Gated Corroboration Selector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the contrastive drop gate from `docs/superpowers/specs/2026-07-20-gated-corroboration-selector-design.md`: deterministic answer canonicalization, shared extraction/clustering substrate, `GatedCorroborationSelector`, config registration, and the HPC experiment triple (marked, not submitted — blocked on the §14 dataset decision).

**Architecture:** Selector module gains a strict gate stage after the existing tolerant rerank stage. Gate drops candidate c iff: valid extracted answer AND a competing answer cluster exists AND independent-vote margin ≥ `margin` AND own support ≤ `support_cap`. No scores in the gate — integer in-pool votes only. Contract (`SelectionResult`) unchanged; shortfall below `max_selected` is already legal.

**Tech Stack:** Python 3.11, pydantic v2 (frozen contracts), pytest, mypy strict, ruff. No new dependencies.

**Experiment marking (running-hpc-experiments):** every experiment below is tagged **LOCAL** or **HPC**. HPC runs ship as triple code+slurm+ledger; nothing is submitted until the §14 dataset decision lands.

| ID | Experiment | Where | Status |
|---|---|---|---|
| T1–T4 | unit/typecheck suites | LOCAL | run in this plan |
| E1 | edge/cluster component eval (false-conflict / missed-conflict) | **HPC** | BLOCKED(§14 datasets); triple ships with the eval harness |
| E2 | gate-on/off paired selector comparison (dual Gate: harm↓, recall non-inf −0.01) | **HPC** | slurm + ledger created in Task 5; configs BLOCKED(§14) |

---

## File Structure

| File | Responsibility |
|---|---|
| Create `src/evidence_rag/selector/answer_norm.py` | text normalization (moved) + deterministic numeric/date canonicalization + answer validity |
| Create `src/evidence_rag/selector/extraction.py` | prompts + `TextGenerator` protocol + `AnswerExtractionEngine` (one extraction pass shared by rerank and gate) |
| Create `src/evidence_rag/selector/clusters.py` | `AnswerCluster` + `build_clusters` (canonical-answer grouping, document_id independent votes) |
| Create `src/evidence_rag/selector/gated.py` | `GateDecision` + `GatedCorroborationSelector` |
| Modify `src/evidence_rag/selector/corroboration.py` | delegate to substrate; re-export moved names; votes switch to canonical equality |
| Modify `src/evidence_rag/composition.py` | register `corroboration` + `gated-corroboration` with validated parameters and `llm` seam |
| Create `scripts/run_selector_gate.slurm` + `docs/hpc-run-log.md` | E2 triple (code+script+ledger) |
| Tests | `tests/selector/test_answer_norm.py`, `tests/selector/test_clusters.py`, `tests/selector/test_gated.py`, `tests/pipeline/test_selector_registration.py` |

---

### Task 0: Baseline

- [ ] **Step 0.1: Verify suite is green before touching anything**

Run: `python -m pytest`
Expected: all pass (repo was clean at ab10dc0/eefed3a).

### Task 1: answer_norm.py

**Files:** Create `src/evidence_rag/selector/answer_norm.py`, Test `tests/selector/test_answer_norm.py`

- [ ] **Step 1.1: Write the failing tests**

```python
from evidence_rag.selector.answer_norm import canonicalize_answer, is_valid_answer, normalize_answer


def test_text_normalization_is_preserved() -> None:
    assert normalize_answer("The Acme.") == "acme"
    assert canonicalize_answer("Acme Corp.") == "acme corp"


def test_thousands_separators_collapse() -> None:
    assert canonicalize_answer("1,200") == "1200"
    assert canonicalize_answer("1200") == "1200"


def test_currency_and_magnitude_aliases_merge() -> None:
    assert canonicalize_answer("$1.2B") == "1200000000"
    assert canonicalize_answer("1,200 million") == "1200000000"
    assert canonicalize_answer("1.2 billion") == "1200000000"
    assert canonicalize_answer("US$500m") == "500000000"


def test_percent_forms_merge() -> None:
    assert canonicalize_answer("18%") == "18"
    assert canonicalize_answer("18 percent") == "18"


def test_trailing_zero_trim() -> None:
    assert canonicalize_answer("$45.30") == "45.3"


def test_date_whitelist_merges() -> None:
    assert canonicalize_answer("January 5, 2019") == "2019-01-05"
    assert canonicalize_answer("5 January 2019") == "2019-01-05"
    assert canonicalize_answer("2019-01-05") == "2019-01-05"


def test_month_year_is_not_parsed() -> None:
    assert canonicalize_answer("March 2019") == "march 2019"


def test_mixed_tokens_fall_back_to_text() -> None:
    assert canonicalize_answer("covid-19") == "covid-19"
    assert canonicalize_answer("three") == "three"


def test_idempotent() -> None:
    for value in ("$1.2B", "January 5, 2019", "Acme Corp.", "18%"):
        once = canonicalize_answer(value)
        assert canonicalize_answer(once) == once


def test_validity_rules_unchanged() -> None:
    assert is_valid_answer("Acme")
    assert not is_valid_answer("none")
    assert not is_valid_answer("it")
    assert is_valid_answer("42")
```

- [ ] **Step 1.2: Run to verify failure**

Run: `python -m pytest tests/selector/test_answer_norm.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'evidence_rag.selector.answer_norm'`

- [ ] **Step 1.3: Implement**

```python
"""Deterministic answer canonicalization (spec §6.2).

Cluster keys and corroboration votes compare answers AFTER canonicalization so that
numeric aliases ("$1.2B" vs "1,200 million") land in one cluster. Rules are
deterministic and unit-tested; known tradeoff: bare magnitude suffixes ("19b")
are read as numbers — the §12 component eval (false-conflict rate) measures the
cost of such merges on real pools.
"""

import re
from decimal import Decimal, InvalidOperation

MIN_ANSWER_LENGTH = 3
STOPWORDS = {"the", "a", "an", "none", "n/a", "unknown", "it", "yes", "no"}

_CURRENCY_PREFIX = re.compile(r"^(?:us\$|\$|£|€)\s*")
_PERCENT_SUFFIX = re.compile(r"\s*(?:%|percent|per cent)$")
_NUMBER_PATTERN = re.compile(
    r"^(?P<number>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"\s*(?P<magnitude>k|m|b|thousand|million|billion|trillion)?$"
)
_MAGNITUDES = {
    "k": Decimal(1_000),
    "thousand": Decimal(1_000),
    "m": Decimal(1_000_000),
    "million": Decimal(1_000_000),
    "b": Decimal(1_000_000_000),
    "billion": Decimal(1_000_000_000),
    "trillion": Decimal(10**12),
}
_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12,
}
_DATE_ISO = re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})$")
_DATE_DMY = re.compile(r"^(\d{1,2})\s+([a-z]+)\s+(\d{4})$")
_DATE_MDY = re.compile(r"^([a-z]+)\s+(\d{1,2}),?\s+(\d{4})$")


def normalize_answer(answer: str) -> str:
    normalized = answer.strip().lower()
    normalized = re.sub(r"^(the|a|an)\s+", "", normalized)
    return normalized.strip(" \t\n.,;:!?\"'()[]")


def is_valid_answer(answer: str) -> bool:
    normalized = normalize_answer(answer)
    if not normalized or normalized in STOPWORDS:
        return False
    if len(normalized) < MIN_ANSWER_LENGTH and not normalized.isdigit():
        return False
    return True


def _format_decimal(value: Decimal) -> str:
    return format(value.normalize(), "f")


def _canonical_number(text: str) -> str | None:
    stripped = _PERCENT_SUFFIX.sub("", _CURRENCY_PREFIX.sub("", text))
    match = _NUMBER_PATTERN.match(stripped)
    if match is None:
        return None
    try:
        value = Decimal(match.group("number").replace(",", ""))
    except InvalidOperation:
        return None
    magnitude = match.group("magnitude")
    if magnitude is not None:
        value *= _MAGNITUDES[magnitude]
    return _format_decimal(value)


def _build_date(year: str, month: int, day: str) -> str | None:
    day_number = int(day)
    if not 1 <= day_number <= 31:
        return None
    return f"{int(year):04d}-{month:02d}-{day_number:02d}"


def _canonical_date(text: str) -> str | None:
    if match := _DATE_ISO.match(text):
        year, month, day = match.groups()
        if 1 <= int(month) <= 12:
            return _build_date(year, int(month), day)
        return None
    if match := _DATE_DMY.match(text):
        day, month_name, year = match.groups()
        if month_name in _MONTHS:
            return _build_date(year, _MONTHS[month_name], day)
        return None
    if match := _DATE_MDY.match(text):
        month_name, day, year = match.groups()
        if month_name in _MONTHS:
            return _build_date(year, _MONTHS[month_name], day)
    return None


def canonicalize_answer(answer: str) -> str:
    text = normalize_answer(answer)
    if (date := _canonical_date(text)) is not None:
        return date
    if (number := _canonical_number(text)) is not None:
        return number
    return text
```

- [ ] **Step 1.4: Run to verify pass**

Run: `python -m pytest tests/selector/test_answer_norm.py -v`
Expected: PASS (11 tests)

- [ ] **Step 1.5: Commit**

```bash
git add src/evidence_rag/selector/answer_norm.py tests/selector/test_answer_norm.py
git commit -m "feat(selector): deterministic answer canonicalization"
```

### Task 2: substrate extraction + clusters

**Files:** Create `src/evidence_rag/selector/extraction.py`, `src/evidence_rag/selector/clusters.py`; Modify `src/evidence_rag/selector/corroboration.py`; Test `tests/selector/test_clusters.py` (+ existing `tests/selector/test_corroboration.py` must stay green)

- [ ] **Step 2.1: Write the failing cluster tests**

```python
from evidence_rag.contracts.models import EvidenceCandidate
from evidence_rag.selector.clusters import build_clusters


def evidence(evidence_id: str, document_id: str) -> EvidenceCandidate:
    return EvidenceCandidate(
        evidence_id=evidence_id,
        document_id=document_id,
        chunk_id=f"chunk-{evidence_id}",
        text=f"text {evidence_id}",
        source_uri=f"fixture://{evidence_id}",
        retrieval_score=1.0,
        retrieval_rank=int(evidence_id[-1]),
    )


def test_clusters_group_by_canonical_answer() -> None:
    window = (evidence("e1", "d1"), evidence("e2", "d2"), evidence("e3", "d3"))
    clusters = build_clusters(window, ("$1.2B", "1,200 million", "3 billion"))
    by_answer = {cluster.answer: cluster for cluster in clusters}
    assert set(by_answer) == {"1200000000", "3000000000"}
    assert by_answer["1200000000"].member_ids == ("e1", "e2")
    assert by_answer["1200000000"].independent_support == 2


def test_same_document_counts_once() -> None:
    window = (evidence("e1", "dX"), evidence("e2", "dX"), evidence("e3", "dX"))
    clusters = build_clusters(window, ("Acme", "Acme", "Acme"))
    assert clusters[0].independent_support == 1
    assert clusters[0].member_ids == ("e1", "e2", "e3")


def test_invalid_answers_are_excluded() -> None:
    window = (evidence("e1", "d1"), evidence("e2", "d2"))
    clusters = build_clusters(window, ("NONE", "it"))
    assert clusters == ()


def test_length_mismatch_raises() -> None:
    window = (evidence("e1", "d1"),)
    try:
        build_clusters(window, ("Acme", "Globex"))
    except ValueError:
        return
    raise AssertionError("expected ValueError")
```

- [ ] **Step 2.2: Run to verify failure**

Run: `python -m pytest tests/selector/test_clusters.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'evidence_rag.selector.clusters'`

- [ ] **Step 2.3: Create extraction.py**

```python
"""Shared answer-extraction substrate (spec §6.1).

One extraction pass per query window feeds BOTH the tolerant rerank stage and the
strict gate stage — the gate adds no LLM calls.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from evidence_rag.contracts.models import EvidenceCandidate, Query

EXTRACT_PROMPT = (
    "Using ONLY the passage below, answer the question with the shortest exact answer "
    "(a name, place, date, or number). If the passage does not answer it, reply NONE.\n"
    "Question: {question}\n"
    "Passage: {passage}\n"
    "Answer:"
)

PARAMETRIC_PROMPT = (
    "Answer the question with the shortest exact answer from your own knowledge. "
    "If you are not sure, reply NONE.\n"
    "Question: {question}\n"
    "Answer:"
)


class TextGenerator(Protocol):
    def generate(self, prompt: str) -> str: ...


@dataclass(frozen=True)
class ExtractedAnswers:
    answers: tuple[str, ...]
    parametric: str | None


class AnswerExtractionEngine:
    def __init__(
        self,
        answer_extractor: TextGenerator,
        *,
        passage_chars: int = 600,
        use_parametric: bool = True,
    ) -> None:
        if passage_chars <= 0:
            raise ValueError("passage_chars must be positive")
        self.answer_extractor = answer_extractor
        self.passage_chars = passage_chars
        self.use_parametric = use_parametric

    def _extract_answer(self, query: Query, candidate: EvidenceCandidate) -> str:
        prompt = EXTRACT_PROMPT.format(
            question=query.text,
            passage=candidate.text[: self.passage_chars],
        )
        return self.answer_extractor.generate(prompt).strip()

    def extract(self, query: Query, window: Sequence[EvidenceCandidate]) -> ExtractedAnswers:
        answers = tuple(self._extract_answer(query, candidate) for candidate in window)
        parametric = (
            self.answer_extractor.generate(PARAMETRIC_PROMPT.format(question=query.text)).strip()
            if self.use_parametric
            else None
        )
        return ExtractedAnswers(answers=answers, parametric=parametric)
```

- [ ] **Step 2.4: Create clusters.py**

```python
"""Answer clusters over the rerank window (spec §6.3).

Independent support counts distinct document_id values — chunks of one document
are one vote. Parametric answers never enter clusters (spec §8: signal graduation).
"""

from collections.abc import Sequence
from dataclasses import dataclass

from evidence_rag.contracts.models import EvidenceCandidate
from evidence_rag.selector.answer_norm import canonicalize_answer, is_valid_answer


@dataclass(frozen=True)
class AnswerCluster:
    answer: str
    member_ids: tuple[str, ...]
    independent_support: int


def build_clusters(
    window: Sequence[EvidenceCandidate],
    answers: Sequence[str],
) -> tuple[AnswerCluster, ...]:
    if len(window) != len(answers):
        raise ValueError("window and answers lengths differ")
    groups: dict[str, list[EvidenceCandidate]] = {}
    for candidate, answer in zip(window, answers, strict=True):
        if not is_valid_answer(answer):
            continue
        groups.setdefault(canonicalize_answer(answer), []).append(candidate)
    return tuple(
        AnswerCluster(
            answer=answer,
            member_ids=tuple(candidate.evidence_id for candidate in members),
            independent_support=len({candidate.document_id for candidate in members}),
        )
        for answer, members in groups.items()
    )
```

- [ ] **Step 2.5: Rewrite corroboration.py to delegate (public API preserved)**

```python
from collections.abc import Sequence

from evidence_rag.contracts.models import (
    CandidateSet,
    EvidenceCandidate,
    Query,
    SelectionItem,
    SelectionResult,
)
from evidence_rag.selector.answer_norm import (
    MIN_ANSWER_LENGTH as MIN_ANSWER_LENGTH,
    STOPWORDS as STOPWORDS,
    canonicalize_answer,
    is_valid_answer as is_valid_answer,
    normalize_answer as normalize_answer,
)
from evidence_rag.selector.extraction import (
    EXTRACT_PROMPT as EXTRACT_PROMPT,
    PARAMETRIC_PROMPT as PARAMETRIC_PROMPT,
    AnswerExtractionEngine,
    TextGenerator as TextGenerator,
)


def corroboration_scores(
    answers: Sequence[str],
    parametric_answer: str | None = None,
) -> tuple[float, ...]:
    normalized_answers = tuple(
        canonicalize_answer(answer) if is_valid_answer(answer) else None
        for answer in answers
    )
    parametric = (
        canonicalize_answer(parametric_answer)
        if parametric_answer is not None and is_valid_answer(parametric_answer)
        else None
    )
    scores: list[float] = []
    for index, answer in enumerate(normalized_answers):
        if answer is None:
            scores.append(0.0)
            continue
        votes = sum(
            1
            for other_index, other in enumerate(normalized_answers)
            if other_index != index and other == answer
        )
        if parametric is not None and parametric == answer:
            votes += 1
        scores.append(float(votes))
    return tuple(scores)


def minmax(values: Sequence[float]) -> tuple[float, ...]:
    if not values:
        return ()
    low = min(values)
    high = max(values)
    if low == high:
        return tuple(0.0 for _ in values)
    return tuple((value - low) / (high - low) for value in values)


class CorroborationSelector:
    """Selector that blends retrieval relevance with cross-evidence answer support."""

    def __init__(
        self,
        answer_extractor: TextGenerator,
        *,
        alpha: float = 0.6,
        top_n: int = 20,
        use_parametric: bool = True,
        passage_chars: int = 600,
    ) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if top_n <= 0:
            raise ValueError("top_n must be positive")
        self.answer_extractor = answer_extractor
        self.alpha = alpha
        self.top_n = top_n
        self.use_parametric = use_parametric
        self.passage_chars = passage_chars
        self._engine = AnswerExtractionEngine(
            answer_extractor,
            passage_chars=passage_chars,
            use_parametric=use_parametric,
        )

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

        ranked_candidates = tuple(
            sorted(candidates.candidates, key=lambda item: item.retrieval_rank)
        )
        window = ranked_candidates[: self.top_n]
        extracted = self._engine.extract(query, window)
        corroboration = minmax(corroboration_scores(extracted.answers, extracted.parametric))
        relevance = minmax(tuple(candidate.retrieval_score for candidate in window))
        blended = tuple(
            self.alpha * relevance[index] + (1.0 - self.alpha) * corroboration[index]
            for index in range(len(window))
        )
        reranked_window = tuple(
            window[index]
            for index in sorted(
                range(len(window)),
                key=lambda i: (-blended[i], window[i].retrieval_rank, window[i].evidence_id),
            )
        )
        score_by_id = {
            candidate.evidence_id: blended[index]
            for index, candidate in enumerate(window)
        }
        output = (reranked_window + ranked_candidates[self.top_n :])[:max_selected]
        return SelectionResult(
            query_id=query.query_id,
            items=tuple(
                SelectionItem(
                    evidence_id=candidate.evidence_id,
                    selection_score=score_by_id.get(
                        candidate.evidence_id,
                        candidate.retrieval_score,
                    ),
                    selection_rank=rank,
                )
                for rank, candidate in enumerate(output, start=1)
            ),
        )
```

Note: `EvidenceCandidate` import retained for type context of the module's helpers; if ruff flags it unused after the rewrite, drop it.

- [ ] **Step 2.6: Run selector tests**

Run: `python -m pytest tests/selector -v`
Expected: PASS — clusters tests green AND both existing corroboration tests green (behavior preserved: text answers canonicalize to the same strings the old normalize produced).

- [ ] **Step 2.7: Commit**

```bash
git add src/evidence_rag/selector/extraction.py src/evidence_rag/selector/clusters.py src/evidence_rag/selector/corroboration.py tests/selector/test_clusters.py
git commit -m "refactor(selector): extract shared extraction/cluster substrate; canonical votes"
```

### Task 3: GatedCorroborationSelector

**Files:** Create `src/evidence_rag/selector/gated.py`; Test `tests/selector/test_gated.py`

- [ ] **Step 3.1: Write the failing tests (spec §7.2 table, complete)**

```python
from evidence_rag.contracts.models import (
    CandidateSet,
    EvidenceCandidate,
    GenerationResult,
    Query,
    SelectedEvidenceSet,
)
from evidence_rag.pipeline.service import EvidenceRAGPipeline
from evidence_rag.selector.corroboration import CorroborationSelector
from evidence_rag.selector.gated import GateDecision, GatedCorroborationSelector


class MappedExtractor:
    """Maps passage text -> extracted answer; parametric answer configurable."""

    def __init__(self, mapping: dict[str, str], parametric: str = "NONE") -> None:
        self.mapping = mapping
        self.parametric = parametric

    def generate(self, prompt: str) -> str:
        if "own knowledge" in prompt:
            return self.parametric
        for text, answer in self.mapping.items():
            if f"Passage: {text}" in prompt:
                return answer
        return "NONE"


def evidence(
    evidence_id: str,
    text: str,
    score: float,
    rank: int,
    document_id: str | None = None,
) -> EvidenceCandidate:
    return EvidenceCandidate(
        evidence_id=evidence_id,
        document_id=document_id or f"doc-{evidence_id}",
        chunk_id=f"chunk-{evidence_id}",
        text=text,
        source_uri=f"fixture://{evidence_id}",
        retrieval_score=score,
        retrieval_rank=rank,
    )


QUERY = Query(query_id="q", text="Which value?")


def candidate_set(*items: EvidenceCandidate) -> CandidateSet:
    return CandidateSet(query_id="q", candidates=items)


def selected_ids(selector: GatedCorroborationSelector, candidates: CandidateSet, k: int) -> tuple[str, ...]:
    return tuple(item.evidence_id for item in selector.select(QUERY, candidates, k).items)


def three_vs_one_pool() -> tuple[CandidateSet, MappedExtractor]:
    pool = candidate_set(
        evidence("lone", "text lone", 1.0, 1),
        evidence("w1", "text w1", 0.9, 2),
        evidence("w2", "text w2", 0.8, 3),
        evidence("w3", "text w3", 0.7, 4),
    )
    extractor = MappedExtractor(
        {"text lone": "Globex", "text w1": "Acme", "text w2": "Acme", "text w3": "Acme"}
    )
    return pool, extractor


def test_lone_source_with_no_competitor_is_never_dropped() -> None:
    pool = candidate_set(
        evidence("only", "text only", 1.0, 1),
        evidence("bg1", "text bg1", 0.9, 2),
        evidence("bg2", "text bg2", 0.8, 3),
    )
    extractor = MappedExtractor({"text only": "Acme"})
    selector = GatedCorroborationSelector(extractor, use_parametric=False)
    assert set(selected_ids(selector, pool, 3)) == {"only", "bg1", "bg2"}


def test_one_vs_one_conflict_keeps_both() -> None:
    pool = candidate_set(
        evidence("a", "text a", 1.0, 1),
        evidence("b", "text b", 0.9, 2),
    )
    extractor = MappedExtractor({"text a": "Acme", "text b": "Globex"})
    selector = GatedCorroborationSelector(extractor, use_parametric=False)
    assert set(selected_ids(selector, pool, 2)) == {"a", "b"}


def test_three_vs_one_drops_isolated_claim_without_padding() -> None:
    pool, extractor = three_vs_one_pool()
    selector = GatedCorroborationSelector(extractor, use_parametric=False)
    ids = selected_ids(selector, pool, 4)
    assert "lone" not in ids
    assert len(ids) == 3


def test_two_vs_one_within_margin_keeps_both() -> None:
    pool = candidate_set(
        evidence("y", "text y", 1.0, 1),
        evidence("x1", "text x1", 0.9, 2),
        evidence("x2", "text x2", 0.8, 3),
    )
    extractor = MappedExtractor({"text y": "Globex", "text x1": "Acme", "text x2": "Acme"})
    selector = GatedCorroborationSelector(extractor, use_parametric=False)
    assert "y" in selected_ids(selector, pool, 3)


def test_support_cap_blocks_drop_of_supported_cluster() -> None:
    pool = candidate_set(
        evidence("y1", "text y1", 1.0, 1),
        evidence("y2", "text y2", 0.9, 2),
        evidence("x1", "text x1", 0.8, 3),
        evidence("x2", "text x2", 0.7, 4),
        evidence("x3", "text x3", 0.6, 5),
        evidence("x4", "text x4", 0.5, 6),
    )
    extractor = MappedExtractor(
        {
            "text y1": "Globex", "text y2": "Globex",
            "text x1": "Acme", "text x2": "Acme", "text x3": "Acme", "text x4": "Acme",
        }
    )
    selector = GatedCorroborationSelector(extractor, use_parametric=False)
    ids = selected_ids(selector, pool, 6)
    assert "y1" in ids and "y2" in ids


def test_alias_answers_cluster_together_and_gate_uses_merged_votes() -> None:
    pool = candidate_set(
        evidence("b", "text b", 1.0, 1),
        evidence("a1", "text a1", 0.9, 2),
        evidence("a2", "text a2", 0.8, 3),
    )
    extractor = MappedExtractor(
        {"text b": "3 billion", "text a1": "$1.2B", "text a2": "1,200 million"}
    )
    selector = GatedCorroborationSelector(extractor, margin=1, use_parametric=False)
    assert "b" not in selected_ids(selector, pool, 3)


def test_same_document_chunks_collapse_to_one_vote() -> None:
    pool = candidate_set(
        evidence("x1", "text x1", 1.0, 1, document_id="doc-X"),
        evidence("x2", "text x2", 0.9, 2, document_id="doc-X"),
        evidence("x3", "text x3", 0.8, 3, document_id="doc-X"),
        evidence("y", "text y", 0.7, 4),
    )
    extractor = MappedExtractor(
        {"text x1": "Acme", "text x2": "Acme", "text x3": "Acme", "text y": "Globex"}
    )
    selector = GatedCorroborationSelector(extractor, margin=1, use_parametric=False)
    assert "y" in selected_ids(selector, pool, 4)


def test_gate_silent_when_nothing_extracts() -> None:
    pool = candidate_set(
        evidence("r1", "text r1", 1.0, 1),
        evidence("r2", "text r2", 0.9, 2),
        evidence("r3", "text r3", 0.8, 3),
    )
    selector = GatedCorroborationSelector(MappedExtractor({}), use_parametric=False)
    assert selected_ids(selector, pool, 3) == ("r1", "r2", "r3")


def test_drop_frees_budget_slot_for_replacement() -> None:
    pool, extractor = three_vs_one_pool()
    selector = GatedCorroborationSelector(extractor, use_parametric=False)
    ids = selected_ids(selector, pool, 3)
    assert set(ids) == {"w1", "w2", "w3"}


def test_matches_plain_corroboration_when_gate_never_fires() -> None:
    pool = candidate_set(
        evidence("wrong", "Globex is named in a distractor.", 1.0, 1),
        evidence("right-a", "Acme is named in one source.", 0.7, 2),
        evidence("right-b", "Acme is named in another source.", 0.6, 3),
    )
    extractor = MappedExtractor(
        {
            "Globex is named in a distractor.": "Globex",
            "Acme is named in one source.": "Acme",
            "Acme is named in another source.": "Acme",
        },
        parametric="Acme",
    )
    gated = GatedCorroborationSelector(extractor, alpha=0.2)
    plain = CorroborationSelector(extractor, alpha=0.2)
    assert gated.select(QUERY, pool, 2) == plain.select(QUERY, pool, 2)


def test_sink_receives_structured_decisions() -> None:
    decisions: list[GateDecision] = []
    pool, extractor = three_vs_one_pool()
    selector = GatedCorroborationSelector(
        extractor, use_parametric=False, on_gate_decision=decisions.append
    )
    selector.select(QUERY, pool, 4)
    by_id = {decision.evidence_id: decision for decision in decisions}
    assert len(decisions) == 4
    lone = by_id["lone"]
    assert lone.action == "drop"
    assert lone.own_support == 1
    assert lone.winner_support == 3
    assert lone.margin == 2
    assert by_id["w1"].action == "keep"


def test_deterministic_across_repeats() -> None:
    pool, extractor = three_vs_one_pool()
    selector = GatedCorroborationSelector(extractor, use_parametric=False)
    assert selector.select(QUERY, pool, 4) == selector.select(QUERY, pool, 4)


def test_invalid_arguments_raise() -> None:
    extractor = MappedExtractor({})
    for kwargs in ({"alpha": 1.5}, {"margin": 0}, {"support_cap": -1}, {"top_n": 0}):
        try:
            GatedCorroborationSelector(extractor, **kwargs)  # type: ignore[arg-type]
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {kwargs}")
    selector = GatedCorroborationSelector(extractor)
    try:
        selector.select(Query(query_id="other", text="t"), candidate_set(), 1)
    except ValueError:
        pass
    else:
        raise AssertionError("expected query ID mismatch error")


class PoolRetriever:
    def __init__(self, pool: CandidateSet) -> None:
        self.pool = pool

    def retrieve(self, query: Query, top_k: int) -> CandidateSet:
        return self.pool


class CitingGenerator:
    def generate(self, query: Query, selected: SelectedEvidenceSet) -> GenerationResult:
        first = selected.evidence[0]
        return GenerationResult(
            query_id=query.query_id,
            answer=first.text,
            cited_evidence_ids=(first.evidence_id,),
        )


def test_pipeline_accepts_gated_shortfall() -> None:
    pool, extractor = three_vs_one_pool()
    pipeline = EvidenceRAGPipeline(
        PoolRetriever(pool),
        GatedCorroborationSelector(extractor, use_parametric=False),
        CitingGenerator(),
    )
    run = pipeline.run_with_trace(QUERY, top_k=4, max_selected=4)
    assert len(run.selection.items) == 3
    assert "lone" not in {item.evidence_id for item in run.selection.items}
```

- [ ] **Step 3.2: Run to verify failure**

Run: `python -m pytest tests/selector/test_gated.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'evidence_rag.selector.gated'`

- [ ] **Step 3.3: Implement gated.py**

```python
"""Gated corroboration selector (spec §7).

Tolerant rerank stage (convex blend, unchanged from CorroborationSelector) followed
by a strict contrastive gate. The gate reads NO scores — only in-pool integer votes:
drop c iff (1) c extracted a valid answer, (2) a competing answer cluster exists,
(3) the winner's independent support beats c's by >= margin, (4) c's support is
<= support_cap. Failure direction is silence: fragmented voting shrinks margins and
the gate stops firing (spec §7.1).
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from evidence_rag.contracts.models import (
    CandidateSet,
    Query,
    SelectionItem,
    SelectionResult,
)
from evidence_rag.selector.answer_norm import is_valid_answer
from evidence_rag.selector.clusters import AnswerCluster, build_clusters
from evidence_rag.selector.corroboration import corroboration_scores, minmax
from evidence_rag.selector.extraction import AnswerExtractionEngine, TextGenerator


@dataclass(frozen=True)
class GateDecision:
    evidence_id: str
    action: Literal["keep", "drop"]
    answer: str | None
    own_support: int
    winner_answer: str | None
    winner_support: int
    margin: int
    has_valid_answer: bool
    has_competitor: bool
    margin_met: bool
    isolation_met: bool


class GatedCorroborationSelector:
    """Corroboration rerank plus contrastive drop gate (spec §5-§7)."""

    def __init__(
        self,
        answer_extractor: TextGenerator,
        *,
        alpha: float = 0.6,
        margin: int = 2,
        support_cap: int = 1,
        top_n: int = 20,
        use_parametric: bool = True,
        passage_chars: int = 600,
        on_gate_decision: Callable[[GateDecision], None] | None = None,
    ) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if margin < 1:
            raise ValueError("margin must be at least 1")
        if support_cap < 0:
            raise ValueError("support_cap must be non-negative")
        if top_n <= 0:
            raise ValueError("top_n must be positive")
        self.alpha = alpha
        self.margin = margin
        self.support_cap = support_cap
        self.top_n = top_n
        self.on_gate_decision = on_gate_decision
        self._engine = AnswerExtractionEngine(
            answer_extractor,
            passage_chars=passage_chars,
            use_parametric=use_parametric,
        )

    def _decide(
        self,
        evidence_id: str,
        answer: str,
        clusters: tuple[AnswerCluster, ...],
        cluster_by_member: dict[str, AnswerCluster],
    ) -> GateDecision:
        has_valid_answer = is_valid_answer(answer)
        own = cluster_by_member.get(evidence_id)
        competitors = tuple(
            cluster for cluster in clusters if own is not None and cluster.answer != own.answer
        )
        winner = max(competitors, key=lambda cluster: cluster.independent_support, default=None)
        own_support = own.independent_support if own is not None else 0
        winner_support = winner.independent_support if winner is not None else 0
        has_competitor = winner is not None
        margin = winner_support - own_support if has_competitor else 0
        margin_met = has_competitor and margin >= self.margin
        isolation_met = own_support <= self.support_cap
        drop = has_valid_answer and has_competitor and margin_met and isolation_met
        return GateDecision(
            evidence_id=evidence_id,
            action="drop" if drop else "keep",
            answer=own.answer if own is not None else None,
            own_support=own_support,
            winner_answer=winner.answer if winner is not None else None,
            winner_support=winner_support,
            margin=margin,
            has_valid_answer=has_valid_answer,
            has_competitor=has_competitor,
            margin_met=margin_met,
            isolation_met=isolation_met,
        )

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

        ranked_candidates = tuple(
            sorted(candidates.candidates, key=lambda item: item.retrieval_rank)
        )
        window = ranked_candidates[: self.top_n]
        tail = ranked_candidates[self.top_n :]
        extracted = self._engine.extract(query, window)
        corroboration = minmax(corroboration_scores(extracted.answers, extracted.parametric))
        relevance = minmax(tuple(candidate.retrieval_score for candidate in window))
        blended = tuple(
            self.alpha * relevance[index] + (1.0 - self.alpha) * corroboration[index]
            for index in range(len(window))
        )

        clusters = build_clusters(window, extracted.answers)
        cluster_by_member = {
            member_id: cluster for cluster in clusters for member_id in cluster.member_ids
        }
        dropped: set[str] = set()
        for index, candidate in enumerate(window):
            decision = self._decide(
                candidate.evidence_id,
                extracted.answers[index],
                clusters,
                cluster_by_member,
            )
            if decision.action == "drop":
                dropped.add(candidate.evidence_id)
            if self.on_gate_decision is not None:
                self.on_gate_decision(decision)

        surviving_window = tuple(
            window[index]
            for index in sorted(
                range(len(window)),
                key=lambda i: (-blended[i], window[i].retrieval_rank, window[i].evidence_id),
            )
            if window[index].evidence_id not in dropped
        )
        score_by_id = {
            candidate.evidence_id: blended[index]
            for index, candidate in enumerate(window)
        }
        output = (surviving_window + tail)[:max_selected]
        return SelectionResult(
            query_id=query.query_id,
            items=tuple(
                SelectionItem(
                    evidence_id=candidate.evidence_id,
                    selection_score=score_by_id.get(
                        candidate.evidence_id,
                        candidate.retrieval_score,
                    ),
                    selection_rank=rank,
                )
                for rank, candidate in enumerate(output, start=1)
            ),
        )
```

- [ ] **Step 3.4: Run to verify pass**

Run: `python -m pytest tests/selector/test_gated.py -v`
Expected: PASS (14 tests)

- [ ] **Step 3.5: Commit**

```bash
git add src/evidence_rag/selector/gated.py tests/selector/test_gated.py
git commit -m "feat(selector): gated corroboration selector with contrastive drop gate"
```

### Task 4: composition registration

**Files:** Modify `src/evidence_rag/composition.py` (imports + `build_selector`); Test `tests/pipeline/test_selector_registration.py`

- [ ] **Step 4.1: Write the failing tests**

```python
import pytest

from evidence_rag.composition import build_selector
from evidence_rag.infrastructure.config import ModuleConfig
from evidence_rag.selector.corroboration import CorroborationSelector
from evidence_rag.selector.gated import GatedCorroborationSelector
from evidence_rag.selector.top_k import TopKSelector


class FakeLLM:
    def generate(self, prompt: str) -> str:
        return "NONE"


def test_top_k_still_builds() -> None:
    assert isinstance(build_selector(ModuleConfig(name="top-k")), TopKSelector)


def test_corroboration_builds_with_parameters() -> None:
    selector = build_selector(
        ModuleConfig(name="corroboration", parameters={"alpha": 0.4, "top_n": 10}),
        llm=FakeLLM(),
    )
    assert isinstance(selector, CorroborationSelector)
    assert selector.alpha == 0.4
    assert selector.top_n == 10


def test_gated_corroboration_builds_with_parameters() -> None:
    selector = build_selector(
        ModuleConfig(
            name="gated-corroboration",
            parameters={"alpha": 0.6, "margin": 3, "support_cap": 2, "top_n": 15},
        ),
        llm=FakeLLM(),
    )
    assert isinstance(selector, GatedCorroborationSelector)
    assert selector.margin == 3
    assert selector.support_cap == 2
    assert selector.top_n == 15


def test_gated_corroboration_defaults_match_spec() -> None:
    selector = build_selector(ModuleConfig(name="gated-corroboration"), llm=FakeLLM())
    assert isinstance(selector, GatedCorroborationSelector)
    assert selector.margin == 2
    assert selector.support_cap == 1


def test_unknown_parameter_rejected() -> None:
    with pytest.raises(ValueError, match="unknown selector parameter"):
        build_selector(
            ModuleConfig(name="gated-corroboration", parameters={"tau": 0.3}),
            llm=FakeLLM(),
        )


def test_out_of_range_parameters_rejected() -> None:
    for parameters in ({"alpha": 1.5}, {"margin": 0}, {"support_cap": -1}, {"top_n": 0}):
        with pytest.raises(ValueError, match="invalid selector parameter"):
            build_selector(
                ModuleConfig(name="gated-corroboration", parameters=parameters),
                llm=FakeLLM(),
            )


def test_non_integer_margin_rejected() -> None:
    with pytest.raises(ValueError, match="invalid selector parameter"):
        build_selector(
            ModuleConfig(name="gated-corroboration", parameters={"margin": 1.5}),
            llm=FakeLLM(),
        )


def test_top_k_rejects_parameters() -> None:
    with pytest.raises(ValueError):
        build_selector(ModuleConfig(name="top-k", parameters={"alpha": 0.5}))


def test_unknown_selector_rejected() -> None:
    with pytest.raises(ValueError, match="unknown selector"):
        build_selector(ModuleConfig(name="mystery"))
```

- [ ] **Step 4.2: Run to verify failure**

Run: `python -m pytest tests/pipeline/test_selector_registration.py -v`
Expected: FAIL — `TypeError: build_selector() got an unexpected keyword argument 'llm'`

- [ ] **Step 4.3: Implement registration in composition.py**

Add imports:

```python
from evidence_rag.selector.gated import GatedCorroborationSelector
```

Replace `build_selector` with:

```python
def _float_parameter(name: str, value: object, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"invalid selector parameter {name}: expected a number")
    number = float(value)
    if not low <= number <= high:
        raise ValueError(f"invalid selector parameter {name}: must be in [{low}, {high}]")
    return number


def _int_parameter(name: str, value: object, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"invalid selector parameter {name}: expected an integer")
    if value < minimum:
        raise ValueError(f"invalid selector parameter {name}: must be >= {minimum}")
    return value


def _corroboration_parameters(config: ModuleConfig) -> dict[str, float | int]:
    allowed = {"alpha", "top_n"}
    unknown = sorted(set(config.parameters) - allowed)
    if unknown:
        raise ValueError(f"unknown selector parameter: {unknown[0]}")
    parameters: dict[str, float | int] = {}
    if "alpha" in config.parameters:
        parameters["alpha"] = _float_parameter("alpha", config.parameters["alpha"], 0.0, 1.0)
    if "top_n" in config.parameters:
        parameters["top_n"] = _int_parameter("top_n", config.parameters["top_n"], 1)
    return parameters


def _gated_parameters(config: ModuleConfig) -> dict[str, float | int]:
    allowed = {"alpha", "margin", "support_cap", "top_n"}
    unknown = sorted(set(config.parameters) - allowed)
    if unknown:
        raise ValueError(f"unknown selector parameter: {unknown[0]}")
    parameters: dict[str, float | int] = {}
    if "alpha" in config.parameters:
        parameters["alpha"] = _float_parameter("alpha", config.parameters["alpha"], 0.0, 1.0)
    if "margin" in config.parameters:
        parameters["margin"] = _int_parameter("margin", config.parameters["margin"], 1)
    if "support_cap" in config.parameters:
        parameters["support_cap"] = _int_parameter(
            "support_cap", config.parameters["support_cap"], 0
        )
    if "top_n" in config.parameters:
        parameters["top_n"] = _int_parameter("top_n", config.parameters["top_n"], 1)
    return parameters


def build_selector(config: ModuleConfig, *, llm: TextGenerator | None = None) -> Selector:
    if config.name == "top-k":
        _reject_parameters(config, "selector")
        return TopKSelector()
    if config.name == "corroboration":
        parameters = _corroboration_parameters(config)
        client = llm if llm is not None else GraniteLLMClient()
        return CorroborationSelector(client, **parameters)
    if config.name == "gated-corroboration":
        parameters = _gated_parameters(config)
        client = llm if llm is not None else GraniteLLMClient()
        return GatedCorroborationSelector(client, **parameters)
    raise ValueError(f"unknown selector: {config.name}")
```

Note: `GraniteLLMClient` and `TextGenerator` are already imported at the top of composition.py. `**parameters` unpacking into typed keyword-only arguments is accepted by mypy because the values are `float | int` and the signature takes `float`/`int` — if mypy strict rejects the unpack, replace with explicit keyword forwarding:

```python
        return GatedCorroborationSelector(
            client,
            alpha=float(parameters.get("alpha", 0.6)),
            margin=int(parameters.get("margin", 2)),
            support_cap=int(parameters.get("support_cap", 1)),
            top_n=int(parameters.get("top_n", 20)),
        )
```

(and the equivalent for `CorroborationSelector` with `alpha`/`top_n`). Use the explicit form directly if in doubt — it is the deterministic choice.

- [ ] **Step 4.4: Run to verify pass**

Run: `python -m pytest tests/pipeline/test_selector_registration.py -v`
Expected: PASS (9 tests)

- [ ] **Step 4.5: Full suite + typecheck**

Run: `python -m pytest && python -m mypy`
Expected: all tests pass; mypy strict clean.

- [ ] **Step 4.6: Commit**

```bash
git add src/evidence_rag/composition.py tests/pipeline/test_selector_registration.py
git commit -m "feat(composition): register corroboration and gated-corroboration selectors"
```

### Task 5: HPC experiment triple (E2) — marked, not submitted

**Files:** Create `scripts/run_selector_gate.slurm`, `docs/hpc-run-log.md`

- [ ] **Step 5.1: Verify GraniteLLMClient env-var handling**

Read `src/evidence_rag/generator/granite.py` and confirm which env var (if any) selects the model id. Record the finding in the slurm header comment. If no env var exists, the header must say the model id is code-default and note where to change it.

- [ ] **Step 5.2: Create `scripts/run_selector_gate.slurm`**

Skeleton (env block verbatim from `week4_SPLADE:scripts/run_niah_rag.slurm`; adjust the model-id line per Step 5.1's finding):

```bash
#!/bin/bash
# E2: gate-on/off paired selector comparison (spec §12 main comparison).
# Two configs differ ONLY in [selector]: gated-corroboration vs corroboration.
# BLOCKED until the §14 dataset decision produces the two TOMLs — do not submit before.
#   mkdir -p logs runs && sbatch scripts/run_selector_gate.slurm \
#     configs/experiments/<gate_on>.toml configs/experiments/<gate_off>.toml
#SBATCH --job-name=selector-gate
#SBATCH --account=coms039904
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
CONFIG_ON="${1:?usage: sbatch run_selector_gate.slurm CONFIG_ON CONFIG_OFF}"
CONFIG_OFF="${2:?usage: sbatch run_selector_gate.slurm CONFIG_ON CONFIG_OFF}"

module load languages/python/3.12.3
export HF_HOME=/user/work/$USER/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export IR_DATASETS_HOME=/user/work/$USER/ir_datasets PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /user/work/$USER/venv/bin/activate
cd /user/work/$USER/IBM_Granite_Project

python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| torch', torch.__version__)"

for CONFIG in "$CONFIG_ON" "$CONFIG_OFF"; do
  echo "=== running $CONFIG ==="
  python -m evidence_rag.cli.experiment --config "$CONFIG" all
done
echo "=== selector reports ==="
for CONFIG in "$CONFIG_ON" "$CONFIG_OFF"; do
  python - "$CONFIG" <<'PY'
import sys, tomllib, pathlib, json
config_path = pathlib.Path(sys.argv[1]).resolve()
raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
output = (config_path.parent / raw["output"]["directory"]).resolve()
report = output / "selector_report.json"
print(config_path.name, "->", report)
print(report.read_text(encoding="utf-8") if report.is_file() else "MISSING")
PY
done
```

- [ ] **Step 5.3: Create `docs/hpc-run-log.md` with pre-registered BLOCKED entries**

Ledger content: header explaining the triple discipline, then E1/E2 entries with purpose, hypothesis, expected metric + direction, blockers, and empty AFTER sections. (Full text in the execution step; E1's slurm ships later WITH the component-eval harness — its code does not exist yet, and pre-writing a script for a nonexistent entry point would invent CLI flags.)

- [ ] **Step 5.4: Commit**

```bash
git add scripts/run_selector_gate.slurm docs/hpc-run-log.md
git commit -m "ops(hpc): E2 gate-on/off slurm triple and pre-registered run ledger"
```

---

## Self-review notes

- Spec coverage: §6.2→Task 1; §6.1/§6.3→Task 2; §7/§7.2/§11→Task 3 (sink = `on_gate_decision`); registration (spec §10 row 3)→Task 4; §12 E2 vehicle + running-hpc-experiments triple→Task 5. §12 component eval (E1) is intentionally NOT implemented here — blocked on §14 datasets; ledger pre-registers it.
- No placeholders: every code step is complete; the two `<gate_on>/<gate_off>` names in the slurm header are argument documentation for files whose creation is explicitly blocked on §14, stated as such.
- Type consistency: `TextGenerator` protocol lives in `extraction.py`; `corroboration.py` re-exports it; composition already imports it from `generator.granite` — both are structural `Protocol`s with the same `generate` method, so either satisfies the annotation.
