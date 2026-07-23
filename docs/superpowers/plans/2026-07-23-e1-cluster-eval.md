# E1 Cluster-Eval Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone, offline-first harness that measures the E1 edge/cluster error modes (missed-conflict, false-conflict, needle-gold-recovery, dedup determinism, injection selection-bias) over the real E2 pools, reusing the gate's exact `build_clusters`.

**Architecture:** A pure metric module (`cluster_eval.py`, no I/O, no LLM) computes per-query cases and aggregates them with Wilson CIs; a thin CLI (`cluster_eval_cli.py`) loads the injected dataset + E2 `candidate_sets.jsonl` + provenance, runs one Granite extraction pass over each injected query's top_n window, and calls the pure module. No frozen contract, no `gated.py`, and no `ExperimentWorkflow` is touched. Ships as an HPC triple (module+CLI+tests, slurm, ledger).

**Tech Stack:** Python 3.12, pydantic v2, pytest, Granite via `GraniteLLMClient`, slurm on BluePebble.

**Spec:** `docs/superpowers/specs/2026-07-23-e1-cluster-eval-design.md`

---

## File Structure

- Create `src/evidence_rag/evaluation/cluster_eval.py` — pure functions: `contains_alias`, `evaluate_case`, `wilson_interval`, `summarize`, `aggregate`, `selection_bias`, and the frozen result dataclasses. Imports the real `build_clusters` + `canonicalize_answer`.
- Create `src/evidence_rag/evaluation/cluster_eval_cli.py` — argparse + jsonl loaders + Granite extraction loop + report writer. `main(argv, *, llm=None)` so tests inject a fake extractor.
- Create `tests/evaluation/test_cluster_eval.py` — CPU unit tests for the pure module.
- Create `tests/evaluation/test_cluster_eval_cli.py` — CLI assembly test with a fake `TextGenerator` + tmp files.
- Create `scripts/run_cluster_eval.slurm` — GPU triple leg; env block copied verbatim from `run_selector_gate.slurm`.
- Modify `docs/hpc-run-log.md` — fill the E1 entry's BEFORE (pre-registration).

---

## Task 1: Pure metric core — `contains_alias` + `evaluate_case`

**Files:**
- Create: `src/evidence_rag/evaluation/cluster_eval.py`
- Test: `tests/evaluation/test_cluster_eval.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/evaluation/test_cluster_eval.py
from evidence_rag.contracts.models import EvidenceCandidate
from evidence_rag.evaluation.cluster_eval import contains_alias, evaluate_case


def _cand(evidence_id: str, document_id: str, text: str, rank: int) -> EvidenceCandidate:
    return EvidenceCandidate(
        evidence_id=evidence_id,
        document_id=document_id,
        chunk_id=f"{document_id}::c0",
        text=text,
        source_uri=f"synthetic://{document_id}",
        retrieval_score=1.0 / rank,
        retrieval_rank=rank,
    )


def test_contains_alias_word_boundary():
    assert contains_alias("born in Paris today", ["Paris"]) is True
    assert contains_alias("Parisian cafe", ["Paris"]) is False
    assert contains_alias("no match here", ["Paris", "Lyon"]) is False


def test_missed_conflict_true_when_needle_and_cf_share_cluster():
    # needle doc and cf doc both extract to "kennedy" -> same cluster -> gate blind.
    window = [
        _cand("e_needle", "needle", "the needle passage", 1),
        _cand("e_cf", "cf::needle", "the counterfactual passage", 2),
    ]
    answers = ["Kennedy", "Kennedy"]
    case = evaluate_case(
        window,
        answers,
        query_id="q1",
        needle_document_id="needle",
        counterfactual_document_id="cf::needle",
        gold_value="Kennedy",
        gold_aliases=("Kennedy",),
    )
    assert case.missed_conflict is True
    assert case.needle_gold_recovery is True


def test_missed_conflict_false_when_distinct_clusters():
    window = [
        _cand("e_needle", "needle", "the needle passage", 1),
        _cand("e_cf", "cf::needle", "the counterfactual passage", 2),
    ]
    answers = ["Kennedy", "Nixon"]
    case = evaluate_case(
        window,
        answers,
        query_id="q1",
        needle_document_id="needle",
        counterfactual_document_id="cf::needle",
        gold_value="Kennedy",
        gold_aliases=("Kennedy",),
    )
    assert case.missed_conflict is False


def test_missed_conflict_unscored_when_needle_has_no_valid_answer():
    window = [
        _cand("e_needle", "needle", "the needle passage", 1),
        _cand("e_cf", "cf::needle", "the counterfactual passage", 2),
    ]
    answers = ["NONE", "Nixon"]  # needle not clustered
    case = evaluate_case(
        window,
        answers,
        query_id="q1",
        needle_document_id="needle",
        counterfactual_document_id="cf::needle",
        gold_value="Kennedy",
        gold_aliases=("Kennedy",),
    )
    assert case.missed_conflict is None
    assert case.needle_gold_recovery is False  # in window, but did not recover gold


def test_false_conflict_true_when_second_gold_doc_splits():
    # needle -> "Kennedy"; another gold-bearing haystack doc -> "JFK" (different canonical).
    window = [
        _cand("e_needle", "needle", "answer is Kennedy", 1),
        _cand("e_other", "hay1", "also mentions JFK here", 2),
        _cand("e_cf", "cf::needle", "counterfactual", 3),
    ]
    answers = ["Kennedy", "JFK", "Nixon"]
    case = evaluate_case(
        window,
        answers,
        query_id="q1",
        needle_document_id="needle",
        counterfactual_document_id="cf::needle",
        gold_value="Kennedy",
        gold_aliases=("Kennedy", "JFK"),
    )
    assert case.false_conflict is True


def test_false_conflict_unscored_when_no_second_gold_doc():
    window = [
        _cand("e_needle", "needle", "answer is Kennedy", 1),
        _cand("e_cf", "cf::needle", "counterfactual", 2),
    ]
    answers = ["Kennedy", "Nixon"]
    case = evaluate_case(
        window,
        answers,
        query_id="q1",
        needle_document_id="needle",
        counterfactual_document_id="cf::needle",
        gold_value="Kennedy",
        gold_aliases=("Kennedy", "JFK"),
    )
    assert case.false_conflict is None


def test_needle_not_in_window_leaves_metrics_unscored():
    window = [_cand("e_cf", "cf::needle", "counterfactual", 1)]
    answers = ["Nixon"]
    case = evaluate_case(
        window,
        answers,
        query_id="q1",
        needle_document_id="needle",
        counterfactual_document_id="cf::needle",
        gold_value="Kennedy",
        gold_aliases=("Kennedy",),
    )
    assert case.needle_in_window is False
    assert case.cf_in_window is True
    assert case.missed_conflict is None
    assert case.false_conflict is None
    assert case.needle_gold_recovery is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/evaluation/test_cluster_eval.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'evidence_rag.evaluation.cluster_eval'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/evidence_rag/evaluation/cluster_eval.py
"""E1 edge/cluster component eval (spec docs/superpowers/specs/2026-07-23-e1-cluster-eval-design.md).

Offline over the gate's own build_clusters — measures how faithfully the exact-string
answer clusters capture agreement/conflict on the injected NIAH pools. No LLM here; the
CLI feeds extracted answers. Statistic unit = injected query.
"""

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from math import sqrt

from evidence_rag.contracts.models import EvidenceCandidate
from evidence_rag.selector.answer_norm import canonicalize_answer
from evidence_rag.selector.clusters import build_clusters

MISSED_CONFLICT = "selector.cluster.missed_conflict"
FALSE_CONFLICT = "selector.cluster.false_conflict"
NEEDLE_GOLD_RECOVERY = "selector.cluster.needle_gold_recovery"


def contains_alias(text: str, aliases: Sequence[str]) -> bool:
    """Word-boundary alias presence, mirroring the injector's _occurrences pattern."""
    return any(
        re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", text) is not None for alias in aliases
    )


@dataclass(frozen=True)
class ClusterEvalCase:
    query_id: str
    needle_in_window: bool
    cf_in_window: bool
    missed_conflict: bool | None
    false_conflict: bool | None
    needle_gold_recovery: bool | None


def evaluate_case(
    window: Sequence[EvidenceCandidate],
    answers: Sequence[str],
    *,
    query_id: str,
    needle_document_id: str,
    counterfactual_document_id: str,
    gold_value: str,
    gold_aliases: Sequence[str],
) -> ClusterEvalCase:
    clusters = build_clusters(window, answers)
    cluster_answer_by_member = {
        member_id: cluster.answer
        for cluster in clusters
        for member_id in cluster.member_ids
    }
    needle_candidates = [c for c in window if c.document_id == needle_document_id]
    cf_candidates = [c for c in window if c.document_id == counterfactual_document_id]
    needle_cluster_answers = {
        cluster_answer_by_member[c.evidence_id]
        for c in needle_candidates
        if c.evidence_id in cluster_answer_by_member
    }
    cf_cluster_answers = {
        cluster_answer_by_member[c.evidence_id]
        for c in cf_candidates
        if c.evidence_id in cluster_answer_by_member
    }
    needle_in_window = bool(needle_candidates)
    cf_in_window = bool(cf_candidates)
    needle_clustered = bool(needle_cluster_answers)

    if needle_clustered and cf_cluster_answers:
        missed_conflict: bool | None = bool(needle_cluster_answers & cf_cluster_answers)
    else:
        missed_conflict = None

    if needle_in_window:
        needle_gold_recovery: bool | None = (
            canonicalize_answer(gold_value) in needle_cluster_answers
        )
    else:
        needle_gold_recovery = None

    false_conflict: bool | None = None
    if needle_clustered:
        other_gold_clusters = {
            cluster_answer_by_member[c.evidence_id]
            for c in window
            if c.document_id != needle_document_id
            and c.evidence_id in cluster_answer_by_member
            and contains_alias(c.text, gold_aliases)
        }
        if other_gold_clusters:
            false_conflict = bool(other_gold_clusters - needle_cluster_answers)

    return ClusterEvalCase(
        query_id=query_id,
        needle_in_window=needle_in_window,
        cf_in_window=cf_in_window,
        missed_conflict=missed_conflict,
        false_conflict=false_conflict,
        needle_gold_recovery=needle_gold_recovery,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/evaluation/test_cluster_eval.py -q`
Expected: PASS (7 passed).

- [ ] **Step 5: Commit**

```bash
git add src/evidence_rag/evaluation/cluster_eval.py tests/evaluation/test_cluster_eval.py
git commit -m "feat(selector): E1 cluster-eval per-case metrics (missed/false-conflict, recovery)"
```

---

## Task 2: Wilson CI + aggregation

**Files:**
- Modify: `src/evidence_rag/evaluation/cluster_eval.py`
- Test: `tests/evaluation/test_cluster_eval.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/evaluation/test_cluster_eval.py
from evidence_rag.evaluation.cluster_eval import (
    ClusterEvalCase,
    aggregate,
    summarize,
    wilson_interval,
)


def test_wilson_interval_bounds():
    low, high = wilson_interval(5, 10)
    assert 0.0 <= low < 0.5 < high <= 1.0
    assert wilson_interval(0, 0) == (0.0, 0.0)
    zlow, zhigh = wilson_interval(0, 20)
    assert zlow == 0.0
    assert 0.0 < zhigh < 0.3


def test_summarize_skips_none():
    summary = summarize([True, False, None, True])
    assert summary.n_scored == 3
    assert summary.successes == 2
    assert summary.rate == 2 / 3
    assert summary.ci_low <= summary.rate <= summary.ci_high


def test_summarize_all_none_is_unscored():
    summary = summarize([None, None])
    assert summary.n_scored == 0
    assert summary.rate is None


def test_aggregate_counts_window_presence():
    cases = [
        ClusterEvalCase("q1", True, True, True, None, True),
        ClusterEvalCase("q2", True, False, None, False, False),
    ]
    report = aggregate(cases)
    assert report.n_cases == 2
    assert report.needle_in_window == 2
    assert report.cf_in_window == 1
    assert report.missed_conflict.n_scored == 1
    assert report.needle_gold_recovery.n_scored == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/evaluation/test_cluster_eval.py -q`
Expected: FAIL with `ImportError: cannot import name 'aggregate'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/evidence_rag/evaluation/cluster_eval.py
@dataclass(frozen=True)
class MetricSummary:
    rate: float | None
    n_scored: int
    successes: int
    ci_low: float
    ci_high: float


@dataclass(frozen=True)
class ClusterEvalReport:
    missed_conflict: MetricSummary
    false_conflict: MetricSummary
    needle_gold_recovery: MetricSummary
    n_cases: int
    needle_in_window: int
    cf_in_window: int


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    phat = successes / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = z * sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def summarize(values: Iterable[bool | None]) -> MetricSummary:
    scored = [value for value in values if value is not None]
    n = len(scored)
    successes = sum(1 for value in scored if value)
    if n == 0:
        return MetricSummary(rate=None, n_scored=0, successes=0, ci_low=0.0, ci_high=0.0)
    low, high = wilson_interval(successes, n)
    return MetricSummary(
        rate=successes / n, n_scored=n, successes=successes, ci_low=low, ci_high=high
    )


def aggregate(cases: Sequence[ClusterEvalCase]) -> ClusterEvalReport:
    return ClusterEvalReport(
        missed_conflict=summarize(case.missed_conflict for case in cases),
        false_conflict=summarize(case.false_conflict for case in cases),
        needle_gold_recovery=summarize(case.needle_gold_recovery for case in cases),
        n_cases=len(cases),
        needle_in_window=sum(1 for case in cases if case.needle_in_window),
        cf_in_window=sum(1 for case in cases if case.cf_in_window),
    )
```

Add `Sequence` is already imported. Add nothing else.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/evaluation/test_cluster_eval.py -q`
Expected: PASS (11 passed).

- [ ] **Step 5: Commit**

```bash
git add src/evidence_rag/evaluation/cluster_eval.py tests/evaluation/test_cluster_eval.py
git commit -m "feat(selector): E1 aggregation with Wilson CIs"
```

---

## Task 3: Injection selection-bias counter

**Files:**
- Modify: `src/evidence_rag/evaluation/cluster_eval.py`
- Test: `tests/evaluation/test_cluster_eval.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/evaluation/test_cluster_eval.py
from evidence_rag.evaluation.cluster_eval import selection_bias


def test_selection_bias_counts_multi_canonical_key():
    # "JFK"/"John F. Kennedy" -> two canonical keys -> multi-key skip.
    # "1,200"/"1200" -> one canonical key (number canonicalization) -> not multi-key.
    reference_answers = [
        ("JFK", "John F. Kennedy"),
        ("1,200", "1200"),
        ("Paris",),
        None,
    ]
    bias = selection_bias(reference_answers, n_injected=2)
    assert bias.n_gold_cases == 4
    assert bias.n_answerable == 3
    assert bias.n_multi_key == 1
    assert bias.multi_key_rate == 1 / 3
    assert bias.skip_rate == (4 - 2) / 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/evaluation/test_cluster_eval.py::test_selection_bias_counts_multi_canonical_key -q`
Expected: FAIL with `ImportError: cannot import name 'selection_bias'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/evidence_rag/evaluation/cluster_eval.py
@dataclass(frozen=True)
class SelectionBias:
    n_gold_cases: int
    n_answerable: int
    n_multi_key: int
    multi_key_rate: float | None
    n_injected: int
    skip_rate: float | None


def selection_bias(
    reference_answers_per_case: Iterable[Sequence[str] | None],
    *,
    n_injected: int,
) -> SelectionBias:
    n_total = 0
    n_answerable = 0
    n_multi = 0
    for answers in reference_answers_per_case:
        n_total += 1
        if not answers:
            continue
        n_answerable += 1
        if len({canonicalize_answer(answer) for answer in answers}) != 1:
            n_multi += 1
    return SelectionBias(
        n_gold_cases=n_total,
        n_answerable=n_answerable,
        n_multi_key=n_multi,
        multi_key_rate=(n_multi / n_answerable) if n_answerable else None,
        n_injected=n_injected,
        skip_rate=((n_total - n_injected) / n_total) if n_total else None,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/evaluation/test_cluster_eval.py -q`
Expected: PASS (12 passed).

- [ ] **Step 5: Commit**

```bash
git add src/evidence_rag/evaluation/cluster_eval.py tests/evaluation/test_cluster_eval.py
git commit -m "feat(selector): E1 injection selection-bias counter"
```

---

## Task 4: CLI orchestration (`cluster_eval_cli.py`)

**Files:**
- Create: `src/evidence_rag/evaluation/cluster_eval_cli.py`
- Test: `tests/evaluation/test_cluster_eval_cli.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/evaluation/test_cluster_eval_cli.py
import json
from pathlib import Path

from evidence_rag.contracts.models import CandidateSet, Document, EvidenceCandidate, Query
from evidence_rag.evaluation.cluster_eval_cli import main
from evidence_rag.infrastructure.datasets import GoldCase
from evidence_rag.materializer.provenance import MutationRecord, write_provenance


class FakeLLM:
    """Deterministic extractor: returns the answer keyed by the passage's marker line."""

    def __init__(self, answer_by_marker: dict[str, str]) -> None:
        self._answers = answer_by_marker

    def generate(self, prompt: str) -> str:
        for marker, answer in self._answers.items():
            if marker in prompt:
                return answer
        return "NONE"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _manifest(tmp_path: Path) -> Path:
    documents = [
        {"schema_version": "1.0", "document_id": "needle", "text": "MARK_NEEDLE Kennedy won",
         "source_uri": "s://needle"},
        {"schema_version": "1.0", "document_id": "cf::needle", "text": "MARK_CF Nixon won",
         "source_uri": "s://cf"},
    ]
    queries = [{"schema_version": "1.0", "query_id": "q1", "text": "who won?"}]
    gold_cases = [
        {"query_id": "q1", "relevant_document_ids": ["needle"], "reference_answers": ["Kennedy"]}
    ]
    _write_jsonl(tmp_path / "documents.jsonl", documents)
    _write_jsonl(tmp_path / "queries.jsonl", queries)
    _write_jsonl(tmp_path / "gold_cases.jsonl", gold_cases)
    manifest = {
        "schema_version": "1.0", "dataset_id": "niah", "dataset_version": "test+cf42",
        "split": "dev", "documents_file": "documents.jsonl", "queries_file": "queries.jsonl",
        "gold_cases_file": "gold_cases.jsonl",
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path / "manifest.json"


def _candidates(tmp_path: Path) -> Path:
    candidate_set = CandidateSet(
        query_id="q1",
        candidates=(
            EvidenceCandidate(
                evidence_id="e_needle", document_id="needle", chunk_id="needle::c0",
                text="MARK_NEEDLE Kennedy won", source_uri="s://needle",
                retrieval_score=2.0, retrieval_rank=1,
            ),
            EvidenceCandidate(
                evidence_id="e_cf", document_id="cf::needle", chunk_id="cf::needle::c0",
                text="MARK_CF Nixon won", source_uri="s://cf",
                retrieval_score=1.0, retrieval_rank=2,
            ),
        ),
    )
    path = tmp_path / "candidate_sets.jsonl"
    path.write_text(candidate_set.model_dump_json() + "\n", encoding="utf-8")
    return path


def test_cli_writes_report_with_fake_extractor(tmp_path, capsys):
    manifest = _manifest(tmp_path)
    candidates = _candidates(tmp_path)
    provenance = tmp_path / "provenance.jsonl"
    write_provenance(
        provenance,
        [
            MutationRecord(
                query_id="q1", needle_document_id="needle",
                counterfactual_document_id="cf::needle", gold_value="Kennedy",
                gold_alias_used="Kennedy", replacement_value="Nixon", string_class="name-1",
                seed=42, char_span=(0, 7), text_hash_before="a", text_hash_after="b",
                answer_bank_hash="h",
            )
        ],
    )
    output = tmp_path / "cluster_eval_report.json"

    exit_code = main(
        [
            "--manifest", str(manifest),
            "--candidates", str(candidates),
            "--provenance", str(provenance),
            "--output", str(output),
        ],
        llm=FakeLLM({"MARK_NEEDLE": "Kennedy", "MARK_CF": "Nixon"}),
    )

    assert exit_code == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    # needle -> Kennedy, cf -> Nixon: distinct clusters, gold recovered, no missed conflict.
    assert report["needle_gold_recovery"]["rate"] == 1.0
    assert report["missed_conflict"]["rate"] == 0.0
    assert report["selection_bias"]["n_injected"] == 1
    printed = capsys.readouterr().out
    assert "missed_conflict" in printed
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/evaluation/test_cluster_eval_cli.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'evidence_rag.evaluation.cluster_eval_cli'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/evidence_rag/evaluation/cluster_eval_cli.py
"""CLI: offline E1 edge/cluster component eval over the injected NIAH pools (spec §12).

Reuses the E2 candidate_sets.jsonl (pools carry passage text), runs one Granite extraction
pass over each injected query's top_n window, and reports missed/false-conflict, needle-gold
recovery (Wilson CIs), plus the deterministic injection selection-bias framing line.
"""

import argparse
import dataclasses
import json
from collections.abc import Sequence
from pathlib import Path

from evidence_rag.contracts.models import CandidateSet
from evidence_rag.evaluation.cluster_eval import aggregate, evaluate_case, selection_bias
from evidence_rag.generator.granite import GraniteLLMClient, TextGenerator
from evidence_rag.infrastructure.datasets import JsonlDatasetAdapter
from evidence_rag.materializer.provenance import read_provenance
from evidence_rag.selector.extraction import AnswerExtractionEngine


def _read_candidates(path: Path) -> dict[str, CandidateSet]:
    text = Path(path).read_text(encoding="utf-8")
    sets = (
        CandidateSet.model_validate_json(line) for line in text.splitlines() if line.strip()
    )
    return {candidate_set.query_id: candidate_set for candidate_set in sets}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline E1 edge/cluster component eval")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--provenance", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--passage-chars", type=int, default=600)
    return parser


def main(argv: Sequence[str] | None = None, *, llm: TextGenerator | None = None) -> int:
    arguments = _parser().parse_args(argv)
    bundle = JsonlDatasetAdapter.load(arguments.manifest)
    query_by_id = {query.query_id: query for query in bundle.queries}
    gold_by_id = {gold_case.query_id: gold_case for gold_case in bundle.gold_cases}
    candidates_by_id = _read_candidates(arguments.candidates)
    records = read_provenance(arguments.provenance)

    engine = AnswerExtractionEngine(
        llm if llm is not None else GraniteLLMClient(),
        passage_chars=arguments.passage_chars,
        use_parametric=False,
    )

    cases = []
    for record in records:
        query = query_by_id[record.query_id]
        candidate_set = candidates_by_id[record.query_id]
        gold_case = gold_by_id.get(record.query_id)
        window = tuple(
            sorted(candidate_set.candidates, key=lambda item: item.retrieval_rank)
        )[: arguments.top_n]
        extracted = engine.extract(query, window)
        cases.append(
            evaluate_case(
                window,
                extracted.answers,
                query_id=record.query_id,
                needle_document_id=record.needle_document_id,
                counterfactual_document_id=record.counterfactual_document_id,
                gold_value=record.gold_value,
                gold_aliases=(gold_case.reference_answers or ()) if gold_case else (),
            )
        )

    report = aggregate(cases)
    bias = selection_bias(
        (gold_case.reference_answers for gold_case in bundle.gold_cases),
        n_injected=len(records),
    )
    payload = {
        "missed_conflict": dataclasses.asdict(report.missed_conflict),
        "false_conflict": dataclasses.asdict(report.false_conflict),
        "needle_gold_recovery": dataclasses.asdict(report.needle_gold_recovery),
        "n_cases": report.n_cases,
        "needle_in_window": report.needle_in_window,
        "cf_in_window": report.cf_in_window,
        "selection_bias": dataclasses.asdict(bias),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/evaluation/test_cluster_eval_cli.py -q`
Expected: PASS (1 passed).

- [ ] **Step 5: Run the whole evaluation suite to confirm no regressions**

Run: `PYTHONPATH=src pytest tests/evaluation -q`
Expected: PASS (all green).

- [ ] **Step 6: Commit**

```bash
git add src/evidence_rag/evaluation/cluster_eval_cli.py tests/evaluation/test_cluster_eval_cli.py
git commit -m "feat(selector): E1 cluster-eval CLI over injected NIAH pools"
```

---

## Task 5: HPC triple — slurm + ledger pre-registration

**Files:**
- Create: `scripts/run_cluster_eval.slurm`
- Modify: `docs/hpc-run-log.md` (E1 entry BEFORE)

- [ ] **Step 1: Write the slurm script**

Copy the env block verbatim from `scripts/run_selector_gate.slurm`. Content:

```bash
#!/bin/bash
# E1: edge/cluster component eval over the injected NIAH pools
# (spec docs/superpowers/specs/2026-07-23-e1-cluster-eval-design.md).
# Reuses E2's candidate_sets.jsonl (pools carry text); one Granite extraction pass over each
# injected query's top_n window -> missed/false-conflict + needle-gold-recovery + selection-bias.
#
# One-time on the login node before submitting (E2 must have produced the pool dump):
#   ls runs/e2-gate-on/candidate_sets.jsonl runs/niah-injected/manifest.json runs/niah-injected/provenance.jsonl
# Submit from the project root:
#   mkdir -p logs results && sbatch scripts/run_cluster_eval.slurm \
#     runs/e2-gate-on/candidate_sets.jsonl runs/niah-injected/manifest.json \
#     runs/niah-injected/provenance.jsonl results/e1-cluster-eval/cluster_eval_report.json
#SBATCH --job-name=cluster-eval
#SBATCH --account=coms039904
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
CANDIDATES="${1:?usage: sbatch run_cluster_eval.slurm CANDIDATES MANIFEST PROVENANCE OUTPUT}"
MANIFEST="${2:?usage: sbatch run_cluster_eval.slurm CANDIDATES MANIFEST PROVENANCE OUTPUT}"
PROVENANCE="${3:?usage: sbatch run_cluster_eval.slurm CANDIDATES MANIFEST PROVENANCE OUTPUT}"
OUTPUT="${4:?usage: sbatch run_cluster_eval.slurm CANDIDATES MANIFEST PROVENANCE OUTPUT}"

module load languages/python/3.12.3
export HF_HOME=/user/work/$USER/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export IR_DATASETS_HOME=/user/work/$USER/ir_datasets PYTHONUNBUFFERED=1
export GRANITE_MODEL_ID=ibm-granite/granite-4.1-3b
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /user/work/$USER/venv/bin/activate
cd /user/work/$USER/IBM_Granite_Project
export PYTHONPATH=src

python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| torch', torch.__version__)"

echo "=== E1 edge/cluster component eval ==="
python -m evidence_rag.evaluation.cluster_eval_cli \
  --candidates "$CANDIDATES" \
  --manifest "$MANIFEST" \
  --provenance "$PROVENANCE" \
  --output "$OUTPUT"

echo "=== E1 report ==="
cat "$OUTPUT"
```

- [ ] **Step 2: Verify the submit line flags against the CLI argparse**

Run: `PYTHONPATH=src python -m evidence_rag.evaluation.cluster_eval_cli --help`
Expected: shows exactly `--manifest --candidates --provenance --output --top-n --passage-chars`. The four positional slurm args map to `--candidates --manifest --provenance --output` — confirm the order in the script matches.

- [ ] **Step 3: Fill the E1 ledger BEFORE (pre-registration)**

In `docs/hpc-run-log.md`, replace the E1 entry's status + BEFORE block (currently "BLOCKED / harness 落地时补全") with:

```markdown
## E1 — 边/簇检测组件评估(spec §12,pool 级 B2 harness)

**状态:** READY——三件套齐(cluster_eval + cluster_eval_cli + tests、slurm、本条目;commit <FILL_AFTER_COMMIT>)。依赖 E2 已产出的 runs/e2-gate-on/candidate_sets.jsonl。

**BEFORE(预注册):**

- 目的/假设:量化 E2 池上 exact-string 答案簇的错误模式,把 E2 的 harm 收益/recall 代价与簇错摘开。
  假设:missed-conflict 低(注入按设计使 gold 与 counterfactual canonically 可分)、needle-gold-recovery 高;
  若 missed-conflict 偏高则门对部分注入题失明、E2 harm 收益被高估。
- 预期指标 + 方向:`selector.cluster.missed_conflict` ↓(主)、`selector.cluster.false_conflict` ↓、
  `selector.cluster.needle_gold_recovery` ↑,各带 Wilson 95% CI;框架句 `injection selection-bias`
  (multi-canonical-key skip rate)必并列——注入集已把别名假冲突过滤掉,低 conflict 值须据此解读。
- 精确命令(登录节点确认 E2 dump 在位后 sbatch):
  ```
  ls runs/e2-gate-on/candidate_sets.jsonl runs/niah-injected/manifest.json runs/niah-injected/provenance.jsonl
  mkdir -p logs results && sbatch scripts/run_cluster_eval.slurm \
    runs/e2-gate-on/candidate_sets.jsonl runs/niah-injected/manifest.json \
    runs/niah-injected/provenance.jsonl results/e1-cluster-eval/cluster_eval_report.json
  ```
- Git commit:<FILL_AFTER_COMMIT>;Seed:无(确定性抽取 + 计数,无随机化)。

**AFTER:** 未运行。
```

- [ ] **Step 4: Run the full suite once before shipping the triple**

Run: `PYTHONPATH=src pytest tests/evaluation -q`
Expected: PASS.

- [ ] **Step 5: Commit the triple**

```bash
git add scripts/run_cluster_eval.slurm docs/hpc-run-log.md
git commit -m "ops(hpc): E1 cluster-eval slurm + pre-registered run ledger"
```

Then edit the two `<FILL_AFTER_COMMIT>` placeholders in `docs/hpc-run-log.md` to the hash printed by `git log --oneline -1`, and amend:

```bash
git add docs/hpc-run-log.md && git commit --amend --no-edit
```

---

## Handoff (after all tasks pass, run on the login node)

```bash
# local  :  git push
# bp1    :  ssh bp1 → cd /user/work/$USER/IBM_Granite_Project && git pull && mkdir -p logs results
# submit :  sbatch scripts/run_cluster_eval.slurm runs/e2-gate-on/candidate_sets.jsonl runs/niah-injected/manifest.json runs/niah-injected/provenance.jsonl results/e1-cluster-eval/cluster_eval_report.json
# watch  :  squeue -u $USER | tail -f logs/cluster-eval-<jobid>.out | sacct -j <jobid>
# record :  git add -f results/e1-cluster-eval/cluster_eval_report.json && git commit -m "results(e1): cluster-eval headline" && git push   # on bp1
# local  :  git pull
```

Prerequisite: `runs/e2-gate-on/candidate_sets.jsonl` must still exist on bp1 from the E2 run (job 18130403). If it was cleaned, re-run the E2 gate-on arm first.

---

## Self-Review

**Spec coverage:**
- §3 metric 1 missed-conflict → Task 1 (`test_missed_conflict_*`). ✓
- §3 metric 2 false-conflict (needle-clustered + clustered gold-bearing other) → Task 1 (`test_false_conflict_*`). ✓
- §3 metric 3 needle-gold-recovery → Task 1. ✓
- §3 metric 4 dedup determinism → covered by the existing `build_clusters` document_id vote-merge tests (`tests/selector/test_gated.py`); E1 reuses `build_clusters` so no new dedup code. Noted here rather than duplicated. ✓
- §3 metric 5 selection-bias → Task 3. ✓
- §3 Wilson CI → Task 2. ✓
- §4 architecture (pure module + CLI, reuse build_clusters/AnswerExtractionEngine, no contract change) → Tasks 1-4. ✓
- §6 HPC triple → Task 5. ✓
- §1 stats unit = query, descriptive (no paired test) → aggregate over per-query cases, no significance call. ✓

**Placeholder scan:** Two intentional `<FILL_AFTER_COMMIT>` markers in the ledger are resolved in Task 5 Step 5 (they need the commit hash, which does not exist until the commit). No other placeholders.

**Type consistency:** `evaluate_case` returns `ClusterEvalCase`; `aggregate` consumes `Sequence[ClusterEvalCase]` and returns `ClusterEvalReport` with `MetricSummary` fields; CLI serializes via `dataclasses.asdict`. `selection_bias` returns `SelectionBias`. Names match across Tasks 1-4. `use_parametric=False` is correct because `build_clusters` never reads the parametric answer.
