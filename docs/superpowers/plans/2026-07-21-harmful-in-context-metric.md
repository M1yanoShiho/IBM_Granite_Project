# Harmful-in-Context Metric Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Implement the selector-owned harmful-in-context metric from `docs/superpowers/specs/2026-07-21-harmful-in-context-metric-design.md`: a per-query poison-in-context metric, a pool-hit diagnostic, a paired gate-on/off comparison with a self-contained randomization test + bootstrap CI, and an offline CLI that reads dumped artifacts.

**Architecture:** New `src/evidence_rag/evaluation/harm.py` reusing the existing scoring/report helpers; one metric key registered in `scoring.py`. All offline over dumped `selected_evidence_sets.jsonl` + `provenance.jsonl` — the shared ExperimentWorkflow and frozen contracts are untouched. Pure CPU, deterministic.

**Tech Stack:** Python 3.11, pydantic v2, pytest, mypy strict, ruff. No new deps (stdlib `random` for stats). Run tests with `$env:PYTHONPATH='src'; python -m pytest`.

---

## File Structure

| File | Responsibility |
|---|---|
| Modify `src/evidence_rag/evaluation/scoring.py` | register `selector.core.harmful_in_context` (lower), bump `CORE_METRIC_VERSION` 1.1→1.2 |
| Create `src/evidence_rag/evaluation/harm.py` | metric, per-arm report, pool-hit, `HarmComparison` + `compare_harm` |
| Create `src/evidence_rag/evaluation/harm_cli.py` | `evidence-rag-harm-report` offline analysis |
| Modify `pyproject.toml` | register CLI script |
| Tests | `tests/evaluation/test_harm.py`, `tests/evaluation/test_harm_cli.py` |

---

### Task 1: register metric + core harm.py (metric, report, pool-hit)

**Files:** Modify `src/evidence_rag/evaluation/scoring.py`; Create `src/evidence_rag/evaluation/harm.py`; Test `tests/evaluation/test_harm.py`

- [ ] **Step 1.1: Write failing tests**

```python
from evidence_rag.contracts.models import CandidateSet, EvidenceCandidate, SelectedEvidenceSet
from evidence_rag.evaluation.harm import (
    HARM_METRIC,
    counterfactual_pool_hit_rate,
    evaluate_selector_harm,
    harmful_in_context,
    provenance_harm_map,
)
from evidence_rag.materializer.provenance import MutationRecord


def ev(evidence_id: str, document_id: str) -> EvidenceCandidate:
    return EvidenceCandidate(
        evidence_id=evidence_id,
        document_id=document_id,
        chunk_id=f"c-{evidence_id}",
        text=f"text {evidence_id}",
        source_uri=f"fixture://{evidence_id}",
        retrieval_score=1.0,
        retrieval_rank=int(evidence_id[-1]) if evidence_id[-1].isdigit() else 1,
    )


def selected(query_id: str, *document_ids: str) -> SelectedEvidenceSet:
    return SelectedEvidenceSet(
        query_id=query_id,
        evidence=tuple(ev(f"e{i}", d) for i, d in enumerate(document_ids, start=1)),
    )


def record(query_id: str, cf_doc: str) -> MutationRecord:
    return MutationRecord(
        query_id=query_id,
        needle_document_id="needle",
        counterfactual_document_id=cf_doc,
        gold_value="18",
        gold_alias_used="18%",
        replacement_value="23",
        string_class="integer",
        seed=42,
        char_span=(0, 3),
        text_hash_before="a" * 64,
        text_hash_after="b" * 64,
        answer_bank_hash="c" * 64,
    )


def test_harmful_in_context_flags_poison_in_selection() -> None:
    assert harmful_in_context({"d1", "cf::q1::n"}, "cf::q1::n").value == 1.0
    assert harmful_in_context({"d1", "d2"}, "cf::q1::n").value == 0.0


def test_harmful_in_context_is_none_for_non_injected() -> None:
    assert harmful_in_context({"d1"}, None).value is None


def test_provenance_harm_map_builds_query_to_counterfactual() -> None:
    mapping = provenance_harm_map((record("q1", "cf::q1::n"), record("q2", "cf::q2::n")))
    assert mapping == {"q1": "cf::q1::n", "q2": "cf::q2::n"}


def test_evaluate_selector_harm_aggregates_over_injected_only() -> None:
    selected_sets = (
        selected("q1", "cf::q1::n", "d2"),  # poisoned
        selected("q2", "d3", "d4"),  # clean
        selected("q3", "d5"),  # not injected
    )
    harm_map = {"q1": "cf::q1::n", "q2": "cf::q2::n"}  # q3 absent
    report = evaluate_selector_harm(selected_sets, harm_map, dataset_signature="sig")
    aggregate = report.aggregate[HARM_METRIC]
    assert aggregate.mean == 0.5  # q1=1, q2=0; q3 unscored
    assert aggregate.n_scored == 2
    assert aggregate.n_total == 3


def test_pool_hit_rate_counts_injected_queries_only() -> None:
    candidate_sets = (
        CandidateSet(query_id="q1", candidates=(ev("e1", "cf::q1::n"), ev("e2", "d2"))),
        CandidateSet(query_id="q2", candidates=(ev("e3", "d3"),)),
    )
    harm_map = {"q1": "cf::q1::n", "q2": "cf::q2::n"}
    assert counterfactual_pool_hit_rate(candidate_sets, harm_map) == 0.5


def test_pool_hit_rate_is_none_without_injected_queries() -> None:
    candidate_sets = (CandidateSet(query_id="q1", candidates=(ev("e1", "d1"),)),)
    assert counterfactual_pool_hit_rate(candidate_sets, {}) is None
```

- [ ] **Step 1.2: Run to verify failure**

Run: `$env:PYTHONPATH='src'; python -m pytest tests/evaluation/test_harm.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'evidence_rag.evaluation.harm'`

- [ ] **Step 1.3: Register the metric in scoring.py**

In `src/evidence_rag/evaluation/scoring.py`, add the harm key to `CORE_DIRECTIONS` (after the `selector.core.document_precision` line) and bump the registry version:

```python
    "selector.core.document_precision": "higher",
    "selector.core.harmful_in_context": "lower",
```

Change `CORE_METRIC_VERSION = "1.1"` to `CORE_METRIC_VERSION = "1.2"`.

- [ ] **Step 1.4: Implement harm.py core**

```python
"""Harmful-in-context metric and pool-hit diagnostic (spec §1-§5).

Offline over dumped selected sets + the injector's provenance sidecar; the shared
ExperimentWorkflow and frozen contracts are untouched.
"""

from collections.abc import Iterable, Mapping

from evidence_rag.contracts.models import CandidateSet, SelectedEvidenceSet
from evidence_rag.evaluation.models import (
    MetricValue,
    StageCaseEvaluation,
    StageEvaluationReport,
)
from evidence_rag.evaluation.scoring import (
    CORE_DIRECTIONS,
    CORE_METRIC_VERSION,
    CORE_METRIC_VERSIONS,
    aggregate_metrics,
    document_ids,
    signature,
    unscored,
)
from evidence_rag.materializer.provenance import MutationRecord

HARM_METRIC = "selector.core.harmful_in_context"


def provenance_harm_map(records: Iterable[MutationRecord]) -> dict[str, str]:
    return {record.query_id: record.counterfactual_document_id for record in records}


def harmful_in_context(
    selected_document_ids: set[str],
    counterfactual_document_id: str | None,
) -> MetricValue:
    if counterfactual_document_id is None:
        return unscored("query has no injected counterfactual")
    return MetricValue(value=float(counterfactual_document_id in selected_document_ids))


def evaluate_selector_harm(
    selected_sets: Iterable[SelectedEvidenceSet],
    harm_map: Mapping[str, str],
    *,
    dataset_signature: str,
) -> StageEvaluationReport:
    selected = tuple(selected_sets)
    if not selected:
        raise ValueError("evaluation dataset must not be empty")
    directions = {HARM_METRIC: CORE_DIRECTIONS[HARM_METRIC]}
    per_case = tuple(
        StageCaseEvaluation(
            query_id=item.query_id,
            metrics={
                HARM_METRIC: harmful_in_context(
                    document_ids(item.evidence), harm_map.get(item.query_id)
                )
            },
        )
        for item in selected
    )
    return StageEvaluationReport(
        stage="selector",
        dataset_signature=dataset_signature,
        metric_registry_signature=signature(
            {
                "core_version": CORE_METRIC_VERSION,
                "metrics": (
                    (HARM_METRIC, directions[HARM_METRIC], CORE_METRIC_VERSIONS[HARM_METRIC]),
                ),
            }
        ),
        case_ids=tuple(item.query_id for item in selected),
        per_case=per_case,
        aggregate=aggregate_metrics(tuple(case.metrics for case in per_case), directions),
        directions=directions,
    )


def counterfactual_pool_hit_rate(
    candidate_sets: Iterable[CandidateSet],
    harm_map: Mapping[str, str],
) -> float | None:
    hits = 0
    total = 0
    for candidate_set in candidate_sets:
        counterfactual = harm_map.get(candidate_set.query_id)
        if counterfactual is None:
            continue
        total += 1
        if counterfactual in document_ids(candidate_set.candidates):
            hits += 1
    return None if total == 0 else hits / total
```

- [ ] **Step 1.5: Run to verify pass**

Run: `$env:PYTHONPATH='src'; python -m pytest tests/evaluation/test_harm.py -q`
Expected: PASS (6 tests)

- [ ] **Step 1.6: Commit**

```bash
git add src/evidence_rag/evaluation/scoring.py src/evidence_rag/evaluation/harm.py tests/evaluation/test_harm.py
git commit -m "feat(eval): harmful-in-context metric and pool-hit diagnostic"
```

### Task 2: paired comparison (HarmComparison + compare_harm)

**Files:** Modify `src/evidence_rag/evaluation/harm.py`; extend `tests/evaluation/test_harm.py`

- [ ] **Step 2.1: Write failing tests**

```python
from evidence_rag.evaluation.harm import HarmComparison, compare_harm


def _harm_report(pairs: dict[str, float | None]):
    from evidence_rag.evaluation.harm import HARM_METRIC
    from evidence_rag.evaluation.models import MetricValue, StageCaseEvaluation, StageEvaluationReport
    from evidence_rag.evaluation.scoring import CORE_DIRECTIONS, aggregate_metrics

    directions = {HARM_METRIC: CORE_DIRECTIONS[HARM_METRIC]}
    per_case = tuple(
        StageCaseEvaluation(query_id=q, metrics={HARM_METRIC: MetricValue(value=v)})
        for q, v in pairs.items()
    )
    return StageEvaluationReport(
        stage="selector",
        dataset_signature="sig",
        metric_registry_signature="x" * 64,
        case_ids=tuple(pairs),
        per_case=per_case,
        aggregate=aggregate_metrics(tuple(c.metrics for c in per_case), directions),
        directions=directions,
    )


def test_compare_harm_reports_delta_and_deterministic_ci() -> None:
    ids = [f"q{i}" for i in range(5)]
    on = _harm_report({q: 0.0 for q in ids})  # gate-on: no poison
    off = _harm_report({q: 1.0 for q in ids})  # gate-off: all poisoned
    result = compare_harm(on, off, seed=13, iterations=1000)
    assert isinstance(result, HarmComparison)
    assert result.harm_on == 0.0 and result.harm_off == 1.0
    assert result.delta == -1.0
    assert result.ci_low == -1.0 and result.ci_high == -1.0  # all diffs equal
    assert result.p_value < 0.2
    assert result.n_paired == 5


def test_compare_harm_no_difference_gives_zero_delta_and_high_p() -> None:
    ids = [f"q{i}" for i in range(4)]
    on = _harm_report({q: 1.0 for q in ids})
    off = _harm_report({q: 1.0 for q in ids})
    result = compare_harm(on, off, seed=13, iterations=500)
    assert result.delta == 0.0
    assert result.ci_low == 0.0 and result.ci_high == 0.0
    assert result.p_value == 1.0


def test_compare_harm_ignores_unpaired_and_unscored() -> None:
    on = _harm_report({"q1": 0.0, "q2": 0.0, "q3": None})
    off = _harm_report({"q1": 1.0, "q2": 1.0, "q4": 1.0})
    result = compare_harm(on, off, seed=13, iterations=200)
    assert result.n_paired == 2  # q3 unscored, q4 unpaired
    assert result.delta == -1.0
```

- [ ] **Step 2.2: Run to verify failure**

Run: `$env:PYTHONPATH='src'; python -m pytest tests/evaluation/test_harm.py -q`
Expected: FAIL — `ImportError: cannot import name 'compare_harm'`

- [ ] **Step 2.3: Implement (append to harm.py)**

Add imports at the top of harm.py: `import random` and `from pydantic import BaseModel, ConfigDict`. Then append:

```python
class HarmComparison(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    harm_on: float
    harm_off: float
    delta: float
    p_value: float
    ci_low: float
    ci_high: float
    n_paired: int


def compare_harm(
    on_report: StageEvaluationReport,
    off_report: StageEvaluationReport,
    *,
    seed: int = 13,
    iterations: int = 10000,
) -> HarmComparison:
    on = {case.query_id: case.metrics[HARM_METRIC].value for case in on_report.per_case}
    off = {case.query_id: case.metrics[HARM_METRIC].value for case in off_report.per_case}
    diffs: list[float] = []
    on_values: list[float] = []
    off_values: list[float] = []
    for query_id, on_value in on.items():
        off_value = off.get(query_id)
        if on_value is None or off_value is None:
            continue
        on_values.append(on_value)
        off_values.append(off_value)
        diffs.append(on_value - off_value)
    if not diffs:
        raise ValueError("no paired scored queries")
    n = len(diffs)
    harm_on = sum(on_values) / n
    harm_off = sum(off_values) / n
    delta = harm_on - harm_off
    observed = abs(delta)
    rng = random.Random(seed)
    extreme = 0
    for _ in range(iterations):
        permuted = sum(d if rng.random() < 0.5 else -d for d in diffs) / n
        if abs(permuted) >= observed - 1e-12:
            extreme += 1
    p_value = extreme / iterations
    boot: list[float] = []
    for _ in range(iterations):
        boot.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    boot.sort()
    ci_low = boot[int(0.025 * iterations)]
    ci_high = boot[int(0.975 * iterations)]
    return HarmComparison(
        harm_on=harm_on,
        harm_off=harm_off,
        delta=delta,
        p_value=p_value,
        ci_low=ci_low,
        ci_high=ci_high,
        n_paired=n,
    )
```

- [ ] **Step 2.4: Run to verify pass**

Run: `$env:PYTHONPATH='src'; python -m pytest tests/evaluation/test_harm.py -q`
Expected: PASS (9 tests)

- [ ] **Step 2.5: Commit**

```bash
git add src/evidence_rag/evaluation/harm.py tests/evaluation/test_harm.py
git commit -m "feat(eval): paired harm comparison with randomization test and bootstrap CI"
```

### Task 3: offline CLI + entry point

**Files:** Create `src/evidence_rag/evaluation/harm_cli.py`; Modify `pyproject.toml`; Test `tests/evaluation/test_harm_cli.py`

- [ ] **Step 3.1: Write failing test**

```python
import json
from pathlib import Path

from evidence_rag.contracts.models import EvidenceCandidate, SelectedEvidenceSet
from evidence_rag.evaluation.harm_cli import main
from evidence_rag.materializer.provenance import MutationRecord, write_provenance


def ev(document_id: str) -> EvidenceCandidate:
    return EvidenceCandidate(
        evidence_id=f"e-{document_id}",
        document_id=document_id,
        chunk_id=f"c-{document_id}",
        text=f"text {document_id}",
        source_uri=f"fixture://{document_id}",
        retrieval_score=1.0,
        retrieval_rank=1,
    )


def _write_selected(path: Path, sets: tuple[SelectedEvidenceSet, ...]) -> None:
    path.write_text(
        "\n".join(item.model_dump_json() for item in sets) + "\n", encoding="utf-8"
    )


def record(query_id: str, cf_doc: str) -> MutationRecord:
    return MutationRecord(
        query_id=query_id, needle_document_id="n", counterfactual_document_id=cf_doc,
        gold_value="18", gold_alias_used="18%", replacement_value="23", string_class="integer",
        seed=42, char_span=(0, 3), text_hash_before="a" * 64, text_hash_after="b" * 64,
        answer_bank_hash="c" * 64,
    )


def test_cli_emits_harm_comparison(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    write_provenance(tmp_path / "provenance.jsonl", (record("q1", "cf1"), record("q2", "cf2")))
    on = (
        SelectedEvidenceSet(query_id="q1", evidence=(ev("d1"),)),
        SelectedEvidenceSet(query_id="q2", evidence=(ev("d2"),)),
    )
    off = (
        SelectedEvidenceSet(query_id="q1", evidence=(ev("cf1"),)),
        SelectedEvidenceSet(query_id="q2", evidence=(ev("cf2"),)),
    )
    _write_selected(tmp_path / "on.jsonl", on)
    _write_selected(tmp_path / "off.jsonl", off)
    exit_code = main(
        (
            "--provenance", str(tmp_path / "provenance.jsonl"),
            "--selected-on", str(tmp_path / "on.jsonl"),
            "--selected-off", str(tmp_path / "off.jsonl"),
            "--dataset-signature", "sig",
            "--iterations", "500",
        )
    )
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["harm_on"] == 0.0
    assert payload["harm_off"] == 1.0
    assert payload["delta"] == -1.0
    assert payload["n_paired"] == 2
```

- [ ] **Step 3.2: Run to verify failure**

Run: `$env:PYTHONPATH='src'; python -m pytest tests/evaluation/test_harm_cli.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'evidence_rag.evaluation.harm_cli'`

- [ ] **Step 3.3: Implement harm_cli.py**

```python
"""CLI: offline harmful-in-context comparison of two selector runs (spec §7)."""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from evidence_rag.contracts.models import CandidateSet, SelectedEvidenceSet
from evidence_rag.evaluation.harm import (
    compare_harm,
    counterfactual_pool_hit_rate,
    evaluate_selector_harm,
    provenance_harm_map,
)
from evidence_rag.materializer.provenance import read_provenance


def _read_selected(path: Path) -> tuple[SelectedEvidenceSet, ...]:
    text = Path(path).read_text(encoding="utf-8")
    return tuple(
        SelectedEvidenceSet.model_validate_json(line)
        for line in text.splitlines()
        if line.strip()
    )


def _read_candidates(path: Path) -> tuple[CandidateSet, ...]:
    text = Path(path).read_text(encoding="utf-8")
    return tuple(
        CandidateSet.model_validate_json(line) for line in text.splitlines() if line.strip()
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline harmful-in-context gate-on/off report")
    parser.add_argument("--provenance", required=True, type=Path)
    parser.add_argument("--selected-on", required=True, type=Path)
    parser.add_argument("--selected-off", required=True, type=Path)
    parser.add_argument("--candidates", type=Path)
    parser.add_argument("--dataset-signature", required=True)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--iterations", type=int, default=10000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    harm_map = provenance_harm_map(read_provenance(arguments.provenance))
    on_report = evaluate_selector_harm(
        _read_selected(arguments.selected_on), harm_map, dataset_signature=arguments.dataset_signature
    )
    off_report = evaluate_selector_harm(
        _read_selected(arguments.selected_off), harm_map, dataset_signature=arguments.dataset_signature
    )
    comparison = compare_harm(
        on_report, off_report, seed=arguments.seed, iterations=arguments.iterations
    )
    pool_hit = (
        counterfactual_pool_hit_rate(_read_candidates(arguments.candidates), harm_map)
        if arguments.candidates
        else None
    )
    print(
        json.dumps(
            {
                "harm_on": comparison.harm_on,
                "harm_off": comparison.harm_off,
                "delta": comparison.delta,
                "p_value": comparison.p_value,
                "ci_low": comparison.ci_low,
                "ci_high": comparison.ci_high,
                "n_paired": comparison.n_paired,
                "pool_hit_rate": pool_hit,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3.4: Register the entry point in pyproject.toml**

In `[project.scripts]` add:

```toml
evidence-rag-harm-report = "evidence_rag.evaluation.harm_cli:main"
```

- [ ] **Step 3.5: Run to verify pass, full suite, typecheck, ruff**

Run:
```
$env:PYTHONPATH='src'; python -m pytest tests/evaluation/test_harm_cli.py -q
$env:PYTHONPATH='src'; python -m pytest -q
$env:MYPYPATH='src'; python -m mypy
python -m ruff check src/evidence_rag/evaluation/harm.py src/evidence_rag/evaluation/harm_cli.py src/evidence_rag/evaluation/scoring.py tests/evaluation/test_harm.py tests/evaluation/test_harm_cli.py
```
Expected: CLI test passes; full suite parity (no new failures vs baseline); mypy clean; ruff clean.

- [ ] **Step 3.6: Commit**

```bash
git add src/evidence_rag/evaluation/harm_cli.py pyproject.toml tests/evaluation/test_harm_cli.py
git commit -m "feat(eval): evidence-rag-harm-report offline gate-on/off CLI"
```

---

## Experiment marking (running-hpc-experiments)

The metric and CLI are CPU/offline. E2 itself (spec §7) = run the existing `scripts/run_selector_gate.slurm` with two selector TOMLs on the injected dataset (needs Materializer A's base dataset first), then run `evidence-rag-harm-report` on the two dumped `selected_evidence_sets.jsonl`. Record under E2 in `docs/hpc-run-log.md` when the base dataset lands. No new slurm here.

## Self-review notes

- Spec coverage: §1 metric→Task 1; §2 registration→Task 1.3; §3 provenance_harm_map→Task 1; §4 guardrail = existing `conditional_document_recall` (no task needed, already computed); §5 pool-hit→Task 1; §6 compare_harm→Task 2; §7 CLI→Task 3; §8 refinement = experiment procedure (doc, not code); §11 non-goals honored (no workflow/contract change).
- No placeholders: all code complete; stats are concrete stdlib `random`.
- Type consistency: `HARM_METRIC`, `evaluate_selector_harm`, `compare_harm`, `HarmComparison`, `provenance_harm_map`, `counterfactual_pool_hit_rate` names identical across Tasks 1-3 and the CLI.
- Determinism: `compare_harm` seeded; deterministic-diff test cases make delta/CI exact regardless of iterations.
