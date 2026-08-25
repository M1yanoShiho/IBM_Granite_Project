"""F000/F001 runner for the frozen Selector--Generator four-arm experiment.

The runtime command deliberately has no gold-answer argument.  It reconstructs
the already-frozen NIAH decision-dev cases from the question file, TopK pool,
component roles and the committed Seed-13 Selector trace, then runs:

    A: TopK10  + GraniteGenerator
    B: Selector + GraniteGenerator
    C: TopK10  + VerifyAnnotateGenerator
    D: Selector + VerifyAnnotateGenerator

All four arms share one Granite client in one process.  Gold answers are read
only by ``prepare`` (input audit) and ``score`` (post-generation evaluation).
No Retriever or Selector model is rerun.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
import traceback
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from evidence_rag.contracts.models import (
    CandidateSet,
    EvidenceCandidate,
    GenerationResult,
    Query,
    QueryChecklist,
    SelectedEvidenceSet,
    strip_annotations,
)
from evidence_rag.evaluation.paired_metric import compare_paired
from evidence_rag.evaluation.scoring import answer_match
from evidence_rag.generator.granite import (
    GraniteGenerationConfig,
    GraniteGenerator,
    GraniteLLMClient,
)
from evidence_rag.generator.nli import TrueNLIModel, build_nli_model
from evidence_rag.generator.verify_annotate import VerifyAnnotateGenerator
from evidence_rag.infrastructure.datasets import GoldCase

ARMS = ("A_topk_basic", "B_selector_basic", "C_topk_verify", "D_selector_verify")
EXPECTED_CASES = 739
EXPECTED_CHANGED = 109
DEFAULT_MAX_ERROR_RATE = 0.05


class AnswerGenerator(Protocol):
    def generate(
        self,
        query: Query,
        checklist: QueryChecklist,
        selected: SelectedEvidenceSet,
    ) -> GenerationResult: ...


@dataclass(frozen=True, slots=True)
class JointCase:
    query_id: str
    question: str
    component_id: str
    topk10: tuple[EvidenceCandidate, ...]
    selected: tuple[EvidenceCandidate, ...]

    @property
    def selector_changed(self) -> bool:
        return tuple(item.evidence_id for item in self.topk10) != tuple(
            item.evidence_id for item in self.selected
        )

    def identity_row(self) -> dict[str, object]:
        return {
            "query_id": self.query_id,
            "question": self.question,
            "component_id": self.component_id,
            "topk10": [item.model_dump(mode="json") for item in self.topk10],
            "selected_evidence_ids": [item.evidence_id for item in self.selected],
        }


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _jsonl(path: Path) -> Iterable[tuple[int, Mapping[str, Any]]]:
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, Mapping):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        yield line_number, value


def _load_queries(path: Path) -> dict[str, Query]:
    output: dict[str, Query] = {}
    for line_number, value in _jsonl(path):
        query = Query.model_validate(value)
        if query.query_id in output:
            raise ValueError(f"duplicate query at {path}:{line_number}: {query.query_id}")
        output[query.query_id] = query
    return output


def _load_candidate_sets(path: Path) -> dict[str, CandidateSet]:
    output: dict[str, CandidateSet] = {}
    for line_number, value in _jsonl(path):
        candidate_set = CandidateSet.model_validate(value)
        if candidate_set.query_id in output:
            raise ValueError(
                f"duplicate candidate set at {path}:{line_number}: {candidate_set.query_id}"
            )
        output[candidate_set.query_id] = candidate_set
    return output


def _load_decision_roles(path: Path) -> dict[str, str]:
    output: dict[str, str] = {}
    for line_number, value in _jsonl(path):
        if value.get("role") != "decision-dev":
            continue
        query_id = str(value.get("query_id", "")).strip()
        component_id = str(value.get("component_id", "")).strip()
        if not query_id or not component_id:
            raise ValueError(f"invalid decision-dev role at {path}:{line_number}")
        if query_id in output:
            raise ValueError(f"duplicate decision-dev query at {path}:{line_number}: {query_id}")
        output[query_id] = component_id
    return output


def _load_selector_decisions(path: Path) -> dict[str, tuple[str, ...]]:
    output: dict[str, tuple[str, ...]] = {}
    for line_number, value in _jsonl(path):
        if value.get("dataset_kind") != "niah":
            continue
        query_id = str(value.get("query_id", "")).strip()
        raw_ids = value.get("selected_evidence_ids")
        if not query_id or not isinstance(raw_ids, list):
            raise ValueError(f"invalid NIAH Selector decision at {path}:{line_number}")
        selected_ids = tuple(str(item).strip() for item in raw_ids)
        if not selected_ids or any(not item for item in selected_ids):
            raise ValueError(f"empty selected evidence ID at {path}:{line_number}")
        if len(selected_ids) != len(set(selected_ids)):
            raise ValueError(f"duplicate selected evidence ID at {path}:{line_number}")
        if query_id in output:
            raise ValueError(f"duplicate Selector decision at {path}:{line_number}: {query_id}")
        output[query_id] = selected_ids
    return output


def load_runtime_cases(
    *,
    queries_path: Path,
    candidate_pool_path: Path,
    roles_path: Path,
    decision_trace_path: Path,
    require_expected_counts: bool = True,
) -> tuple[JointCase, ...]:
    """Reconstruct runtime cases without reading gold labels or reference answers."""

    queries = _load_queries(queries_path)
    pools = _load_candidate_sets(candidate_pool_path)
    roles = _load_decision_roles(roles_path)
    decisions = _load_selector_decisions(decision_trace_path)
    missing = {
        "queries": sorted(set(roles) - set(queries))[:5],
        "candidate_pool": sorted(set(roles) - set(pools))[:5],
        "selector_decisions": sorted(set(roles) - set(decisions))[:5],
    }
    missing = {name: values for name, values in missing.items() if values}
    if missing:
        raise ValueError(f"runtime inputs omit decision-dev queries: {missing}")

    cases: list[JointCase] = []
    for query_id in sorted(roles):
        ordered = tuple(
            sorted(
                pools[query_id].candidates,
                key=lambda item: (item.retrieval_rank, item.evidence_id),
            )[:10]
        )
        if tuple(item.retrieval_rank for item in ordered) != tuple(range(1, 11)):
            raise ValueError(f"query {query_id} does not have the exact TopK10 prefix")
        by_id = {item.evidence_id: item for item in ordered}
        selected_ids = decisions[query_id]
        unknown = sorted(set(selected_ids) - set(by_id))
        if unknown:
            raise ValueError(f"query {query_id} selects evidence outside TopK10: {unknown}")
        selected = tuple(item for item in ordered if item.evidence_id in set(selected_ids))
        if tuple(item.evidence_id for item in selected) != selected_ids:
            raise ValueError(f"query {query_id} selected order differs from TopK10 order")
        cases.append(
            JointCase(
                query_id=query_id,
                question=queries[query_id].text,
                component_id=roles[query_id],
                topk10=ordered,
                selected=selected,
            )
        )

    changed = sum(case.selector_changed for case in cases)
    if require_expected_counts and (len(cases), changed) != (EXPECTED_CASES, EXPECTED_CHANGED):
        raise ValueError(
            "frozen NIAH counts differ: "
            f"observed cases/changed={len(cases)}/{changed}, "
            f"expected={EXPECTED_CASES}/{EXPECTED_CHANGED}"
        )
    return tuple(cases)


def _load_gold(path: Path, wanted: set[str]) -> dict[str, GoldCase]:
    output: dict[str, GoldCase] = {}
    for line_number, value in _jsonl(path):
        gold = GoldCase.model_validate(value)
        if gold.query_id not in wanted:
            continue
        if gold.query_id in output:
            raise ValueError(f"duplicate gold row at {path}:{line_number}: {gold.query_id}")
        if not gold.reference_answers:
            raise ValueError(f"query {gold.query_id} has no reference answer")
        output[gold.query_id] = gold
    missing = sorted(wanted - set(output))
    if missing:
        raise ValueError(f"gold file omits decision-dev queries: {missing[:5]}")
    return output


def build_input_manifest(
    *,
    cases: Sequence[JointCase],
    source_paths: Mapping[str, Path],
    gold_path: Path,
) -> dict[str, object]:
    _load_gold(gold_path, {case.query_id for case in cases})
    changed = [case.query_id for case in cases if case.selector_changed]
    return {
        "schema_version": "full-flow-f000-input-manifest-v1",
        "dataset": "NIAH decision-dev",
        "runtime_gold_boundary": (
            "run reads questions, candidate pool, component roles and Selector decisions only; "
            "gold is read after generation by score"
        ),
        "counts": {
            "queries": len(cases),
            "selector_changed_queries": len(changed),
            "unchanged_queries": len(cases) - len(changed),
        },
        "runtime_input_sha256": _sha256_bytes(
            _canonical_bytes([case.identity_row() for case in cases])
        ),
        "changed_query_ids_sha256": _sha256_bytes(_canonical_bytes(changed)),
        "source_files": {
            name: {"path": str(path.resolve()), "sha256": _sha256_file(path)}
            for name, path in sorted(source_paths.items())
        },
        "evaluation_file": {
            "path": str(gold_path.resolve()),
            "sha256": _sha256_file(gold_path),
            "runtime_access": False,
        },
        "arms": {
            "A_topk_basic": "TopK10 + GraniteGenerator",
            "B_selector_basic": "Lean v3 Seed-13 selected evidence + GraniteGenerator",
            "C_topk_verify": "TopK10 + VerifyAnnotateGenerator(entity_gate=observe)",
            "D_selector_verify": (
                "Lean v3 Seed-13 selected evidence + "
                "VerifyAnnotateGenerator(entity_gate=observe)"
            ),
        },
        "generation": {
            "model": "ibm-granite/granite-4.1-3b@c0650403e44e78ec0262dab1c90914c65b196c4e",
            "max_new_tokens": 256,
            "temperature": 0.0,
            "top_p": 1.0,
            "shared_model_instance": True,
            "same_process": True,
        },
        "verification": {
            "backend": "TRUE",
            "entity_gate": "observe",
            "abstain_when_unverified": False,
        },
    }


def _routing_rows(generator: AnswerGenerator) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for routing in getattr(generator, "last_routings", ()):
        rows.append(
            {
                "claim_id": routing.claim_id,
                "outcome": routing.outcome,
                "sentence": routing.sentence,
                "citation": routing.citation,
                "claim_text": routing.claim_text,
                "declared_indices": list(routing.declared_indices),
                "declared_verified": routing.declared_verified,
                "rescued_by_scan": routing.rescued_by_scan,
                "gated_outcome": routing.gated_outcome,
                "gated_citation": routing.gated_citation,
                "review_flagged": routing.review_flagged,
            }
        )
    return rows


def _run_one(
    generator: AnswerGenerator,
    query: Query,
    checklist: QueryChecklist,
    selected: SelectedEvidenceSet,
) -> dict[str, object]:
    started = time.perf_counter()
    try:
        generation = generator.generate(query, checklist, selected)
    except Exception as error:  # noqa: BLE001 -- runtime failure is a measured system outcome
        traceback.print_exc()
        return {
            "generation": GenerationResult(
                query_id=query.query_id, answer="", cited_evidence_ids=()
            ).model_dump(mode="json"),
            "routing": [],
            "error": f"{type(error).__name__}: {error}",
            "seconds": time.perf_counter() - started,
        }
    return {
        "generation": generation.model_dump(mode="json"),
        "routing": _routing_rows(generator),
        "error": None,
        "seconds": time.perf_counter() - started,
    }


def run_cases(
    cases: Sequence[JointCase],
    *,
    basic_generator: AnswerGenerator,
    verify_generator: AnswerGenerator,
) -> list[dict[str, object]]:
    """Run four arms; unchanged Selector contexts reuse their paired TopK result."""

    output: list[dict[str, object]] = []
    started = time.perf_counter()
    for index, case in enumerate(cases, start=1):
        query = Query(query_id=case.query_id, text=case.question)
        checklist = QueryChecklist(
            query_id=case.query_id,
            focus=case.question,
            required_facts=(),
        )
        topk = SelectedEvidenceSet(query_id=case.query_id, evidence=case.topk10)
        selected = SelectedEvidenceSet(query_id=case.query_id, evidence=case.selected)
        arms: dict[str, dict[str, object]] = {}
        arms[ARMS[0]] = _run_one(basic_generator, query, checklist, topk)
        if case.selector_changed:
            arms[ARMS[1]] = _run_one(basic_generator, query, checklist, selected)
            arms[ARMS[1]]["reused_from"] = None
        else:
            arms[ARMS[1]] = {**arms[ARMS[0]], "reused_from": ARMS[0]}
        arms[ARMS[0]]["reused_from"] = None

        arms[ARMS[2]] = _run_one(verify_generator, query, checklist, topk)
        if case.selector_changed:
            arms[ARMS[3]] = _run_one(verify_generator, query, checklist, selected)
            arms[ARMS[3]]["reused_from"] = None
        else:
            arms[ARMS[3]] = {**arms[ARMS[2]], "reused_from": ARMS[2]}
        arms[ARMS[2]]["reused_from"] = None
        output.append(
            {
                "schema_version": "full-flow-f001-generation-row-v1",
                "query_id": case.query_id,
                "question": case.question,
                "component_id": case.component_id,
                "selector_changed": case.selector_changed,
                "topk10": [item.model_dump(mode="json") for item in case.topk10],
                "selected_evidence_ids": [item.evidence_id for item in case.selected],
                "arms": arms,
            }
        )
        if index % 10 == 0 or index == len(cases):
            elapsed = time.perf_counter() - started
            print(
                f"[run] {index}/{len(cases)} cases; {elapsed / index:.2f}s/case",
                flush=True,
            )
    return output


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_generation_rows(path: Path) -> list[Mapping[str, Any]]:
    return [value for _, value in _jsonl(path)]


def _metric_value(value: float | None) -> dict[str, float] | None:
    return None if value is None else {"value": value}


def _mean(values: Sequence[float | None]) -> float | None:
    scored = [value for value in values if value is not None]
    return None if not scored else sum(scored) / len(scored)


def _finite_or_none(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("metric must be numeric or null")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError("metric must be finite")
    return numeric


def score_rows(
    rows: Sequence[Mapping[str, Any]],
    gold_by_id: Mapping[str, GoldCase],
) -> dict[str, object]:
    per_arm: dict[str, dict[str, dict[str, float | None]]] = {arm: {} for arm in ARMS}
    component_ids: dict[str, str] = {}
    changed: dict[str, bool] = {}
    errors = dict.fromkeys(ARMS, 0)
    for row in rows:
        query_id = str(row["query_id"])
        gold = gold_by_id[query_id]
        references = gold.reference_answers
        relevant = set(gold.relevant_document_ids or ())
        topk = tuple(EvidenceCandidate.model_validate(item) for item in row["topk10"])
        candidate_by_id = {item.evidence_id: item for item in topk}
        component_ids[query_id] = str(row["component_id"])
        changed[query_id] = bool(row["selector_changed"])
        arm_rows = row["arms"]
        if not isinstance(arm_rows, Mapping):
            raise ValueError(f"query {query_id} has invalid arms")
        for arm in ARMS:
            arm_row = arm_rows[arm]
            if not isinstance(arm_row, Mapping):
                raise ValueError(f"query {query_id}, arm {arm} is invalid")
            generation = GenerationResult.model_validate(arm_row["generation"])
            if arm_row.get("error"):
                errors[arm] += 1
            answer = strip_annotations(generation.answer)
            matched = answer_match(answer, references).value
            cited_documents = {
                candidate_by_id[evidence_id].document_id
                for evidence_id in generation.cited_evidence_ids
                if evidence_id in candidate_by_id
            }
            citation_precision = (
                None
                if not cited_documents or not relevant
                else len(cited_documents & relevant) / len(cited_documents)
            )
            citation_recall = (
                None if not relevant else len(cited_documents & relevant) / len(relevant)
            )
            valid = (
                0.0
                if not generation.cited_evidence_ids
                else len(set(generation.cited_evidence_ids) & set(candidate_by_id))
                / len(generation.cited_evidence_ids)
            )
            per_arm[arm][query_id] = {
                "answer_match": _finite_or_none(matched),
                "coverage": float(bool(generation.answer.strip())),
                "gold_document_citation_precision": citation_precision,
                "gold_document_citation_recall": citation_recall,
                "citation_validity": valid,
            }

    def subset_report(wanted: set[str]) -> dict[str, object]:
        aggregates: dict[str, object] = {}
        for arm in ARMS:
            metrics = per_arm[arm]
            aggregates[arm] = {
                key: _mean([metrics[query_id][key] for query_id in sorted(wanted)])
                for key in next(iter(metrics.values()))
            }

        def values(arm: str, metric: str) -> dict[str, float | None]:
            return {query_id: per_arm[arm][query_id][metric] for query_id in wanted}

        comps = {query_id: component_ids[query_id] for query_id in wanted}
        answer_pairs = {
            "B_minus_A": asdict(
                compare_paired(
                    values(ARMS[1], "answer_match"),
                    values(ARMS[0], "answer_match"),
                    component_ids=comps,
                )
            ),
            "C_minus_A": asdict(
                compare_paired(
                    values(ARMS[2], "answer_match"),
                    values(ARMS[0], "answer_match"),
                    component_ids=comps,
                )
            ),
            "D_minus_C": asdict(
                compare_paired(
                    values(ARMS[3], "answer_match"),
                    values(ARMS[2], "answer_match"),
                    component_ids=comps,
                )
            ),
        }
        selector_basic: dict[str, float] = {}
        selector_verify: dict[str, float] = {}
        for query_id in wanted:
            basic_on = per_arm[ARMS[1]][query_id]["answer_match"]
            basic_off = per_arm[ARMS[0]][query_id]["answer_match"]
            verify_on = per_arm[ARMS[3]][query_id]["answer_match"]
            verify_off = per_arm[ARMS[2]][query_id]["answer_match"]
            if None in (basic_on, basic_off, verify_on, verify_off):
                raise ValueError(f"query {query_id} has unscored answer_match")
            assert basic_on is not None and basic_off is not None
            assert verify_on is not None and verify_off is not None
            selector_basic[query_id] = basic_on - basic_off
            selector_verify[query_id] = verify_on - verify_off
        answer_pairs["interaction"] = asdict(
            compare_paired(selector_verify, selector_basic, component_ids=comps)
        )

        transitions: dict[str, object] = {}
        for label, on_arm, off_arm in (
            ("B_vs_A", ARMS[1], ARMS[0]),
            ("D_vs_C", ARMS[3], ARMS[2]),
        ):
            wrong_to_right = right_to_wrong = 0
            for query_id in wanted:
                before = per_arm[off_arm][query_id]["answer_match"]
                after = per_arm[on_arm][query_id]["answer_match"]
                wrong_to_right += int(before == 0.0 and after == 1.0)
                right_to_wrong += int(before == 1.0 and after == 0.0)
            transitions[label] = {
                "wrong_to_right": wrong_to_right,
                "right_to_wrong": right_to_wrong,
                "net": wrong_to_right - right_to_wrong,
            }
        return {
            "queries": len(wanted),
            "aggregate": aggregates,
            "answer_comparisons": answer_pairs,
            "answer_transitions": transitions,
        }

    all_ids = set(component_ids)
    changed_ids = {query_id for query_id, is_changed in changed.items() if is_changed}
    return {
        "schema_version": "full-flow-f001-report-v1",
        "status": "COMPLETE",
        "metric_notes": {
            "answer_match": "same normalized containment metric used by Selector L003",
            "gold_document_citation_precision": (
                "post-generation overlap with official relevant documents; not sentence-level NLI"
            ),
            "gold_document_citation_recall": (
                "post-generation coverage of official relevant documents; not sentence-level NLI"
            ),
        },
        "errors_by_arm": errors,
        "all_decision_dev": subset_report(all_ids),
        "selector_changed": subset_report(changed_ids),
    }


def _markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# F001 现有三模块四臂联合实验",
        "",
        f"**状态：** `{report['status']}`",
        "",
        "## 总体结果",
        "",
        "| 范围 | 组别 | Answer match | Coverage | Gold-doc citation precision | Gold-doc citation recall |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for scope_key, scope_label in (
        ("all_decision_dev", "全部 decision-dev"),
        ("selector_changed", "Selector 改变上下文"),
    ):
        scope = report[scope_key]
        for arm in ARMS:
            metrics = scope["aggregate"][arm]

            answer = metrics["answer_match"]
            coverage = metrics["coverage"]
            precision = metrics["gold_document_citation_precision"]
            recall = metrics["gold_document_citation_recall"]

            def pct(value: float | None) -> str:
                return "—" if value is None else f"{100 * value:.2f}%"

            lines.append(
                f"| {scope_label} ({scope['queries']}) | {arm} | {pct(answer)} | "
                f"{pct(coverage)} | {pct(precision)} | {pct(recall)} |"
            )
    lines.extend(
        [
            "",
            "## Selector 对答案的影响",
            "",
            "| 范围 | 比较 | Delta | 95% CI | wrong→right | right→wrong |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for scope_key, scope_label in (
        ("all_decision_dev", "全部"),
        ("selector_changed", "改动题"),
    ):
        scope = report[scope_key]
        for comparison, transition in (("B_minus_A", "B_vs_A"), ("D_minus_C", "D_vs_C")):
            result = scope["answer_comparisons"][comparison]
            changes = scope["answer_transitions"][transition]
            lines.append(
                f"| {scope_label} | {comparison} | {100 * result['delta']:+.3f} pp | "
                f"[{100 * result['ci_low']:+.3f}, {100 * result['ci_high']:+.3f}] pp | "
                f"{changes['wrong_to_right']} | {changes['right_to_wrong']} |"
            )
        interaction = scope["answer_comparisons"]["interaction"]
        lines.append(
            f"| {scope_label} | interaction | {100 * interaction['delta']:+.3f} pp | "
            f"[{100 * interaction['ci_low']:+.3f}, "
            f"{100 * interaction['ci_high']:+.3f}] pp | — | — |"
        )
    lines.extend(
        [
            "",
            "Gold-doc citation precision/recall 是生成后使用官方相关文档计算的诊断指标，",
            "不是逐句 NLI 引用指标，也没有进入 Generator 运行时。",
            "",
        ]
    )
    return "\n".join(lines)


def _add_runtime_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--queries", required=True, type=Path)
    parser.add_argument("--candidate-pool", required=True, type=Path)
    parser.add_argument("--roles", required=True, type=Path)
    parser.add_argument("--decision-trace", required=True, type=Path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="audit F000 inputs and write the manifest")
    _add_runtime_inputs(prepare)
    prepare.add_argument("--gold", required=True, type=Path)
    prepare.add_argument("--output", required=True, type=Path)

    run = commands.add_parser("run", help="run F001 without loading gold labels")
    _add_runtime_inputs(run)
    run.add_argument("--granite-snapshot", required=True, type=Path)
    run.add_argument("--nli-backend", choices=("true", "minicheck", "deberta"), default="true")
    run.add_argument(
        "--nli-model-id",
        help="local TRUE snapshot path; avoids network lookup during the frozen run",
    )
    run.add_argument("--output-dir", required=True, type=Path)
    run.add_argument("--limit", type=int)
    run.add_argument("--max-error-rate", type=float, default=DEFAULT_MAX_ERROR_RATE)

    score = commands.add_parser("score", help="score saved generations after runtime")
    score.add_argument("--generations", required=True, type=Path)
    score.add_argument("--gold", required=True, type=Path)
    score.add_argument("--output-json", required=True, type=Path)
    score.add_argument("--output-report", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command in {"prepare", "run"}:
        cases = load_runtime_cases(
            queries_path=args.queries,
            candidate_pool_path=args.candidate_pool,
            roles_path=args.roles,
            decision_trace_path=args.decision_trace,
        )
    if args.command == "prepare":
        manifest = build_input_manifest(
            cases=cases,
            source_paths={
                "queries": args.queries,
                "candidate_pool": args.candidate_pool,
                "roles": args.roles,
                "selector_decision_trace": args.decision_trace,
            },
            gold_path=args.gold,
        )
        _write_json(args.output, manifest)
        print(json.dumps(manifest, ensure_ascii=True, sort_keys=True))
        return 0
    if args.command == "run":
        if args.limit is not None:
            if args.limit <= 0:
                raise ValueError("--limit must be positive")
            cases = cases[: args.limit]
        if not 0.0 <= args.max_error_rate <= 1.0:
            raise ValueError("--max-error-rate must be in [0, 1]")
        output_dir = args.output_dir.resolve()
        if output_dir.exists() and any(output_dir.iterdir()):
            raise ValueError(f"output directory must be absent or empty: {output_dir}")
        if args.nli_model_id is not None and args.nli_backend != "true":
            raise ValueError("--nli-model-id is currently supported only with --nli-backend true")
        llm = GraniteLLMClient(
            model_id=str(args.granite_snapshot.resolve()),
            config=GraniteGenerationConfig(max_new_tokens=256, temperature=0.0, top_p=1.0),
        )
        basic = GraniteGenerator(llm=llm)
        nli = (
            TrueNLIModel(model_id=args.nli_model_id)
            if args.nli_model_id is not None
            else build_nli_model(args.nli_backend)
        )
        verify = VerifyAnnotateGenerator(
            llm=llm,
            nli=nli,
            entity_gate="observe",
            abstain_when_unverified=False,
        )
        generated_rows = run_cases(cases, basic_generator=basic, verify_generator=verify)
        error_counts: dict[str, int] = {}
        for arm in ARMS:
            count = 0
            for row in generated_rows:
                arm_rows = cast(Mapping[str, Mapping[str, object]], row["arms"])
                count += int(bool(arm_rows[arm]["error"]))
            error_counts[arm] = count
        _write_jsonl(output_dir / "generations.jsonl", generated_rows)
        run_manifest = {
            "schema_version": "full-flow-f001-run-manifest-v1",
            "status": "COMPLETE",
            "queries": len(generated_rows),
            "selector_changed_queries": sum(
                bool(row["selector_changed"]) for row in generated_rows
            ),
            "errors_by_arm": error_counts,
            "gold_loaded_at_runtime": False,
            "nli_backend": args.nli_backend,
            "nli_model_id": args.nli_model_id,
            "max_new_tokens": 256,
            "generations_sha256": _sha256_file(output_dir / "generations.jsonl"),
        }
        _write_json(output_dir / "run_manifest.json", run_manifest)
        broken = {
            arm: count / len(generated_rows)
            for arm, count in error_counts.items()
            if generated_rows and count / len(generated_rows) > args.max_error_rate
        }
        if broken:
            _write_json(output_dir / "INVALID_EXCESS_ERRORS.json", broken)
            print(f"[FAIL] arm error rate exceeded threshold: {broken}", flush=True)
            return 1
        print(json.dumps(run_manifest, ensure_ascii=True, sort_keys=True))
        return 0
    if args.command == "score":
        saved_rows = _read_generation_rows(args.generations)
        wanted = {str(row["query_id"]) for row in saved_rows}
        gold = _load_gold(args.gold, wanted)
        report = score_rows(saved_rows, gold)
        _write_json(args.output_json, report)
        args.output_report.parent.mkdir(parents=True, exist_ok=True)
        args.output_report.write_text(_markdown_report(report), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=True, sort_keys=True))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
