"""A002 baseline reproduction and same-input runtime-variance audit.

The ``run`` command never accepts or loads gold. It runs the current
Verify-and-annotate Generator on TopK10 and the frozen Legacy Selector context in
one process, rotates arm order by query, and repeats a deterministic stratified
10% sample. The separate ``score`` command joins saved outputs to gold only after
generation has completed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
import traceback
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, Protocol, cast

import full_flow_joint as joint

from evidence_rag.contracts.models import (
    GenerationResult,
    Query,
    QueryChecklist,
    SelectedEvidenceSet,
    strip_annotations,
)
from evidence_rag.evaluation.paired_metric import compare_paired
from evidence_rag.evaluation.scoring import answer_match
from evidence_rag.generator.claim_splitter import FAITHFULNESS_PROMPT, SPLIT_PROMPT
from evidence_rag.generator.draft import DRAFT_PROMPT
from evidence_rag.generator.granite import GraniteGenerationConfig, GraniteLLMClient
from evidence_rag.generator.nli import TrueNLIModel
from evidence_rag.generator.trace import GeneratorTrace
from evidence_rag.generator.verify_annotate import VerifyAnnotateGenerator
from evidence_rag.infrastructure.datasets import GoldCase

ARMS = ("K_topk_base", "L_legacy_selector_base")
REPEAT_FRACTION = 0.10
SEED = 13
DEFAULT_MAX_ERROR_RATE = 0.05
REPO_ROOT = Path(__file__).resolve().parents[1]


class TracedGenerator(Protocol):
    last_trace: GeneratorTrace | None

    def generate(
        self,
        query: Query,
        checklist: QueryChecklist,
        selected: SelectedEvidenceSet,
    ) -> GenerationResult: ...


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _stable_order(query_id: str, *, namespace: str) -> int:
    return int.from_bytes(
        hashlib.sha256(f"{SEED}:{namespace}:{query_id}".encode()).digest()[:8],
        "big",
    )


def choose_repeat_ids(
    cases: Sequence[joint.JointCase], fraction: float = REPEAT_FRACTION
) -> tuple[str, ...]:
    """Choose 10% independently within changed and unchanged strata."""

    if not 0.0 < fraction <= 1.0:
        raise ValueError("repeat fraction must be in (0, 1]")
    selected: list[str] = []
    for changed in (False, True):
        stratum = [case for case in cases if case.selector_changed is changed]
        count = max(1, round(len(stratum) * fraction)) if stratum else 0
        ordered = sorted(
            stratum,
            key=lambda case: (_stable_order(case.query_id, namespace="repeat"), case.query_id),
        )
        selected.extend(case.query_id for case in ordered[:count])
    return tuple(sorted(selected))


def arm_order(query_id: str, *, repeat_index: int) -> tuple[str, str]:
    """Rotate arm order deterministically; the repeated pass uses the opposite order."""

    first = _stable_order(query_id, namespace="arm-order") % 2
    if repeat_index % 2:
        first = 1 - first
    return (ARMS[first], ARMS[1 - first])


def _run_one(
    generator: TracedGenerator,
    query: Query,
    checklist: QueryChecklist,
    selected: SelectedEvidenceSet,
) -> dict[str, object]:
    started = time.perf_counter()
    try:
        generation = generator.generate(query, checklist, selected)
        trace = generator.last_trace
        if trace is None:
            raise RuntimeError("trace-enabled Generator produced no trace")
    except Exception as error:  # noqa: BLE001 -- a runtime failure is a measured outcome
        traceback.print_exc()
        return {
            "generation": GenerationResult(
                query_id=query.query_id,
                answer="",
                cited_evidence_ids=(),
            ).model_dump(mode="json"),
            "trace": None,
            "error": f"{type(error).__name__}: {error}",
            "seconds": time.perf_counter() - started,
        }
    return {
        "generation": generation.model_dump(mode="json"),
        "trace": trace.model_dump(mode="json"),
        "error": None,
        "seconds": time.perf_counter() - started,
    }


def run_cases(
    cases: Sequence[joint.JointCase],
    *,
    generator: TracedGenerator,
    repeat_ids: set[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    started = time.perf_counter()
    for index, case in enumerate(cases, start=1):
        query = Query(query_id=case.query_id, text=case.question)
        checklist = QueryChecklist(
            query_id=case.query_id,
            focus=case.question,
            required_facts=(),
        )
        contexts = {
            ARMS[0]: SelectedEvidenceSet(query_id=case.query_id, evidence=case.topk10),
            ARMS[1]: SelectedEvidenceSet(query_id=case.query_id, evidence=case.selected),
        }
        arm_runs: dict[str, list[dict[str, object]]] = {arm: [] for arm in ARMS}
        orders = [arm_order(case.query_id, repeat_index=0)]
        if case.query_id in repeat_ids:
            orders.append(arm_order(case.query_id, repeat_index=1))
        for order in orders:
            for arm in order:
                arm_runs[arm].append(
                    _run_one(generator, query, checklist, contexts[arm])
                )
        rows.append(
            {
                "schema_version": "full-flow-a002-generation-row-v1",
                "query_id": case.query_id,
                "component_id": case.component_id,
                "selector_changed": case.selector_changed,
                "topk_evidence_ids": [item.evidence_id for item in case.topk10],
                "selected_evidence_ids": [item.evidence_id for item in case.selected],
                "repeat_selected": case.query_id in repeat_ids,
                "primary_arm_order": list(orders[0]),
                "repeat_arm_order": list(orders[1]) if len(orders) > 1 else None,
                "arms": arm_runs,
            }
        )
        if index % 10 == 0 or index == len(cases):
            elapsed = time.perf_counter() - started
            print(
                f"[A002 run] {index}/{len(cases)}; {elapsed / index:.2f}s/case",
                flush=True,
            )
    return rows


def _runtime_environment() -> dict[str, object]:
    torch = __import__("torch")
    transformers = __import__("transformers")
    gpu_names = [
        torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
    ]
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu_names": gpu_names,
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _jsonl(path: Path) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, Mapping):
            raise ValueError(f"{path}:{line_number} is not an object")
        rows.append(value)
    return rows


def _gold(path: Path, wanted: set[str]) -> dict[str, GoldCase]:
    output: dict[str, GoldCase] = {}
    for value in _jsonl(path):
        case = GoldCase.model_validate(value)
        if case.query_id in wanted:
            output[case.query_id] = case
    if set(output) != wanted:
        raise ValueError("gold does not exactly cover A002 generations")
    return output


def _generation(run: Mapping[str, object]) -> GenerationResult:
    return GenerationResult.model_validate(run["generation"])


def _answer_score(answer: str, references: tuple[str, ...]) -> float:
    value = answer_match(strip_annotations(answer), references).value
    if value is None:
        raise ValueError("answer_match unexpectedly returned an unscored value")
    return float(value)


def _scope_report(
    rows: Sequence[Mapping[str, Any]],
    scores: Mapping[str, Mapping[str, float]],
) -> dict[str, object]:
    query_ids = [str(row["query_id"]) for row in rows]
    components = {str(row["query_id"]): str(row["component_id"]) for row in rows}
    by_arm = {
        arm: {
            "answer_match": sum(scores[arm][query_id] for query_id in query_ids)
            / len(query_ids),
            "coverage": sum(scores[f"{arm}:coverage"][query_id] for query_id in query_ids)
            / len(query_ids),
        }
        for arm in ARMS
    }
    answer_comparison = asdict(
        compare_paired(
            {query_id: scores[ARMS[1]][query_id] for query_id in query_ids},
            {query_id: scores[ARMS[0]][query_id] for query_id in query_ids},
            component_ids=components,
        )
    )
    coverage_comparison = asdict(
        compare_paired(
            {query_id: scores[f"{ARMS[1]}:coverage"][query_id] for query_id in query_ids},
            {query_id: scores[f"{ARMS[0]}:coverage"][query_id] for query_id in query_ids},
            component_ids=components,
        )
    )
    wrong_to_right = sum(
        scores[ARMS[0]][query_id] == 0.0 and scores[ARMS[1]][query_id] == 1.0
        for query_id in query_ids
    )
    right_to_wrong = sum(
        scores[ARMS[0]][query_id] == 1.0 and scores[ARMS[1]][query_id] == 0.0
        for query_id in query_ids
    )
    return {
        "queries": len(query_ids),
        "by_arm": by_arm,
        "L_minus_K_answer": answer_comparison,
        "L_minus_K_coverage": coverage_comparison,
        "answer_transitions": {
            "wrong_to_right": wrong_to_right,
            "right_to_wrong": right_to_wrong,
        },
    }


def score_rows(
    rows: Sequence[Mapping[str, Any]], gold: Mapping[str, GoldCase]
) -> dict[str, object]:
    scores: dict[str, dict[str, float]] = {}
    for arm in ARMS:
        scores[arm] = {}
        scores[f"{arm}:coverage"] = {}
    errors = dict.fromkeys(ARMS, 0)
    repeat: dict[str, dict[str, int]] = {
        arm: {
            "queries": 0,
            "exact_answer": 0,
            "exact_generation": 0,
            "answer_metric_agreement": 0,
            "coverage_agreement": 0,
        }
        for arm in ARMS
    }
    unchanged_cross_arm = {
        "queries": 0,
        "exact_answer": 0,
        "exact_generation": 0,
        "answer_metric_agreement": 0,
    }

    for row in rows:
        query_id = str(row["query_id"])
        references = gold[query_id].reference_answers
        if not references:
            raise ValueError(f"query {query_id} has no reference answers")
        arms = cast(Mapping[str, Sequence[Mapping[str, object]]], row["arms"])
        primary: dict[str, GenerationResult] = {}
        for arm in ARMS:
            runs = arms[arm]
            if not runs:
                raise ValueError(f"query {query_id}, arm {arm} has no primary run")
            errors[arm] += sum(bool(run.get("error")) for run in runs)
            first = _generation(runs[0])
            primary[arm] = first
            matched = _answer_score(first.answer, references)
            scores[arm][query_id] = matched
            scores[f"{arm}:coverage"][query_id] = float(bool(first.answer.strip()))
            if len(runs) == 2:
                second = _generation(runs[1])
                second_match = _answer_score(second.answer, references)
                repeat[arm]["queries"] += 1
                repeat[arm]["exact_answer"] += int(first.answer == second.answer)
                repeat[arm]["exact_generation"] += int(first == second)
                repeat[arm]["answer_metric_agreement"] += int(matched == second_match)
                repeat[arm]["coverage_agreement"] += int(
                    bool(first.answer.strip()) == bool(second.answer.strip())
                )
        if not bool(row["selector_changed"]):
            unchanged_cross_arm["queries"] += 1
            unchanged_cross_arm["exact_answer"] += int(
                primary[ARMS[0]].answer == primary[ARMS[1]].answer
            )
            unchanged_cross_arm["exact_generation"] += int(
                primary[ARMS[0]] == primary[ARMS[1]]
            )
            unchanged_cross_arm["answer_metric_agreement"] += int(
                scores[ARMS[0]][query_id] == scores[ARMS[1]][query_id]
            )

    all_rows = list(rows)
    changed_rows = [row for row in rows if bool(row["selector_changed"])]
    unchanged_rows = [row for row in rows if not bool(row["selector_changed"])]
    return {
        "schema_version": "full-flow-a002-report-v1",
        "status": "COMPLETE",
        "queries": len(rows),
        "errors_by_arm_all_runs": errors,
        "all_decision_dev": _scope_report(all_rows, scores),
        "selector_changed": _scope_report(changed_rows, scores),
        "selector_unchanged": _scope_report(unchanged_rows, scores),
        "same_input_repeat": repeat,
        "same_input_cross_arm_unchanged": unchanged_cross_arm,
    }


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# A002 Baseline And Variance Report",
        "",
        "| Scope | Arm | Answer | Coverage |",
        "|---|---|---:|---:|",
    ]
    for scope_key, label in (
        ("all_decision_dev", "all"),
        ("selector_changed", "changed"),
        ("selector_unchanged", "unchanged"),
    ):
        scope = report[scope_key]
        for arm in ARMS:
            metrics = scope["by_arm"][arm]
            lines.append(
                f"| {label} ({scope['queries']}) | {arm} | "
                f"{100 * metrics['answer_match']:.2f}% | "
                f"{100 * metrics['coverage']:.2f}% |"
            )
    lines.extend(
        [
            "",
            "## Repeat Stability",
            "",
            "| Arm | n | Exact answer | Exact result | Metric agreement | Coverage agreement |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for arm in ARMS:
        item = report["same_input_repeat"][arm]
        n = item["queries"]
        lines.append(
            f"| {arm} | {n} | {item['exact_answer']}/{n} | "
            f"{item['exact_generation']}/{n} | {item['answer_metric_agreement']}/{n} | "
            f"{item['coverage_agreement']}/{n} |"
        )
    return "\n".join(lines) + "\n"


def _add_runtime_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--queries", required=True, type=Path)
    parser.add_argument("--candidate-pool", required=True, type=Path)
    parser.add_argument("--roles", required=True, type=Path)
    parser.add_argument("--decision-trace", required=True, type=Path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    _add_runtime_inputs(run)
    run.add_argument("--granite-snapshot", required=True, type=Path)
    run.add_argument("--true-snapshot", required=True, type=Path)
    run.add_argument("--output-dir", required=True, type=Path)
    run.add_argument("--limit", type=int)
    run.add_argument("--max-error-rate", type=float, default=DEFAULT_MAX_ERROR_RATE)
    score = commands.add_parser("score")
    score.add_argument("--generations", required=True, type=Path)
    score.add_argument("--gold", required=True, type=Path)
    score.add_argument("--output-json", required=True, type=Path)
    score.add_argument("--output-report", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "run":
        if not 0.0 <= args.max_error_rate <= 1.0:
            raise ValueError("--max-error-rate must be in [0, 1]")
        cases = joint.load_runtime_cases(
            queries_path=args.queries,
            candidate_pool_path=args.candidate_pool,
            roles_path=args.roles,
            decision_trace_path=args.decision_trace,
        )
        if args.limit is not None:
            if args.limit <= 0:
                raise ValueError("--limit must be positive")
            cases = cases[: args.limit]
        output_dir = args.output_dir.resolve()
        if output_dir.exists() and any(output_dir.iterdir()):
            raise ValueError(f"output directory must be absent or empty: {output_dir}")
        repeat_ids = set(choose_repeat_ids(cases))
        llm = GraniteLLMClient(
            model_id=str(args.granite_snapshot.resolve()),
            config=GraniteGenerationConfig(
                max_new_tokens=256,
                temperature=0.0,
                top_p=1.0,
            ),
        )
        nli = TrueNLIModel(model_id=str(args.true_snapshot.resolve()))
        generator = VerifyAnnotateGenerator(
            llm=llm,
            nli=nli,
            entity_gate="observe",
            abstain_when_unverified=False,
            trace_enabled=True,
        )
        generated_rows = run_cases(cases, generator=generator, repeat_ids=repeat_ids)
        _write_jsonl(output_dir / "generations.jsonl", generated_rows)
        errors = {
            arm: sum(
                bool(run.get("error"))
                for row in generated_rows
                for run in cast(Mapping[str, Sequence[Mapping[str, object]]], row["arms"])[
                    arm
                ]
            )
            for arm in ARMS
        }
        attempts = {
            arm: sum(
                len(cast(Mapping[str, Sequence[object]], row["arms"])[arm])
                for row in generated_rows
            )
            for arm in ARMS
        }
        manifest = {
            "schema_version": "full-flow-a002-run-manifest-v1",
            "status": "COMPLETE",
            "script_sha256": _sha256_file(Path(__file__)),
            "git_commit": _git_commit(),
            "queries": len(generated_rows),
            "selector_changed_queries": sum(
                bool(row["selector_changed"]) for row in generated_rows
            ),
            "repeat_queries": len(repeat_ids),
            "repeat_query_ids": sorted(repeat_ids),
            "repeat_query_ids_sha256": _sha256_bytes(_canonical_bytes(sorted(repeat_ids))),
            "arm_order": "stable hash by query; repeated pass reverses primary order",
            "attempts_by_arm": attempts,
            "errors_by_arm": errors,
            "gold_loaded_at_runtime": False,
            "runtime_input_sha256": _sha256_bytes(
                _canonical_bytes([case.identity_row() for case in cases])
            ),
            "source_sha256": {
                "queries": _sha256_file(args.queries),
                "candidate_pool": _sha256_file(args.candidate_pool),
                "roles": _sha256_file(args.roles),
                "decision_trace": _sha256_file(args.decision_trace),
                "granite_config": _sha256_file(args.granite_snapshot / "config.json"),
                "true_config": _sha256_file(args.true_snapshot / "config.json"),
            },
            "prompt_sha256": {
                "draft": _sha256_bytes(DRAFT_PROMPT.encode()),
                "split": _sha256_bytes(SPLIT_PROMPT.encode()),
                "faithfulness": _sha256_bytes(FAITHFULNESS_PROMPT.encode()),
            },
            "decode": {
                "max_new_tokens": 256,
                "temperature": 0.0,
                "top_p": 1.0,
                "do_sample": False,
            },
            "trace_enabled": True,
            "same_process": True,
            "shared_granite_and_true_instances": True,
            "environment": _runtime_environment(),
            "generations_sha256": _sha256_file(output_dir / "generations.jsonl"),
        }
        _write_json(output_dir / "run_manifest.json", manifest)
        broken = {
            arm: errors[arm] / attempts[arm]
            for arm in ARMS
            if attempts[arm] and errors[arm] / attempts[arm] > args.max_error_rate
        }
        if broken:
            _write_json(output_dir / "INVALID_EXCESS_ERRORS.json", broken)
            return 1
        print(json.dumps(manifest, ensure_ascii=True, sort_keys=True))
        return 0
    if args.command == "score":
        saved_rows = _jsonl(args.generations)
        gold = _gold(args.gold, {str(row["query_id"]) for row in saved_rows})
        report = score_rows(saved_rows, gold)
        report["generations_sha256"] = _sha256_file(args.generations)
        report["gold_sha256"] = _sha256_file(args.gold)
        _write_json(args.output_json, report)
        args.output_report.parent.mkdir(parents=True, exist_ok=True)
        args.output_report.write_text(_markdown(report), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=True, sort_keys=True))
        return 0
    raise AssertionError(f"unknown command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
