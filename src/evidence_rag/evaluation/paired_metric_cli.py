"""CLI: paired randomization test for a stage metric across two selector reports (CPU, offline).

Reads the per_case metric values from two stage_evaluation reports (e.g. selector_report.json),
pairs them by query_id, and reports the paired mean difference with a randomization p-value and a
bootstrap CI — the same protocol harm_cli uses for harm. No GPU; runs on the login node in seconds.
"""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from evidence_rag.evaluation.models import StageEvaluationReport
from evidence_rag.evaluation.paired_metric import compare_paired


def _read_report(path: Path) -> StageEvaluationReport:
    try:
        payload = Path(path).read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError(f"unable to read stage report {path}: {error}") from error
    try:
        return StageEvaluationReport.model_validate_json(payload)
    except ValidationError as error:
        raise ValueError(f"invalid stage evaluation report {path}: {error}") from error


def _metric_values(report: StageEvaluationReport, metric: str) -> dict[str, float | None]:
    if metric not in report.directions:
        raise ValueError(f"metric {metric!r} is absent from the report registry")
    return {case.query_id: case.metrics[metric].value for case in report.per_case}


def _require_same_report_identity(
    on: StageEvaluationReport, off: StageEvaluationReport
) -> None:
    if on.stage != off.stage:
        raise ValueError(f"on/off reports use different stages: {on.stage!r} vs {off.stage!r}")
    if on.dataset_signature != off.dataset_signature:
        raise ValueError("on/off reports use different dataset signatures")
    if on.metric_registry_signature != off.metric_registry_signature:
        raise ValueError("on/off reports use different metric registry signatures")
    if on.directions != off.directions:
        raise ValueError("on/off reports use different metric registries or directions")
    if on.case_ids != off.case_ids:
        raise ValueError("on/off report case IDs or their frozen order differ")


def _read_component_map(path: Path) -> dict[str, str]:
    components: dict[str, str] = {}
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"unable to read component map {path}: {error}") from error
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise ValueError(f"blank component-map record at {path}:{line_number}")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON at {path}:{line_number}: {error}") from error
        if not isinstance(record, dict):
            raise ValueError(f"component-map record at {path}:{line_number} must be an object")
        query_id = record.get("query_id")
        component_id = record.get("component_id")
        if not isinstance(query_id, str) or not query_id.strip():
            raise ValueError(
                f"component-map record at {path}:{line_number} has no non-blank query_id"
            )
        if not isinstance(component_id, str) or not component_id.strip():
            raise ValueError(
                f"component-map record at {path}:{line_number} has no non-blank component_id"
            )
        if query_id in components:
            raise ValueError(f"duplicate component-map query ID {query_id!r} at {path}:{line_number}")
        components[query_id] = component_id
    if not components:
        raise ValueError(f"component map {path} is empty")
    return components


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Paired randomization test for a stage metric")
    parser.add_argument("--on-report", required=True, type=Path)
    parser.add_argument("--off-report", required=True, type=Path)
    parser.add_argument("--metric", required=True)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument(
        "--component-map",
        type=Path,
        help="optional JSONL with query_id/component_id; whole components are resampled",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    on_report = _read_report(arguments.on_report)
    off_report = _read_report(arguments.off_report)
    _require_same_report_identity(on_report, off_report)
    on = _metric_values(on_report, arguments.metric)
    off = _metric_values(off_report, arguments.metric)
    components = (
        _read_component_map(arguments.component_map)
        if arguments.component_map is not None
        else None
    )
    comparison = compare_paired(
        on,
        off,
        component_ids=components,
        seed=arguments.seed,
        iterations=arguments.iterations,
    )
    print(
        json.dumps(
            {
                "metric": arguments.metric,
                "metric_direction": on_report.directions[arguments.metric],
                "stage": on_report.stage,
                "dataset_signature": on_report.dataset_signature,
                "metric_registry_signature": on_report.metric_registry_signature,
                "mean_on": comparison.mean_on,
                "mean_off": comparison.mean_off,
                "delta": comparison.delta,
                "p_value": comparison.p_value,
                "ci_low": comparison.ci_low,
                "ci_high": comparison.ci_high,
                "n_paired": comparison.n_paired,
                "n_queries": comparison.n_queries,
                "n_clusters": comparison.n_clusters,
                "n_total": comparison.n_total,
                "n_unscored": comparison.n_unscored,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
