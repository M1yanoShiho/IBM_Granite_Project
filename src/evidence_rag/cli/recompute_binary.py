"""CLI: recompute the binary 0B-2 report from a per-pair dump (M0 §9.10a).

A1 narrowed the relation model's output space to SUPPORTS / NOT_SUPPORTED after R012 and R012b
had already run in the three-class space. §9.10a rules that those results are **recomputed from
their `dump-*.jsonl`, never re-run** — re-running would change the model, the probe and the
label space at once and destroy the comparability the ablation ladder was built on. That ruling
had no implementation until this module, which made it an obligation resting on whoever
remembered it.

Why a three-class prediction column is the input rather than an error: both A1 metrics are
defined against SUPPORTS (`predicted == SUPPORTS`, `predicted != SUPPORTS`) and never read the
gold label, so a dump whose predictions are still REFUTES or UNKNOWN scores correctly with no
translation. See `relations.gate0b.task_report`.

`--against` is the point of the tool, not a convenience. A1 §9.1 pins `gold_supports_recall` as
UNCHANGED by the amendment, so recomputing it from a historical dump MUST reproduce the number
that dump's own sweep published. If it does not, the collapse is wrong — most likely summing
REFUTES and UNKNOWN, which is a threshold at .5 rather than a relabelling of argmax, and which
silently reclassifies every pair whose winning probability fell below it. That is the one bug in
this path that produces plausible numbers instead of a crash.
"""

import argparse
import dataclasses
import json
from collections.abc import Sequence
from pathlib import Path

from evidence_rag.relations.gate0b import TaskReport, task_report
from evidence_rag.relations.models import RelationLabel

TASK_TIER = "task"


def _model_of(row: dict[str, object]) -> str:
    """Dumps written before the weight fingerprint carry `model_id`; later ones carry
    `model_version`. Both name the arm, and a historical dump must stay readable."""
    for field in ("model_id", "model_version"):
        value = row.get(field)
        if value is not None:
            return str(value)
    raise ValueError(f"dump row names no model: {row!r}")


def recompute_from_dump(path: Path) -> dict[str, TaskReport]:
    """One binary TaskReport per arm. External-tier rows are skipped — 0B-1 is suspended
    (§9.11) and its rows carry no `kind`, so the task report cannot score them anyway."""
    by_model: dict[str, list[dict[str, object]]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("tier") != TASK_TIER:
            continue
        by_model.setdefault(_model_of(row), []).append(row)

    reports: dict[str, TaskReport] = {}
    for model, rows in by_model.items():
        reports[model] = task_report(
            kinds=[str(row["kind"]) for row in rows],
            gold=[RelationLabel(str(row["gold"])) for row in rows],
            predicted=[RelationLabel(str(row["predicted"])) for row in rows],
        )
    return reports


def _check_against(reports: dict[str, TaskReport], sweep_path: Path) -> None:
    """`gold_supports_recall` is pinned unchanged by A1 §9.1, so it is the invariant that makes
    the recomputation checkable. A mismatch is raised, not warned: if the historical number
    cannot be reproduced then every binary reading published from these dumps is suspect, which
    is not a condition to print and carry on from."""
    published = json.loads(Path(sweep_path).read_text(encoding="utf-8"))["models"]
    for model, report in reports.items():
        if model not in published:
            raise ValueError(
                f"sweep {sweep_path} has no arm {model!r}; the dump and the sweep are not from "
                "the same run"
            )
        was = float(published[model]["task"]["gold_supports_recall"])
        if abs(was - report.gold_supports_recall) > 1e-9:
            raise ValueError(
                f"gold_supports_recall for {model!r} does not reproduce: sweep published {was!r}, "
                f"recomputation gives {report.gold_supports_recall!r}. A1 §9.1 pins this metric "
                "as UNCHANGED, so a difference means the collapse is wrong — check that it "
                "relabels argmax rather than summing REFUTES and UNKNOWN, which is a threshold."
            )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Recompute the binary 0B-2 report from a dump")
    parser.add_argument("--dump", required=True, type=Path, help="per-pair dump jsonl")
    parser.add_argument(
        "--against",
        type=Path,
        help="the sweep json this dump came from; gold_supports_recall must reproduce exactly",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    reports = recompute_from_dump(arguments.dump)
    if arguments.against is not None:
        _check_against(reports, arguments.against)
    print(
        json.dumps(
            {
                "source_dump": str(arguments.dump),
                "checked_against": str(arguments.against) if arguments.against else None,
                "models": {
                    model: dataclasses.asdict(report) for model, report in reports.items()
                },
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
