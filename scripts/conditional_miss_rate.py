"""Conditional miss rate: how often the answer is lost *after* the right document arrives.

    CMR = P(answer_match == 0 | final_document_recall > 0)

Ledger R14 measured, on NQ at a fixed 600-word evidence budget, that the 60x10 arm gets
the gold document in front of the generator *more* often (+0.0879) while answering *fewer*
questions correctly (-0.0465), both at p=0.0001. Mean `answer_match` cannot separate the
two causes that produce that -- never retrieving the document, and retrieving it but losing
the answer string -- so R15 pre-registered this metric, which conditions on the document
having arrived and therefore isolates the second.

Two conditioning events are available and they are not interchangeable:

`selected` (the default) reuses the repository's own eligibility rule. `evaluator.py` marks
`generator.core.conditional_answer_match` unscored exactly when `generator_ineligibility`
fires, i.e. when no gold document was selected, so a non-null value *is* the "the answer
text reached the generator's input" event. That is the event R15's cut-by-chunking
hypothesis is about, which is why it is the default.

`cited` conditions on `system.core.final_document_recall > 0` instead. That metric is
`recall(cited, relevant)` -- the documents the generator *cited*, not the ones it was
given. Kept because a citation-side view answers a different question, but it is the wrong
conditioning for a question about what survived chunking.

`--baseline` additionally runs the R14 paired protocol on the per-case miss indicator,
restricted to the cases eligible in *both* arms. The restriction is what makes the
comparison possible at all: `compare_paired` refuses two arms whose scoring masks differ,
and a conditional metric's eligible set moves whenever the chunker does.

    python scripts/conditional_miss_rate.py runs/*/evaluation_report.json \
        --baseline runs/chunk-nq-b-c120o20/evaluation_report.json
"""

import argparse
import json
from pathlib import Path

from evidence_rag.evaluation.paired_metric import compare_paired

SELECTED_ANSWER = "generator.core.conditional_answer_match"
CITED_ANSWER = "system.core.answer_match"
CITED_COVER = "system.core.final_document_recall"

CONDITIONS = ("selected", "cited")


def _value(group: dict[str, object], metric: str) -> float | None:
    entry = group[metric]
    if isinstance(entry, dict):
        return entry["value"]  # type: ignore[return-value]
    return entry  # type: ignore[return-value]


def read_cases(path: Path) -> dict[str, tuple[float | None, float | None, float | None]]:
    """Return {query_id: (conditional_answer_match, answer_match, final_document_recall)}."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        case["query_id"]: (
            _value(case["generator"], SELECTED_ANSWER),
            _value(case["system"], CITED_ANSWER),
            _value(case["system"], CITED_COVER),
        )
        for case in payload["per_case"]
    }


def eligible_answers(
    cases: dict[str, tuple[float | None, float | None, float | None]],
    condition: str,
) -> dict[str, float | None]:
    """Answers for the cases that met `condition`; the rest are not defined for this metric."""
    if condition == "selected":
        return {
            query_id: conditional
            for query_id, (conditional, _, _) in cases.items()
            if conditional is not None
        }
    return {
        query_id: answer
        for query_id, (_, answer, cover) in cases.items()
        if cover is not None and cover > 0
    }


def _miss_rate(eligible: dict[str, float | None]) -> float:
    if not eligible:
        return float("nan")
    missed = sum(1 for answer in eligible.values() if answer is not None and answer == 0)
    return missed / len(eligible)


def _mean_answer(
    cases: dict[str, tuple[float | None, float | None, float | None]],
) -> float:
    answers = [answer for _, answer, _ in cases.values() if answer is not None]
    return sum(answers) / len(answers) if answers else float("nan")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument(
        "--condition",
        choices=CONDITIONS,
        default="selected",
        help="what the miss rate conditions on; see the module docstring for why they differ",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        help="also run the paired randomization test of every arm's miss indicator against this one",
    )
    arguments = parser.parse_args(argv)

    loaded = {path: read_cases(path) for path in arguments.reports}
    condition = arguments.condition

    print(f"conditioning on: gold document {condition}")
    print(f"{'report':<58} {'n':>5} {'eligible':>9} {'CMR':>8} {'answer':>8}")
    print("-" * 93)
    for path, cases in loaded.items():
        eligible = eligible_answers(cases, condition)
        print(
            f"{str(path)[-58:]:<58} {len(cases):>5} {len(eligible):>9} "
            f"{_miss_rate(eligible):>8.4f} {_mean_answer(cases):>8.4f}"
        )

    if arguments.baseline is None:
        return 0

    base = loaded.get(arguments.baseline) or read_cases(arguments.baseline)
    base_eligible = eligible_answers(base, condition)
    print()
    print("paired randomization on the conditional-miss indicator")
    print("(restricted to the cases eligible in BOTH arms)")
    for path, cases in loaded.items():
        if path == arguments.baseline:
            continue
        arm_eligible = eligible_answers(cases, condition)
        shared = sorted(set(arm_eligible) & set(base_eligible))
        if not shared:
            print(f"  {path}: no case is eligible in both arms")
            continue
        on = {query_id: float(arm_eligible[query_id] == 0) for query_id in shared}
        off = {query_id: float(base_eligible[query_id] == 0) for query_id in shared}
        result = compare_paired(on, off)
        print(
            f"  {str(path)[-52:]:<52} CMR {result.mean_on:.4f} vs {result.mean_off:.4f} "
            f"delta {result.delta:+.4f}  p {result.p_value:.4f}  "
            f"CI [{result.ci_low:+.4f}, {result.ci_high:+.4f}]  n {result.n_paired}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
