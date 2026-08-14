"""R15's main metric conditions on an event, and conditioning is where it can go wrong quietly.

CMR jumping from exactly 0.0000 to 0.0668 is what located the threshold on the source passage
length, and every one of those numbers depends on two choices the summary line does not show:
which cases count as eligible, and which cases a paired comparison may use. Get either wrong
and the output still looks like a rate.

The conditioning event also is not fixed. `selected` reads the repository's own eligibility
rule; `cited` reads what the generator cited. They agreed to the last digit across all eight
points R15 and R16 measured, because the extractive generator cites exactly what it was given
-- an equivalence that breaks under a generator which cites selectively, so the two paths are
kept separate and both are exercised here.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from conditional_miss_rate import (  # noqa: E402
    eligible_answers,
    main,
    read_cases,
)


def case(
    query_id: str,
    *,
    conditional: float | None,
    answer: float | None,
    cover: float | None,
) -> dict[str, object]:
    return {
        "query_id": query_id,
        "generator": {"generator.core.conditional_answer_match": {"value": conditional}},
        "system": {
            "system.core.answer_match": {"value": answer},
            "system.core.final_document_recall": {"value": cover},
        },
    }


def write_report(path: Path, cases: list[dict[str, object]]) -> Path:
    path.write_text(json.dumps({"per_case": cases}), encoding="utf-8")
    return path


def test_selected_eligibility_follows_the_repository_s_own_rule(tmp_path: Path) -> None:
    # conditional_answer_match is unscored exactly when no gold document was selected, so a
    # non-null value *is* the "the answer text reached the generator" event.
    cases = read_cases(
        write_report(
            tmp_path / "r.json",
            [
                case("reached", conditional=0.0, answer=0.0, cover=1.0),
                case("never-reached", conditional=None, answer=0.0, cover=0.0),
            ],
        )
    )
    assert set(eligible_answers(cases, "selected")) == {"reached"}


def test_cited_eligibility_reads_a_different_event(tmp_path: Path) -> None:
    # final_document_recall is recall over *cited* documents. Under a generator that cites
    # selectively this diverges from selection, which is why the two are not interchangeable.
    cases = read_cases(
        write_report(
            tmp_path / "r.json",
            [
                case("selected-not-cited", conditional=0.0, answer=0.0, cover=0.0),
                case("cited", conditional=None, answer=0.0, cover=1.0),
            ],
        )
    )
    assert set(eligible_answers(cases, "selected")) == {"selected-not-cited"}
    assert set(eligible_answers(cases, "cited")) == {"cited"}


def test_the_rate_counts_only_the_eligible(tmp_path: Path, capsys) -> None:
    # Three eligible, one of them missed: 1/3, not 1/4. Counting the ineligible case in the
    # denominator would mix "never retrieved" back into the metric that exists to exclude it.
    report = write_report(
        tmp_path / "r.json",
        [
            case("missed", conditional=0.0, answer=0.0, cover=1.0),
            case("found-a", conditional=1.0, answer=1.0, cover=1.0),
            case("found-b", conditional=1.0, answer=1.0, cover=1.0),
            case("never-reached", conditional=None, answer=0.0, cover=0.0),
        ],
    )
    main([str(report)])
    line = next(line for line in capsys.readouterr().out.splitlines() if "r.json" in line)
    assert line.split()[-3:] == ["3", "0.3333", "0.5000"]  # eligible, CMR, mean answer


def test_the_paired_comparison_uses_only_cases_eligible_in_both_arms(
    tmp_path: Path, capsys
) -> None:
    # The restriction is what makes the comparison possible at all: compare_paired refuses two
    # arms whose scoring masks differ, and a conditional metric's eligible set moves whenever
    # the chunker does. Here "only-in-arm" is eligible on one side only and must be dropped.
    baseline = write_report(
        tmp_path / "base.json",
        [
            case("shared", conditional=1.0, answer=1.0, cover=1.0),
            case("only-in-base", conditional=1.0, answer=1.0, cover=1.0),
        ],
    )
    arm = write_report(
        tmp_path / "arm.json",
        [
            case("shared", conditional=0.0, answer=0.0, cover=1.0),
            case("only-in-base", conditional=None, answer=0.0, cover=0.0),
        ],
    )
    main([str(arm), str(baseline), "--baseline", str(baseline)])
    paired = next(line for line in capsys.readouterr().out.splitlines() if "arm.json" in line and "CMR" in line)
    assert paired.split()[-1] == "1"  # one shared case, not two
    assert "delta +1.0000" in paired  # missed here, found there


def test_the_two_conditions_are_reported_so_a_reader_knows_which_ran(
    tmp_path: Path, capsys
) -> None:
    report = write_report(
        tmp_path / "r.json", [case("q", conditional=0.0, answer=0.0, cover=1.0)]
    )
    main([str(report), "--condition", "cited"])
    assert "conditioning on: gold document cited" in capsys.readouterr().out
