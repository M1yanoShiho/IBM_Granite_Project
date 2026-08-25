"""The recommendations table must stay traceable to the entries holding its evidence.

This exists because the report and `docs/hpc-run-log.md` drifted apart: a reconciliation pass
on 2026-08-13 found nine disagreements, including a recommendation that contradicted a
measurement taken three days earlier in our own ledger, and measured results that changed a
production setting but had never reached the document the other groups read.

The first test is the guard. The rest exercise the checker's own failure modes, because a
checker that has never been watched fail cannot be distinguished from one that always passes
— which is the exact failure this whole exercise was about.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from check_recommendations import check  # noqa: E402

LEDGER_TEXT = "## R7 — something measured\n\n## R9 — something else\n"


def report_with(evidence: str) -> str:
    return (
        "# Report\n\n## Current recommendations (date)\n\n"
        "| Question | Answer | Evidence |\n|---|---|---|\n"
        f"| Which retriever? | Hybrid | {evidence} |\n\n"
        "## R2 - the benchmark\n\ntext\n\n## R6 - the pipeline\n\ntext\n"
    )


def test_an_unsourced_recommendation_is_caught() -> None:
    problems = check(report_with("measured on SciFact, trust me"), LEDGER_TEXT)
    assert len(problems) == 1
    assert "cites no entry" in problems[0]


def test_a_reference_to_a_nonexistent_report_entry_is_caught() -> None:
    problems = check(report_with("R42"), LEDGER_TEXT)
    assert len(problems) == 1
    assert "no entry in the report" in problems[0]


def test_the_numbering_collision_produces_a_usable_hint() -> None:
    # The report's R9 is the inverted index; the ledger's R9 is the chunk sweep. Citing a
    # bare R9 when the ledger was meant is the mistake most likely to happen, and the least
    # likely to be noticed, so the message has to name the fix rather than just the fault.
    problems = check(report_with("R9"), LEDGER_TEXT)
    assert len(problems) == 1
    assert "ledger R9" in problems[0]
    assert "do not correspond" in problems[0]


def test_a_ledger_reference_resolves_against_the_ledger_not_the_report() -> None:
    assert check(report_with("ledger R7"), LEDGER_TEXT) == []
    # ...and the report's own entries do not satisfy a ledger reference.
    problems = check(report_with("ledger R2"), LEDGER_TEXT)
    assert len(problems) == 1
    assert "no entry in the run log" in problems[0]


def test_several_references_in_one_cell_are_all_checked() -> None:
    assert check(report_with("R2 (matrix); ledger R7 (downstream); R6 (end to end)"), LEDGER_TEXT) == []
    problems = check(report_with("R2 (matrix); ledger R99 (downstream)"), LEDGER_TEXT)
    assert len(problems) == 1
    assert "ledger R99" in problems[0]


def test_a_renamed_section_fails_loudly_rather_than_passing_on_nothing() -> None:
    # The dangerous failure: someone retitles the section, the table stops being found, and a
    # check that reports success on zero rows would let every recommendation go unsourced.
    problems = check("# Report\n\n## Recommendations\n\n| a | b | c |\n", LEDGER_TEXT)
    assert len(problems) == 1
    assert "no recommendations table found" in problems[0]
