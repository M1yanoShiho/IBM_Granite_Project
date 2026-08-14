"""A pre-registration must say where it sits among the entries written before it.

This exists because ledger R10 ruled out boundary cutting on 2026-08-10, in its own title,
and R14 then proposed that same mechanism and R15 wrote it up as fact -- removed four days
later only by an arithmetic argument. No single entry was careless; the missing step was
asking whether the ledger had already answered the question.

The first test is the guard. The rest exercise the checker's own failure modes, because a
checker that has never been watched fail cannot be distinguished from one that always passes
-- which is the exact failure this whole exercise was about.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from check_preregistrations import LEDGER, RELATION_MARKER, check  # noqa: E402

PRIOR = "### R7 — something measured [2026-08-09]\n\ntext\n\n"


def ledger_with(relation: str, *, date: str = "2026-08-14") -> str:
    """A ledger holding one prior entry and one pre-registration carrying `relation`."""
    return (
        f"# ledger\n\n{PRIOR}"
        f"### R9 — a swept variable [PRE-REGISTERED {date}]\n\n"
        f"{relation}"
        "**设计:** four points.\n"
    )


def test_the_real_preregistrations_are_all_situated() -> None:
    problems = check(LEDGER.read_text(encoding="utf-8"))
    assert problems == [], "\n".join(problems)


def test_a_missing_relation_paragraph_is_caught() -> None:
    problems = check(ledger_with("**动机:** because it seemed interesting.\n\n"))
    assert len(problems) == 1
    assert "has no" in problems[0]
    assert "R10 had already ruled out" in problems[0]


def test_a_relation_paragraph_naming_nothing_is_caught() -> None:
    # The likeliest way to satisfy the letter of the rule while skipping the step it exists
    # for: write the heading, then say nothing checkable under it.
    problems = check(ledger_with(f"{RELATION_MARKER}\n本条独立于既有条目。\n\n"))
    assert len(problems) == 1
    assert "naming no prior entry" in problems[0]


def test_a_reference_to_a_nonexistent_entry_is_caught() -> None:
    problems = check(ledger_with(f"{RELATION_MARKER}\n承接 R42 的结论。\n\n"))
    assert len(problems) == 1
    assert "R42" in problems[0]
    assert "no entry in the ledger" in problems[0]


def test_citing_only_itself_does_not_count() -> None:
    # R9 naming R9 is not situating anything, and the regex would otherwise accept it.
    problems = check(ledger_with(f"{RELATION_MARKER}\n见 R9 自身的设计。\n\n"))
    assert len(problems) == 1
    assert "naming no prior entry" in problems[0]


def test_several_references_in_one_paragraph_are_all_checked() -> None:
    assert check(ledger_with(f"{RELATION_MARKER}\n承接 R7,并与 R9 的设计一致。\n\n")) == []
    problems = check(ledger_with(f"{RELATION_MARKER}\n承接 R7 与 R99。\n\n"))
    assert len(problems) == 1
    assert "R99" in problems[0]


def test_entries_written_before_the_rule_are_left_alone() -> None:
    # Backfilling a section onto an entry whose author never wrote one would misrepresent what
    # they checked, so the rule starts on the day it was adopted and not before.
    assert check(ledger_with("**动机:** none given.\n\n", date="2026-08-13")) == []


def test_a_changed_heading_format_fails_loudly_rather_than_passing_on_nothing() -> None:
    # The dangerous failure: the heading style drifts, no pre-registration is found, and a
    # check reporting success on zero entries lets every one of them go unsituated.
    problems = check("# ledger\n\n## R9 - pre-registered, but formatted differently\n\ntext\n")
    assert len(problems) == 1
    assert "no pre-registered entries found" in problems[0]
