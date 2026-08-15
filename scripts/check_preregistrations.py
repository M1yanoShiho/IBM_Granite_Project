"""Enforce that every pre-registration says where it sits among the entries before it.

Ledger R10 measured, on 2026-08-10, that overlap does not matter from 0 to 60, and drew the
conclusion in its own title: boundary cutting is not a meaningful factor, because reducing
boundary cuts is the whole mechanism overlap has. Four days later R14's AFTER proposed
"the answer straddles a chunk boundary" as its explanation, R15 wrote it up as the mechanism,
and it took an arithmetic argument on 2026-08-14 to remove it again -- a span survives the
window whenever its length is at most the overlap, which no arm ever violated.

Nothing was wrong with any single entry. R14 cited ledger R9 at length and was scrupulous
about its own limits. What was missing was one step: asking whether the ledger had already
ruled the hypothesis out. It had, in a heading, in plain language.

So this checks the step, not the judgement. Every pre-registration from RULE_EFFECTIVE_FROM
onward must carry a `**与既有结论的关系:**` paragraph naming the prior entries it was checked
against, and every entry it names must exist. Inside the ledger a bare `R<n>` means a ledger
entry; the report's separate numbering is not reachable from here.

What it deliberately does **not** check, in the same spirit as `check_recommendations.py`:
whether the author actually read those entries, or whether the ones they named are the ones
that mattered. No script can do that. It makes the omission visible, which is the part that
failed. A pre-registration that names R10 and then contradicts it is a different and much
louder mistake than one that never mentions it.

Run directly, or via `tests/docs/test_preregistrations_situated.py`:

    python scripts/check_preregistrations.py
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "docs/hpc-run-log.md"

# The rule starts with the entries written the day it was adopted. Earlier pre-registrations
# are left as they were written: backfilling a section they never had would misrepresent what
# their authors actually checked, which is the opposite of the point.
RULE_EFFECTIVE_FROM = "2026-08-14"

RELATION_MARKER = "**与既有结论的关系:**"

PREREG = re.compile(
    r"^#{2,3}\s+R(?P<number>\d+)\s+—.*?\[PRE-REGISTERED\s+(?P<date>\d{4}-\d{2}-\d{2})",
    re.MULTILINE,
)
HEADING = re.compile(r"^#{2,3}\s+R(\d+)\b", re.MULTILINE)
REFERENCE = re.compile(r"\bR(\d+)\b")


def _entries(text: str) -> set[str]:
    """Every ``R<n>`` in the ledger that has a heading of its own."""
    return {match.group(1) for match in HEADING.finditer(text)}


def _body(text: str, start: int) -> str:
    """An entry's text, from its heading to the next heading of the same level or higher."""
    following = re.search(r"^#{1,3}\s", text[start:], re.MULTILINE)
    tail = text[start:]
    if following is None:
        return tail
    nxt = re.search(r"^#{1,3}\s", tail[following.end() :], re.MULTILINE)
    return tail if nxt is None else tail[: following.end() + nxt.start()]


def _relation_paragraph(body: str) -> str | None:
    """The relation paragraph, or None when the entry has no relation section at all."""
    index = body.find(RELATION_MARKER)
    if index == -1:
        return None
    rest = body[index + len(RELATION_MARKER) :]
    end = rest.find("\n\n")
    return rest if end == -1 else rest[:end]


def check(ledger_text: str) -> list[str]:
    """Return one message per problem; empty means every pre-registration is situated.

    Takes text rather than a path so the checker's own failure modes can be exercised — a
    checker nobody has watched fail is indistinguishable from one that always passes.
    """
    entries = _entries(ledger_text)
    matches = list(PREREG.finditer(ledger_text))
    problems: list[str] = []

    if not matches:
        return [
            "no pre-registered entries found — if the heading format changed, update PREREG so "
            "this check keeps running rather than silently passing on nothing"
        ]

    for match in matches:
        number, date = match.group("number"), match.group("date")
        if date < RULE_EFFECTIVE_FROM:
            continue
        label = f"R{number} [PRE-REGISTERED {date}]"
        paragraph = _relation_paragraph(_body(ledger_text, match.start()))

        if paragraph is None:
            problems.append(
                f"{label} has no {RELATION_MARKER} paragraph. Name the prior entries this "
                "hypothesis was checked against — ledger R10 had already ruled out the "
                "mechanism R14 and R15 went on to propose."
            )
            continue

        references = {found for found in REFERENCE.findall(paragraph) if found != number}
        if not references:
            problems.append(
                f"{label} has a relation paragraph naming no prior entry. It must cite at "
                "least one, written as 'R<n>'; a pre-registration that genuinely stands alone "
                "should say which entries it looked at and why none of them bear on it."
            )
            continue

        for reference in sorted(references, key=int):
            if reference not in entries:
                problems.append(
                    f"{label} cites R{reference} as related, which has no entry in the ledger"
                )
    return problems


def main() -> int:
    ledger_text = LEDGER.read_text(encoding="utf-8")
    problems = check(ledger_text)
    if problems:
        print("Pre-registrations are not situated against the ledger:\n", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    covered = sum(
        1 for match in PREREG.finditer(ledger_text) if match.group("date") >= RULE_EFFECTIVE_FROM
    )
    print(f"{covered} pre-registrations under the rule, all naming real prior entries.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
