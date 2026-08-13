"""Enforce that every retriever recommendation names the entry its evidence lives in.

The retriever report and `docs/hpc-run-log.md` evolved side by side with nothing tying them
together, and a reconciliation pass on 2026-08-13 found nine places where they disagreed or
where one held a measured result the other never learned. Several were recommendations that
had quietly gone stale, and one was a recommendation that contradicted a measurement taken
three days earlier in our own ledger.

Fixing those was the backlog. This is the part meant to stop it recurring: the
recommendations table is what other groups act on, so every row must name where its evidence
lives, and that name must resolve to a real entry.

Two hazards it exists to catch:

- **An unsourced recommendation.** A row whose evidence cell is empty, or carries prose with
  no entry reference, cannot be checked against anything by the person acting on it.
- **The numbering collision.** The report and the ledger both number entries `R<n>`, and they
  do not correspond — the report's R9 is the inverted index while the ledger's R9 is the chunk
  sweep. A bare `R9` therefore means the report; a ledger entry must be written `ledger R9`.
  A reference resolving in neither file is an error, which is what catches a typo or a
  reference to an entry that has been renumbered.

What it deliberately does **not** check: whether the cited entry actually supports the claim.
No script can do that. It checks that a human can find the evidence in one step, which is the
part that was missing.

Run directly, or via `tests/docs/test_recommendations_sourced.py`:

    python scripts/check_recommendations.py
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs/retriever/report/retriever-progress.md"
LEDGER = ROOT / "docs/hpc-run-log.md"

TABLE_HEADING = "## Current recommendations"
REFERENCE = re.compile(r"\b(ledger\s+)?R(\d+)\b", re.IGNORECASE)
HEADING = re.compile(r"^#{2,3}\s+R(\d+)\b", re.MULTILINE)


def _headings(text: str) -> set[str]:
    """Every ``R<n>`` that has a heading of its own."""
    return {match.group(1) for match in HEADING.finditer(text)}


def _recommendation_rows(text: str) -> list[list[str]]:
    """The body rows of the recommendations table, as lists of stripped cells."""
    lines = text.splitlines()
    try:
        start = next(index for index, line in enumerate(lines) if line.startswith(TABLE_HEADING))
    except StopIteration:
        return []

    rows: list[list[str]] = []
    seen_table = False
    for line in lines[start + 1 :]:
        if line.startswith(("## ", "### ")):
            break
        if not line.startswith("|"):
            if seen_table:
                break  # the table ended; later tables in the section are not recommendations
            continue
        seen_table = True
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 3 or set("".join(cells)) <= set("-: "):
            continue  # separator row
        rows.append(cells)
    return rows[1:]  # drop the header row


def check(report_text: str, ledger_text: str) -> list[str]:
    """Return one message per problem; empty means every recommendation is sourced.

    Takes text rather than paths so the checker's own failure modes can be exercised — a
    checker nobody has watched fail is indistinguishable from one that always passes.
    """
    rows = _recommendation_rows(report_text)
    problems: list[str] = []

    if not rows:
        return [
            f"no recommendations table found under '{TABLE_HEADING}' — if the section was "
            "renamed, update TABLE_HEADING so this check keeps running rather than silently "
            "passing on nothing"
        ]

    report_entries = _headings(report_text)
    ledger_entries = _headings(ledger_text)

    for cells in rows:
        question, evidence = cells[0], cells[-1]
        references = REFERENCE.findall(evidence)
        if not references:
            problems.append(
                f"recommendation {question!r} cites no entry. Every row must name where its "
                "evidence lives: 'R<n>' for this report, 'ledger R<n>' for the run log."
            )
            continue
        for ledger_prefix, number in references:
            if ledger_prefix:
                if number not in ledger_entries:
                    problems.append(
                        f"recommendation {question!r} cites 'ledger R{number}', which has no "
                        "entry in the run log"
                    )
            elif number not in report_entries:
                hint = (
                    f" — there IS a ledger R{number}, so write 'ledger R{number}' if that is "
                    "what was meant; the two numbering schemes do not correspond"
                    if number in ledger_entries
                    else ""
                )
                problems.append(
                    f"recommendation {question!r} cites 'R{number}', which has no entry in the "
                    f"report{hint}"
                )
    return problems


def main() -> int:
    report_text = REPORT.read_text(encoding="utf-8")
    problems = check(report_text, LEDGER.read_text(encoding="utf-8"))
    if problems:
        print("Recommendations are not fully sourced:\n", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print(f"{len(_recommendation_rows(report_text))} recommendations, all citing real entries.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
