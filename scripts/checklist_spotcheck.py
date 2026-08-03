"""Blind human spot check of the generated checklists (Route B, Task 3).

Human adjudication is the instrument that caught both previous checklist
defects while automatic metrics showed nothing, so it runs before this one is
trusted. Same protocol as the earlier packets: the adjudicator sees the question
and the generated requirements only -- no gold readings, no match results, no
counts -- and the sampling seed is recorded.

The question asked is deliberately about the requirements *as requirements*: are
these reasonable things a complete answer must cover? Not "are they true", which
the analyzer is forbidden from asserting anyway.

Usage:
  python scripts/checklist_spotcheck.py --dump results/checklist-validation.jsonl \
      --packet local/audit/checklist-spotcheck.md --key local/audit/checklist-spotcheck-key.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

STANDARD = (
    "- **reasonable** — these are things a complete answer to this question really "
    "ought to cover, and they are stated as obligations rather than as guessed answers.\n"
    "- **not reasonable** — they are invented, off-topic, restate the question without "
    "adding an obligation, or assert an answer the analyzer could not know.\n"
    "- **partly** — some items are reasonable and others are not.\n"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--packet", type=Path, default=Path("local/audit/checklist-spotcheck.md"))
    parser.add_argument("--key", type=Path, default=Path("local/audit/checklist-spotcheck-key.json"))
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--items", type=int, default=20)
    args = parser.parse_args()

    records: list[dict[str, Any]] = [
        json.loads(line)
        for line in args.dump.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rng = random.Random(args.seed)

    # Sample across gold ambiguity so the check covers both "did it find the real
    # readings" and "did it invent obligations for a single-reading question".
    ambiguous = [r for r in records if len(r.get("gold_readings", [])) > 1]
    unambiguous = [r for r in records if len(r.get("gold_readings", [])) <= 1]
    rng.shuffle(ambiguous)
    rng.shuffle(unambiguous)
    # ASQA contains only ambiguous questions, so the unambiguous bucket is empty
    # there; fill from whichever bucket has items rather than under-sampling.
    half = min(len(ambiguous), args.items // 2)
    chosen = ambiguous[:half] + unambiguous[: args.items - half]
    if len(chosen) < args.items:
        remaining = [r for r in ambiguous + unambiguous if r not in chosen]
        chosen += remaining[: args.items - len(chosen)]
    rng.shuffle(chosen)

    lines: list[str] = []
    lines.append("# Checklist spot check")
    lines.append("")
    lines.append(
        "Each item shows a question and the checklist an analyzer generated for it. "
        "Judge whether those requirements are reasonable things a complete answer must "
        "cover. You are not being asked whether they are true — the analyzer is not "
        "allowed to know the answer, only to say what must be covered."
    )
    lines.append("")
    lines.append("## Standard")
    lines.append("")
    lines.append(STANDARD)
    lines.append(
        "Fill both lines under each item. `adjudication:` must be exactly one of "
        "`reasonable`, `not reasonable`, `partly`. `reason:` is one free-text line."
    )
    lines.append("")
    lines.append(f"Sampling seed: `{args.seed}`. Items: {len(chosen)}.")
    lines.append("")
    key: dict[str, Any] = {"seed": args.seed, "items": {}}
    for index, record in enumerate(chosen, start=1):
        item_id = f"K{index:02d}"
        lines.append("---")
        lines.append("")
        lines.append(f"### {item_id}")
        lines.append("")
        lines.append(f"**Question:** {record['question']}")
        lines.append("")
        lines.append("**Generated checklist:**")
        lines.append("")
        if record["requirements"]:
            for requirement in record["requirements"]:
                lines.append(f"- {requirement}")
        else:
            lines.append("- *(the analyzer produced no requirements for this question)*")
        lines.append("")
        lines.append("adjudication: ")
        lines.append("reason: ")
        lines.append("")
        key["items"][item_id] = {
            "query_id": record["query_id"],
            "n_requirements": len(record["requirements"]),
            "n_gold_readings": len(record.get("gold_readings", [])),
            "matched_gold_per_requirement": record.get("matched_gold_per_requirement", []),
        }

    args.packet.parent.mkdir(parents=True, exist_ok=True)
    args.packet.write_text("\n".join(lines) + "\n", encoding="utf-8")
    args.key.parent.mkdir(parents=True, exist_ok=True)
    args.key.write_text(json.dumps(key, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    empty = sum(1 for r in chosen if not r["requirements"])
    print(f"== spot check built ==\nitems: {len(chosen)} (empty checklists: {empty})")
    print("seed:", args.seed)
    print("packet:", args.packet)
    print("key:", args.key)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
