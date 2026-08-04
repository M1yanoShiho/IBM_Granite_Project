"""Blind packet for the entity-conflict drops — the only path that destroys content.

Under verify-and-annotate everything unsupported is kept and labelled; an entity
conflict is the sole remaining reason a claim is deleted outright. `entity_check`
has a documented history of over-vetoing (60% false-veto rate before the tolerant
name-matching fix), so its precision now matters more than it did when other
deletion paths shared the load.

Same protocol as the earlier packets: the system's verdict is hidden, evidence is
never shown in score order, nothing is highlighted, and the sampling seed is
recorded.

The three-way judgement is chosen so each answer maps to a distinct consequence:

  supported  -> the drop was a **false veto**: correct content was destroyed
  conflicting -> the drop was correct
  unrelated  -> the claim should have been **annotated**, not dropped; the
                evidence gives no ground to contradict it

Usage:
  python scripts/entity_conflict_packet.py --arm results/g5/verify-annotate.jsonl \
      --packet local/audit/entity-conflict-packet.md \
      --key local/audit/entity-conflict-key.json --seed 13 --items 20
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

STANDARD = (
    "- **supported** — the passage states the claim (paraphrase is fine).\n"
    "- **conflicting** — the passage states something incompatible with the claim, "
    "for example a different person, date, or number in the same role.\n"
    "- **unrelated** — the passage neither states the claim nor contradicts it.\n"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", type=Path, required=True)
    parser.add_argument("--packet", type=Path, default=Path("local/audit/entity-conflict-packet.md"))
    parser.add_argument("--key", type=Path, default=Path("local/audit/entity-conflict-key.json"))
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--items", type=int, default=20)
    args = parser.parse_args()

    records = [
        json.loads(line)
        for line in args.arm.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    pool: list[dict[str, Any]] = []
    for record in records:
        texts = {item["evidence_id"]: item["text"] for item in record["evidence"]}
        for routing in record.get("routing", []):
            if routing.get("outcome") != "dropped_entity_conflict":
                continue
            evidence_id = routing.get("conflict_evidence_id")
            if not evidence_id or evidence_id not in texts:
                continue
            pool.append(
                {
                    "query_id": record["query_id"],
                    "question": record["question"],
                    "claim": routing.get("claim_text", ""),
                    "evidence_id": evidence_id,
                    "evidence": texts[evidence_id],
                    "conflict_detail": routing.get("conflict_detail", []),
                }
            )

    print(f"== entity-conflict population: {len(pool)} drops ==")
    rng = random.Random(args.seed)
    rng.shuffle(pool)
    chosen = pool[: args.items]

    lines = ["# Entity-conflict adjudication", ""]
    lines.append(
        "Each item shows a claim and one evidence passage. Judge the relation "
        "between them, using the passage only — the question is context."
    )
    lines.append("")
    lines.append("## Standard")
    lines.append("")
    lines.append(STANDARD)
    lines.append(
        "Fill both lines under each item. `adjudication:` must be exactly one of "
        "`supported`, `conflicting`, `unrelated`. `reason:` is one free-text line."
    )
    lines.append("")
    lines.append(f"Sampling seed: `{args.seed}`. Items: {len(chosen)}.")
    lines.append("")
    key: dict[str, Any] = {"seed": args.seed, "population": len(pool), "items": {}}
    for index, item in enumerate(chosen, start=1):
        item_id = f"E{index:02d}"
        lines.append("---")
        lines.append("")
        lines.append(f"### {item_id}")
        lines.append("")
        lines.append(f"**Question:** {item['question']}")
        lines.append("")
        lines.append(f"**Claim:** {item['claim']}")
        lines.append("")
        lines.append(f"**Evidence:** {item['evidence']}")
        lines.append("")
        lines.append("adjudication: ")
        lines.append("reason: ")
        lines.append("")
        key["items"][item_id] = {
            "query_id": item["query_id"],
            "evidence_id": item["evidence_id"],
            "conflict_detail": item["conflict_detail"],
        }

    args.packet.parent.mkdir(parents=True, exist_ok=True)
    args.packet.write_text("\n".join(lines) + "\n", encoding="utf-8")
    args.key.parent.mkdir(parents=True, exist_ok=True)
    args.key.write_text(json.dumps(key, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # population breakdown by which entity type triggered the veto
    kinds: dict[str, int] = {}
    for item in pool:
        for detail in item["conflict_detail"] or ["(none recorded)"]:
            kind = detail.split(":", 1)[0]
            kinds[kind] = kinds.get(kind, 0) + 1
    absent = sum(1 for i in pool for d in i["conflict_detail"] if d.endswith("(absent)"))
    conflicting = sum(1 for i in pool for d in i["conflict_detail"] if not d.endswith("(absent)"))
    print("mismatch kinds:", json.dumps(kinds, sort_keys=True))
    print(f"mismatch reasons: value-conflict {conflicting}, absent-from-evidence {absent}")
    print(f"sampled: {len(chosen)}  seed: {args.seed}")
    print("packet:", args.packet)
    print("key:", args.key)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
