"""Build a blind, stratified audit packet for manual adjudication of the claims
the G2 chain dropped (local/guides/audit-export-spec.md).

Input is the per-claim audit dump from
``verified_generator_calibration.py --audit-dump`` (one record per faithful
claim, carrying the system's hidden verdict + stratum + full evidence). This
script splits it into:

  * ``audit-packet.md`` -- what the human reads and fills in. NO verdicts, no
    stratum, no supporting-evidence ids, no score ordering: seeing what the
    system decided anchors the adjudicator toward agreeing with it.
  * ``audit-key.json`` -- the hidden verdicts keyed by packet item id, for
    scoring after adjudication (scripts/audit_score.py).

This tool only reshapes a dump; it makes no model calls and adjudicates nothing.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# how the packet asks the adjudicator to judge, fixed before they start
STANDARD = (
    "- **supported** -- the evidence states this; paraphrase is fine, but the fact must be there.\n"
    "- **not supported** -- the evidence does not contain this information, or says something else.\n"
    "- **unclear** -- needs a reasoning step across chunks, or is only partly supported.\n"
)


@dataclass(frozen=True)
class PacketItem:
    item_id: str
    query_id: str
    claim_id: str
    question: str
    claim_text: str
    evidence: list[dict[str, str]]  # already in blind (non-score) order
    # hidden -- goes to the key only, never the packet
    stratum: str
    system_status: str
    supporting_evidence_ids: list[str]


def _load(dump_path: Path) -> list[dict[str, Any]]:
    records = [
        json.loads(line)
        for line in dump_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise SystemExit(f"no audit records in {dump_path}")
    return records


def sample_items(
    records: list[dict[str, Any]],
    rng: random.Random,
    dropped_target: int,
    controls_target: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Stratified sample: all of the smaller dropped stratum, fill the rest from
    the larger up to ``dropped_target``; plus up to ``controls_target`` supported
    controls. Returns the chosen records and the population sizes."""
    stratum_a = [r for r in records if r["stratum"] == "A"]
    stratum_b = [r for r in records if r["stratum"] == "B"]
    supported = [r for r in records if r["stratum"] == "supported"]

    sizes = {
        "A_no_entailment": len(stratum_a),
        "B_entity_vetoed": len(stratum_b),
        "supported_controls_available": len(supported),
        "entity_veto_pairs_total": sum(int(r.get("entity_veto_pairs", 0)) for r in records),
    }

    rng.shuffle(stratum_a)
    rng.shuffle(stratum_b)
    rng.shuffle(supported)

    smaller, larger = sorted((stratum_a, stratum_b), key=len)
    if len(smaller) + len(larger) <= dropped_target:
        dropped = smaller + larger
    else:
        # aim for a balanced ~half/half so both failure modes are characterised;
        # but if one stratum is genuinely small (<= half the target) take all of
        # it and give the remainder to the larger one.
        ideal = dropped_target // 2
        take_small = min(len(smaller), ideal)
        take_large = min(len(larger), dropped_target - take_small)
        take_small = min(len(smaller), dropped_target - take_large)  # top up if larger was short
        dropped = smaller[:take_small] + larger[:take_large]

    controls = supported[:controls_target]
    return dropped + controls, sizes


def to_items(chosen: list[dict[str, Any]], rng: random.Random) -> list[PacketItem]:
    """Shuffle the chosen records into a verdict-independent order and give each a
    packet id; shuffle each item's evidence so chunk order never encodes score."""
    order = list(chosen)
    rng.shuffle(order)
    items: list[PacketItem] = []
    for index, record in enumerate(order, start=1):
        evidence = list(record["evidence"])
        rng.shuffle(evidence)
        items.append(
            PacketItem(
                item_id=f"item-{index:02d}",
                query_id=record["query_id"],
                claim_id=record["claim_id"],
                question=record["question"],
                claim_text=record["claim_text"],
                evidence=evidence,
                stratum=record["stratum"],
                system_status=record["system_status"],
                supporting_evidence_ids=list(record.get("supporting_evidence_ids", [])),
            )
        )
    return items


def render_packet(items: list[PacketItem], seed: int) -> str:
    lines: list[str] = []
    lines.append("# Audit packet -- manual adjudication of Generator claims")
    lines.append("")
    lines.append(
        "For each item, read the claim against **all** the evidence chunks shown and "
        "record your judgement. Judge only from the evidence; do not use outside "
        "knowledge. The chunks are in no particular order and none is marked."
    )
    lines.append("")
    lines.append("## Standard (fix this before you start)")
    lines.append("")
    lines.append(STANDARD)
    lines.append(
        "Fill the two lines under each item. `adjudication:` must be exactly one of "
        "`supported`, `not supported`, `unclear`. `reason:` is one free-text line."
    )
    lines.append("")
    lines.append(f"Sampling seed: `{seed}` (reproducible). Items: {len(items)}.")
    lines.append("")
    lines.append("---")
    lines.append("")
    for item in items:
        lines.append(f"### {item.item_id}")
        lines.append("")
        lines.append(f"**Question:** {item.question}")
        lines.append("")
        lines.append(f"**Claim:** {item.claim_text}")
        lines.append("")
        lines.append("**Evidence:**")
        lines.append("")
        for number, chunk in enumerate(item.evidence, start=1):
            lines.append(f"{number}. {chunk['text']}")
            lines.append("")
        lines.append("adjudication: ")
        lines.append("reason: ")
        lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines)


def build_key(items: list[PacketItem], seed: int, sizes: dict[str, int]) -> dict[str, Any]:
    return {
        "seed": seed,
        "population_sizes": sizes,
        "items": {
            item.item_id: {
                "query_id": item.query_id,
                "claim_id": item.claim_id,
                "stratum": item.stratum,
                "system_status": item.system_status,
                "supporting_evidence_ids": item.supporting_evidence_ids,
                # controls are the claims the system marked supported
                "is_control": item.stratum == "supported",
            }
            for item in items
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", type=Path, required=True, help="per-claim audit jsonl")
    parser.add_argument("--packet", type=Path, default=Path("local/audit/audit-packet.md"))
    parser.add_argument("--key", type=Path, default=Path("local/audit/audit-key.json"))
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--dropped", type=int, default=30, help="target dropped-claim sample size")
    parser.add_argument("--controls", type=int, default=10, help="supported controls to shuffle in")
    args = parser.parse_args()

    records = _load(args.dump)
    rng = random.Random(args.seed)
    chosen, sizes = sample_items(records, rng, args.dropped, args.controls)
    items = to_items(chosen, rng)

    args.packet.parent.mkdir(parents=True, exist_ok=True)
    args.packet.write_text(render_packet(items, args.seed) + "\n", encoding="utf-8")
    args.key.parent.mkdir(parents=True, exist_ok=True)
    args.key.write_text(json.dumps(build_key(items, args.seed, sizes), indent=2) + "\n", encoding="utf-8")

    chosen_counts = {
        "A_no_entailment": sum(1 for i in items if i.stratum == "A"),
        "B_entity_vetoed": sum(1 for i in items if i.stratum == "B"),
        "supported_controls": sum(1 for i in items if i.stratum == "supported"),
    }
    print("== audit packet built ==")
    print("population sizes:", json.dumps(sizes))
    print(
        "claim-vs-pair: the run-level entity-mismatch figure counts (claim,evidence) PAIRS "
        f"({sizes['entity_veto_pairs_total']} pairs); stratum B is {sizes['B_entity_vetoed']} CLAIMS."
    )
    print("sampled into packet:", json.dumps(chosen_counts), f"(total {len(items)})")
    print("seed:", args.seed)
    print("packet:", args.packet)
    print("key:", args.key)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
