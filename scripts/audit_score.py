"""Score a filled-in audit packet against its hidden key
(local/guides/audit-export-spec.md, requirement 5).

Reads the adjudicator's ``audit-packet.md`` (with the ``adjudication:`` /
``reason:`` lines filled) plus ``audit-key.json`` and reports, PER STRATUM:

  * agreement rate between human and system
  * false-veto rate -- dropped items the human marked supported
  * in-chain citation precision -- of the supported controls, how many the human
    agrees with
  * the unclear rate, reported separately and never folded into the above

No aggregate accuracy across strata: the two dropped-failure modes (no
entailment vs entity veto) have different remedies, so pooling them hides which
one to fix.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

_ITEM_RE = re.compile(r"^#{1,6}\s*(item-\d+)\s*$", re.IGNORECASE)
_ADJUDICATION_RE = re.compile(r"^adjudication:\s*(.*)$", re.IGNORECASE)
_REASON_RE = re.compile(r"^reason:\s*(.*)$", re.IGNORECASE)

_NORMALISE = {
    "supported": "supported",
    "not supported": "not_supported",
    "not-supported": "not_supported",
    "not_supported": "not_supported",
    "unsupported": "not_supported",
    "unclear": "unclear",
}

# what the system's status means as an adjudication label
_SYSTEM_LABEL = {"supported": "supported", "unsupported": "not_supported"}


def parse_packet(packet_path: Path) -> dict[str, dict[str, str]]:
    """item_id -> {'adjudication': <normalised or ''>, 'reason': <text>}."""
    verdicts: dict[str, dict[str, str]] = {}
    current: str | None = None
    for raw in packet_path.read_text(encoding="utf-8").splitlines():
        item_match = _ITEM_RE.match(raw.strip())
        if item_match:
            current = item_match.group(1).lower()
            verdicts[current] = {"adjudication": "", "reason": ""}
            continue
        if current is None:
            continue
        adj_match = _ADJUDICATION_RE.match(raw.strip())
        if adj_match:
            value = adj_match.group(1).strip().lower()
            verdicts[current]["adjudication"] = _NORMALISE.get(value, value)
            continue
        reason_match = _REASON_RE.match(raw.strip())
        if reason_match:
            verdicts[current]["reason"] = reason_match.group(1).strip()
    return verdicts


def _rate(numerator: int, denominator: int) -> str:
    return f"{numerator}/{denominator} ({numerator / denominator:.3f})" if denominator else "0/0 (n/a)"


def score_stratum(
    item_ids: list[str],
    key_items: dict[str, Any],
    verdicts: dict[str, dict[str, str]],
) -> dict[str, Any]:
    total = len(item_ids)
    adjudicated = 0
    agree = 0
    false_veto = 0
    unclear = 0
    missing = 0
    for item_id in item_ids:
        human = verdicts.get(item_id, {}).get("adjudication", "")
        if human not in {"supported", "not_supported", "unclear"}:
            missing += 1
            continue
        adjudicated += 1
        system_label = _SYSTEM_LABEL.get(key_items[item_id]["system_status"])
        if human == "unclear":
            unclear += 1
            continue
        if human == system_label:
            agree += 1
        # a dropped item (system 'not_supported') the human calls 'supported' is a false veto
        if system_label == "not_supported" and human == "supported":
            false_veto += 1
    return {
        "n": total,
        "adjudicated": adjudicated,
        "missing": missing,
        "agreement": _rate(agree, adjudicated - unclear) if adjudicated - unclear else "n/a (all unclear/missing)",
        "false_veto": _rate(false_veto, adjudicated),
        "unclear": _rate(unclear, adjudicated),
    }


def score_controls(
    item_ids: list[str],
    verdicts: dict[str, dict[str, str]],
) -> dict[str, Any]:
    total = len(item_ids)
    adjudicated = 0
    human_supported = 0
    unclear = 0
    missing = 0
    for item_id in item_ids:
        human = verdicts.get(item_id, {}).get("adjudication", "")
        if human not in {"supported", "not_supported", "unclear"}:
            missing += 1
            continue
        adjudicated += 1
        if human == "unclear":
            unclear += 1
            continue
        if human == "supported":
            human_supported += 1
    return {
        "n": total,
        "adjudicated": adjudicated,
        "missing": missing,
        # of controls the system marked supported, share the human agrees with
        "in_chain_citation_precision": _rate(human_supported, adjudicated - unclear)
        if adjudicated - unclear
        else "n/a",
        "unclear": _rate(unclear, adjudicated),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True, help="filled-in audit-packet.md")
    parser.add_argument("--key", type=Path, required=True, help="audit-key.json")
    args = parser.parse_args()

    key = json.loads(args.key.read_text(encoding="utf-8"))
    key_items: dict[str, Any] = key["items"]
    verdicts = parse_packet(args.packet)

    by_stratum: dict[str, list[str]] = {"A": [], "B": [], "supported": []}
    for item_id, meta in key_items.items():
        by_stratum.setdefault(meta["stratum"], []).append(item_id)

    report: dict[str, Any] = {
        "seed": key.get("seed"),
        "population_sizes": key.get("population_sizes"),
        "stratum_A_no_entailment": score_stratum(by_stratum["A"], key_items, verdicts),
        "stratum_B_entity_vetoed": score_stratum(by_stratum["B"], key_items, verdicts),
        "controls_supported": score_controls(by_stratum["supported"], verdicts),
    }
    print(json.dumps(report, indent=2))
    print(
        "\nNote: strata are reported separately by design -- no aggregate accuracy. "
        "false_veto = dropped items the human calls supported (the 'method is broken' signal); "
        "in_chain_citation_precision is measured here, not extrapolated from G1's 0.966."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
