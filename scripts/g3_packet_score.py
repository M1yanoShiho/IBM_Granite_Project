"""Score the filled-in G3 human packet against its hidden key.

Three sections, three different questions, so they are scored separately and
never pooled:

  Section C -- evidence availability. The split decides attribution across teams:
      facts mostly ABSENT means the Generator abstained correctly and this is a
      Selector recall finding; mostly PRESENT means recheck is underperforming
      and it is a Generator defect.

  Section A -- human claim-level ground truth versus MiniCheck's claim-level and
      answer-level verdicts, per citation origin. If humans agree with the
      claim-level verdict and disagree with the answer-level one, the dilution
      effect is confirmed as a metric artifact rather than a quality difference.

  Section B -- human false-gap rate. The sample is stratified over gaps MiniCheck
      called false and gaps it called genuine, so the estimate is a stratified
      one: MiniCheck's misses in the 'genuine' cell are what correct the
      automatic 6.6% upward.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

_ITEM = re.compile(r"^#{1,6}\s*([ABC]\d+)\s*$")
_ADJ = re.compile(r"^adjudication:\s*(.*)$", re.IGNORECASE)
_REASON = re.compile(r"^reason:\s*(.*)$", re.IGNORECASE)

_NORMALISE = {
    "supported": "supported",
    "not supported": "not_supported",
    "not-supported": "not_supported",
    "unsupported": "not_supported",
    "unclear": "unclear",
    "already answered": "already_answered",
    "answered": "already_answered",
    "not answered": "not_answered",
    "present": "present",
    "absent": "absent",
    "partial": "partial",
}

# population sizes from the diagnosis run, used for the stratified estimate
AUTO_FALSE_POPULATION = 43
AUTO_GENUINE_POPULATION = 548


def parse_packet(path: Path) -> dict[str, dict[str, str]]:
    verdicts: dict[str, dict[str, str]] = {}
    current: str | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        match = _ITEM.match(stripped)
        if match:
            current = match.group(1)
            verdicts[current] = {"adjudication": "", "reason": ""}
            continue
        if current is None:
            continue
        adjudication = _ADJ.match(stripped)
        if adjudication:
            value = adjudication.group(1).strip().lower()
            verdicts[current]["adjudication"] = _NORMALISE.get(value, value)
            continue
        reason = _REASON.match(stripped)
        if reason:
            verdicts[current]["reason"] = reason.group(1).strip()
    return verdicts


def _rate(numerator: int, denominator: int) -> str:
    return (
        f"{numerator}/{denominator} ({numerator / denominator:.3f})"
        if denominator
        else "0/0 (n/a)"
    )


def score_section_c(items: dict[str, Any], verdicts: dict[str, dict[str, str]]) -> dict[str, Any]:
    counts = {"present": 0, "absent": 0, "partial": 0, "missing": 0}
    for item_id, meta in items.items():
        if meta["section"] != "C":
            continue
        human = verdicts.get(item_id, {}).get("adjudication", "")
        counts[human if human in counts else "missing"] += 1
    judged = counts["present"] + counts["absent"] + counts["partial"]
    verdict = "inconclusive"
    if judged:
        if counts["absent"] / judged >= 0.6:
            verdict = "mostly ABSENT -> Generator abstained correctly; Selector recall finding"
        elif (counts["present"] + counts["partial"]) / judged >= 0.6:
            verdict = "mostly PRESENT -> recheck underperforming; Generator defect to fix"
    return {
        "n": judged,
        "present": _rate(counts["present"], judged),
        "absent": _rate(counts["absent"], judged),
        "partial": _rate(counts["partial"], judged),
        "unfilled": counts["missing"],
        "implication": verdict,
    }


def score_section_a(items: dict[str, Any], verdicts: dict[str, dict[str, str]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for origin in ("baseline", "verified-draft", "verified-recheck"):
        rows = [i for i, m in items.items() if m["section"] == "A" and m.get("origin") == origin]
        judged = 0
        unclear = 0
        human_supported = 0
        agree_claim = claim_total = 0
        agree_answer = answer_total = 0
        for item_id in rows:
            human = verdicts.get(item_id, {}).get("adjudication", "")
            if human not in {"supported", "not_supported", "unclear"}:
                continue
            judged += 1
            if human == "unclear":
                unclear += 1
                continue
            human_bool = human == "supported"
            human_supported += int(human_bool)
            claim_level = items[item_id].get("minicheck_claim_level")
            answer_level = items[item_id].get("minicheck_answer_level")
            if claim_level is not None:
                claim_total += 1
                agree_claim += int(bool(claim_level) == human_bool)
            if answer_level is not None:
                answer_total += 1
                agree_answer += int(bool(answer_level) == human_bool)
        out[origin] = {
            "n": len(rows),
            "adjudicated": judged,
            "human_supported": _rate(human_supported, judged - unclear),
            "agreement_with_minicheck_claim_level": _rate(agree_claim, claim_total),
            "agreement_with_minicheck_answer_level": _rate(agree_answer, answer_total),
            "unclear": _rate(unclear, judged),
        }
    return out


def score_section_b(items: dict[str, Any], verdicts: dict[str, dict[str, str]]) -> dict[str, Any]:
    cells: dict[bool, dict[str, int]] = {
        True: {"judged": 0, "already": 0},
        False: {"judged": 0, "already": 0},
    }
    for item_id, meta in items.items():
        if meta["section"] != "B":
            continue
        human = verdicts.get(item_id, {}).get("adjudication", "")
        if human not in {"already_answered", "not_answered", "unclear"}:
            continue
        cell = cells[bool(meta.get("minicheck_false_gap"))]
        cell["judged"] += 1
        if human == "already_answered":
            cell["already"] += 1

    result: dict[str, Any] = {
        "minicheck_called_false": {
            "n": cells[True]["judged"],
            "human_says_already_answered": _rate(cells[True]["already"], cells[True]["judged"]),
        },
        "minicheck_called_genuine": {
            "n": cells[False]["judged"],
            "human_says_already_answered": _rate(cells[False]["already"], cells[False]["judged"]),
            "note": "these are MiniCheck's misses -- what corrects the automatic rate upward",
        },
    }
    if cells[True]["judged"] and cells[False]["judged"]:
        p_false = cells[True]["already"] / cells[True]["judged"]
        p_genuine = cells[False]["already"] / cells[False]["judged"]
        total = AUTO_FALSE_POPULATION + AUTO_GENUINE_POPULATION
        corrected = (AUTO_FALSE_POPULATION * p_false + AUTO_GENUINE_POPULATION * p_genuine) / total
        result["stratified_false_gap_estimate"] = round(corrected, 3)
        result["automatic_false_gap_rate"] = round(AUTO_FALSE_POPULATION / total, 3)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--key", type=Path, required=True)
    args = parser.parse_args()

    key = json.loads(args.key.read_text(encoding="utf-8"))
    items: dict[str, Any] = key["items"]
    verdicts = parse_packet(args.packet)

    report = {
        "seed": key.get("seed"),
        "section_C_evidence_availability": score_section_c(items, verdicts),
        "section_A_citation_adjudication": score_section_a(items, verdicts),
        "section_B_false_gap": score_section_b(items, verdicts),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(
        "\nSections are scored separately by design. Section C decides Generator vs "
        "Selector attribution; Section A decides whether the answer-level dilution was "
        "a metric artifact; Section B corrects the model-judged false-gap rate."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
