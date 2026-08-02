"""G3 remediation Task 3 -- blind human adjudication packet (three sections).

Built from the G3 diagnosis dump; no model is called and nothing is adjudicated
here. Same blind protocol as the earlier audit: the arm and the automatic verdict
are hidden, evidence is never shown in score order, nothing is highlighted, and
the sampling seed is recorded.

  Section A (~30) -- citation adjudication, stratified over baseline citations,
      verified draft-origin citations and verified recheck-added citations, with
      the recheck group OVERSAMPLED. Each item shows the statement the citation
      was selected to support and asks whether that one evidence chunk supports
      it. This yields human CLAIM-level ground truth, which is what decides
      whether the answer-level dilution was a metric artifact or a real quality
      difference.

  Section B (~20) -- gap adjudication. Question + required fact + the draft
      answer: was it already answered? Human ground truth for the false-gap rate,
      needed because the automatic 6.6% is model-judged and MiniCheck's 0.620
      recall biases it downward. Deliberately stratified over gaps MiniCheck
      called false and gaps it called genuine, so the miss rate is measurable.

  Section C (~20) -- evidence availability, the priority section. For gaps
      recheck could not fill, is the fact present in the selected evidence at
      all? This decides attribution across teams: mostly absent means the
      Generator abstained correctly and it is a Selector recall finding; mostly
      present means recheck is underperforming and it is ours to fix.

Usage:
  python scripts/g3_human_packet.py --diagnosis <dump> --baseline-subsample <dump> \
      --packet local/audit/g3-packet.md --key local/audit/g3-key.json --seed 13
"""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_SENT = re.compile(r"[^.!?]*[.!?]+|\S[^.!?]*$")


def _sentences(text: str) -> list[str]:
    return [m.group().strip() for m in _SENT.finditer(text) if m.group().strip()]


@dataclass
class Item:
    item_id: str
    section: str
    prompt_fields: dict[str, Any]
    hidden: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------
# section builders
# --------------------------------------------------------------------------


def build_section_a(
    diagnosis: list[dict[str, Any]],
    baseline: list[dict[str, Any]],
    rng: random.Random,
    n_baseline: int,
    n_draft: int,
    n_recheck: int,
) -> list[Item]:
    """One statement + one cited chunk per item, identical in form across strata."""
    draft_pool: list[tuple[dict[str, Any], str, str]] = []
    recheck_pool: list[tuple[dict[str, Any], str, str]] = []
    for record in diagnosis:
        if not record["answered"]:
            continue
        text_by_id = {e["evidence_id"]: e["text"] for e in record["evidence"]}
        claim_by_id = {c["claim_id"]: c["text"] for c in record["draft_claims"]}
        for verification in record["verified_claims"]:
            if verification["status"] != "supported":
                continue
            statement = claim_by_id.get(verification["claim_id"], "")
            for evidence_id in verification["supporting_evidence_ids"]:
                if statement and evidence_id in text_by_id:
                    draft_pool.append((record, statement, evidence_id))
        for gap in record["gaps"]:
            if not gap.get("recheck_found"):
                continue
            statement = (gap.get("recheck_fragment") or "").strip()
            for evidence_id in gap.get("recheck_evidence_ids", []):
                if statement and evidence_id in text_by_id:
                    recheck_pool.append((record, statement, evidence_id))

    baseline_pool: list[tuple[dict[str, Any], str, str]] = []
    for row in baseline:
        if row["arm"] != "baseline" or not row["answer"].strip() or not row["cited_chunks"]:
            continue
        for sentence in _sentences(row["answer"]):
            for chunk in row["cited_chunks"]:
                baseline_pool.append((row, sentence, chunk["evidence_id"]))

    rng.shuffle(draft_pool)
    rng.shuffle(recheck_pool)
    rng.shuffle(baseline_pool)

    items: list[Item] = []
    for origin, pool, take in (
        ("baseline", baseline_pool, n_baseline),
        ("verified-draft", draft_pool, n_draft),
        ("verified-recheck", recheck_pool, n_recheck),
    ):
        for record, statement, evidence_id in pool[:take]:
            if origin == "baseline":
                text = next(
                    c["text"] for c in record["cited_chunks"] if c["evidence_id"] == evidence_id
                )
                minicheck = next(
                    c["minicheck_supported"]
                    for c in record["cited_chunks"]
                    if c["evidence_id"] == evidence_id
                )
                claim_level = None
            else:
                text = next(
                    e["text"] for e in record["evidence"] if e["evidence_id"] == evidence_id
                )
                minicheck = next(
                    (
                        c["minicheck_supported"]
                        for c in record.get("judged_citations", [])
                        if c["evidence_id"] == evidence_id
                    ),
                    None,
                )
                claim_level = next(
                    (
                        p["minicheck_supported"]
                        for p in record.get("pairs", [])
                        if p["evidence_id"] == evidence_id and p["claim_text"] == statement
                    ),
                    None,
                )
            items.append(
                Item(
                    item_id="",
                    section="A",
                    prompt_fields={
                        "question": record["question"],
                        "statement": statement,
                        "evidence": text,
                    },
                    hidden={
                        "origin": origin,
                        "query_id": record["query_id"],
                        "evidence_id": evidence_id,
                        "minicheck_answer_level": minicheck,
                        "minicheck_claim_level": claim_level,
                    },
                )
            )
    return items


def build_section_b(
    diagnosis: list[dict[str, Any]], rng: random.Random, n_auto_false: int, n_auto_genuine: int
) -> list[Item]:
    auto_false: list[tuple[dict[str, Any], dict[str, Any]]] = []
    auto_genuine: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for record in diagnosis:
        if not record["draft_answer"].strip():
            continue
        for gap in record["gaps"]:
            (auto_false if gap.get("false_gap") else auto_genuine).append((record, gap))
    rng.shuffle(auto_false)
    rng.shuffle(auto_genuine)

    items: list[Item] = []
    for pool, take in ((auto_false, n_auto_false), (auto_genuine, n_auto_genuine)):
        for record, gap in pool[:take]:
            items.append(
                Item(
                    item_id="",
                    section="B",
                    prompt_fields={
                        "question": record["question"],
                        "required_fact": gap["required_fact"],
                        "draft_answer": record["draft_answer"],
                    },
                    hidden={
                        "query_id": record["query_id"],
                        "minicheck_false_gap": bool(gap.get("false_gap")),
                        "recheck_found": gap.get("recheck_found"),
                    },
                )
            )
    return items


def build_section_c(diagnosis: list[dict[str, Any]], rng: random.Random, take: int) -> list[Item]:
    """Gaps recheck could not fill, shown against the FULL selected evidence."""
    pool: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for record in diagnosis:
        for gap in record["gaps"]:
            if gap.get("recheck_found") is False:
                pool.append((record, gap))
    rng.shuffle(pool)

    items: list[Item] = []
    for record, gap in pool[:take]:
        evidence = list(record["evidence"])
        rng.shuffle(evidence)  # never in score order
        items.append(
            Item(
                item_id="",
                section="C",
                prompt_fields={
                    "question": record["question"],
                    "required_fact": gap["required_fact"],
                    "evidence": [e["text"] for e in evidence],
                },
                hidden={
                    "query_id": record["query_id"],
                    "minicheck_false_gap": bool(gap.get("false_gap")),
                    "gap_question": gap.get("gap_question", ""),
                },
            )
        )
    return items


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------

SECTION_INTRO = {
    "A": (
        "## Section A — does this evidence support this statement?",
        "For each item: read the **statement** and the single **evidence** passage, "
        "and judge whether that passage supports that statement. Judge only from the "
        "passage shown; the question is context only.\n\n"
        "- **supported** — the passage states this; paraphrase is fine, but the fact must be there.\n"
        "- **not supported** — the passage does not contain this information, or says something else.\n"
        "- **unclear** — only partly supported, or it needs a reasoning step.\n",
    ),
    "B": (
        "## Section B — was this fact already answered in the draft?",
        "For each item: read the **required fact** and the **draft answer**, and judge "
        "whether the draft already stated that fact. This is not about whether the fact "
        "is true or supported — only whether the draft already said it.\n\n"
        "- **already answered** — the draft states this fact.\n"
        "- **not answered** — the draft does not state it.\n"
        "- **unclear** — partially stated, or arguable.\n",
    ),
    "C": (
        "## Section C — is this fact present in the evidence at all?",
        "For each item: read the **required fact** and **all** the evidence passages, and "
        "judge whether the fact is present anywhere in them. This decides whether a "
        "missing answer was the right call or a missed one.\n\n"
        "- **present** — some passage states the fact.\n"
        "- **absent** — no passage states it.\n"
        "- **partial** — the passages give part of it but not the whole fact.\n",
    ),
}


def render(items: list[Item], seed: int) -> str:
    lines: list[str] = []
    lines.append("# G3 human adjudication packet")
    lines.append("")
    lines.append(
        "Three independent sections. Nothing here shows what the system decided — "
        "please do not try to infer it. Fill the two lines under every item; "
        "`adjudication:` must be exactly one of the words listed for that section, "
        "and `reason:` is one free-text line."
    )
    lines.append("")
    lines.append(f"Sampling seed: `{seed}`. Items: {len(items)}.")
    lines.append("")
    for section in ("C", "A", "B"):
        heading, intro = SECTION_INTRO[section]
        section_items = [item for item in items if item.section == section]
        if not section_items:
            continue
        lines.append("---")
        lines.append("")
        lines.append(heading)
        lines.append("")
        lines.append(intro)
        lines.append("")
        for item in section_items:
            lines.append(f"### {item.item_id}")
            lines.append("")
            fields = item.prompt_fields
            lines.append(f"**Question:** {fields['question']}")
            lines.append("")
            if section == "A":
                lines.append(f"**Statement:** {fields['statement']}")
                lines.append("")
                lines.append(f"**Evidence:** {fields['evidence']}")
            elif section == "B":
                lines.append(f"**Required fact:** {fields['required_fact']}")
                lines.append("")
                lines.append(f"**Draft answer:** {fields['draft_answer']}")
            else:
                lines.append(f"**Required fact:** {fields['required_fact']}")
                lines.append("")
                lines.append("**Evidence:**")
                lines.append("")
                for number, text in enumerate(fields["evidence"], start=1):
                    lines.append(f"{number}. {text}")
                    lines.append("")
            lines.append("")
            lines.append("adjudication: ")
            lines.append("reason: ")
            lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnosis", type=Path, required=True)
    parser.add_argument("--baseline-subsample", type=Path, required=True)
    parser.add_argument("--packet", type=Path, default=Path("local/audit/g3-packet.md"))
    parser.add_argument("--key", type=Path, default=Path("local/audit/g3-key.json"))
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    def load(path: Path) -> list[dict[str, Any]]:
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    diagnosis = load(args.diagnosis)
    baseline = load(args.baseline_subsample)
    rng = random.Random(args.seed)

    items = (
        build_section_c(diagnosis, rng, take=20)
        + build_section_a(diagnosis, baseline, rng, n_baseline=8, n_draft=8, n_recheck=14)
        + build_section_b(diagnosis, rng, n_auto_false=8, n_auto_genuine=12)
    )
    # shuffle WITHIN each section so origin/stratum never correlates with position
    by_section: dict[str, list[Item]] = {}
    for item in items:
        by_section.setdefault(item.section, []).append(item)
    numbered: list[Item] = []
    for section in ("C", "A", "B"):
        section_items = by_section.get(section, [])
        rng.shuffle(section_items)
        for index, item in enumerate(section_items, start=1):
            item.item_id = f"{section}{index:02d}"
            numbered.append(item)

    args.packet.parent.mkdir(parents=True, exist_ok=True)
    args.packet.write_text(render(numbered, args.seed) + "\n", encoding="utf-8")
    key = {
        "seed": args.seed,
        "items": {
            item.item_id: {"section": item.section, **item.hidden} for item in numbered
        },
    }
    args.key.parent.mkdir(parents=True, exist_ok=True)
    args.key.write_text(json.dumps(key, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    counts: dict[str, int] = {}
    for item in numbered:
        label = item.section if item.section != "A" else f"A:{item.hidden['origin']}"
        counts[label] = counts.get(label, 0) + 1
    print("== packet built ==")
    print("items:", json.dumps(counts, sort_keys=True), f"total {len(numbered)}")
    print("seed:", args.seed)
    print("packet:", args.packet)
    print("key:", args.key)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
