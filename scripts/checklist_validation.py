"""Validate the LLM checklist analyzer against ASQA's gold disambiguations.

Route B, Task 3. The checklist must be validated **on its own terms** before it
drives anything: this is the third checklist construction, and the first two
failed in ways downstream metrics could not reveal.

`qa_pairs` cannot build the checklist -- that would feed gold answers into the
Generator's input -- but it can *validate* one, because validation is
evaluation-side. ASQA's disambiguated sub-questions are exactly the set of
readings a complete answer must cover.

Matching rule, stated explicitly: a generated requirement R **corresponds to** a
gold disambiguation G when MiniCheck judges `premise = G's sub-question,
hypothesis = R` to be entailment. MiniCheck's own recall is 0.620, so it misses
genuine correspondences and the measured precision is a **floor**.

  checklist precision = requirements matching at least one gold reading / all requirements
  checklist recall    = gold readings matched by at least one requirement / all gold readings

Precision is the gated quantity: spurious requirements produce spurious gaps,
spurious abstentions, and that is the mechanism that damaged verified-full.

Reported alongside, because they can flatter or explain the headline numbers:
 - requirements per query and the empty rate, split by whether the gold question
   actually is ambiguous (an empty rate near zero on unambiguous questions means
   the analyzer is manufacturing obligations);
 - requirements matching MORE than one gold reading, i.e. over-generic ones that
   inflate precision without carrying information.

The analyzer sees the question text only. HOLD-OUT: ALCE/ASQA only.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import g3_baseline_comparison as g3  # noqa: E402, I001  identical case construction
from evidence_rag.contracts.models import Query  # noqa: E402
from evidence_rag.generator.granite import GraniteLLMClient  # noqa: E402
from evidence_rag.generator.nli import build_nli_model  # noqa: E402
from evidence_rag.query_analysis import CHECKLIST_PROMPT, GraniteQueryAnalyzer  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=400)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--dump", type=Path, required=True)
    args = parser.parse_args()

    raw = g3.vt.ensure_asqa()
    cases = g3.build_cases(raw, args.limit, random.Random(args.seed), args.top_k)
    # gold disambiguations, EVALUATION SIDE ONLY -- never passed to the analyzer
    gold_by_id: dict[str, list[str]] = {}
    for sample in raw:
        sub_questions = [
            pair.get("question", "").strip()
            for pair in sample.get("qa_pairs", [])
            if pair.get("question", "").strip()
        ]
        gold_by_id[str(sample.get("sample_id", ""))] = sub_questions
    print(f"[data] {len(cases)} cases", flush=True)

    analyzer = GraniteQueryAnalyzer(GraniteLLMClient())
    records: list[dict[str, Any]] = []
    for n, case in enumerate(cases, start=1):
        checklist = analyzer.analyze(Query(query_id=case.query_id, text=case.question))
        records.append(
            {
                "query_id": case.query_id,
                "question": case.question,
                "requirements": list(checklist.required_facts),
                "gold_readings": gold_by_id.get(case.query_id, []),
            }
        )
        if n % 50 == 0:
            print(f"[analyze] {n}/{len(cases)}", flush=True)

    print("[judge] loading MiniCheck", flush=True)
    minicheck = build_nli_model("minicheck")
    cache: dict[tuple[str, str], bool] = {}

    def matches(gold: str, requirement: str) -> bool:
        key = (gold, requirement)
        hit = cache.get(key)
        if hit is None:
            hit = minicheck.classify(premise=gold, hypothesis=requirement) == "entailment"
            cache[key] = hit
        return hit

    req_total = req_matched = req_overgeneric = 0
    gold_total = gold_matched = 0
    per_query_counts: Counter[int] = Counter()
    empty_by_ambiguity = {"unambiguous": [0, 0], "ambiguous": [0, 0]}  # [empty, total]

    for n, record in enumerate(records, start=1):
        requirements = record["requirements"]
        gold = record["gold_readings"]
        per_query_counts[len(requirements)] += 1
        bucket = "ambiguous" if len(gold) > 1 else "unambiguous"
        empty_by_ambiguity[bucket][1] += 1
        if not requirements:
            empty_by_ambiguity[bucket][0] += 1

        matched_gold: set[int] = set()
        per_req_hits: list[int] = []
        for requirement in requirements:
            hits = [i for i, g in enumerate(gold) if matches(g, requirement)]
            per_req_hits.append(len(hits))
            matched_gold.update(hits)
        record["matched_gold_per_requirement"] = per_req_hits
        req_total += len(requirements)
        req_matched += sum(1 for h in per_req_hits if h >= 1)
        req_overgeneric += sum(1 for h in per_req_hits if h > 1)
        gold_total += len(gold)
        gold_matched += len(matched_gold)
        if n % 50 == 0:
            print(f"[match] {n}/{len(records)}", flush=True)

    precision = req_matched / req_total if req_total else 0.0
    recall = gold_matched / gold_total if gold_total else 0.0

    lines: list[str] = []
    lines.append("# Checklist validation — LLM analyzer against ASQA disambiguations")
    lines.append("")
    lines.append(
        f"Seed {args.seed}, {len(records)} questions. The analyzer saw the question text "
        "only. `qa_pairs` are evaluation-side; they never entered Generator input."
    )
    lines.append("")
    lines.append("## Matching rule")
    lines.append("")
    lines.append(
        "A generated requirement R corresponds to a gold disambiguation G when "
        "**MiniCheck** judges `premise = G's sub-question, hypothesis = R` to be "
        "entailment. MiniCheck's recall is 0.620, so it misses genuine "
        "correspondences and the measured precision is a **floor**."
    )
    lines.append("")
    lines.append("## Gated numbers")
    lines.append("")
    lines.append("| quantity | value |")
    lines.append("|---|---|")
    lines.append(f"| **checklist precision** (gated, >= 0.60) | **{precision:.3f}** ({req_matched}/{req_total}) |")
    lines.append(f"| checklist recall (reported, not gated) | {recall:.3f} ({gold_matched}/{gold_total}) |")
    lines.append(
        f"| requirements matching >1 gold reading (over-generic) | "
        f"{req_overgeneric}/{req_total} ({req_overgeneric / req_total:.3f}) |"
        if req_total
        else "| over-generic | n/a |"
    )
    lines.append("")
    lines.append("## Distribution")
    lines.append("")
    total = len(records)
    mean = sum(k * v for k, v in per_query_counts.items()) / total if total else 0.0
    lines.append(f"requirements per query — mean **{mean:.2f}**")
    lines.append("")
    lines.append("| requirements | queries |")
    lines.append("|---|---|")
    for k in sorted(per_query_counts):
        lines.append(f"| {k} | {per_query_counts[k]} ({100 * per_query_counts[k] / total:.1f}%) |")
    lines.append("")
    lines.append("| gold question type | empty checklists |")
    lines.append("|---|---|")
    for bucket, (empty, count) in empty_by_ambiguity.items():
        share = f"{empty}/{count} ({empty / count:.3f})" if count else "0/0"
        lines.append(f"| {bucket} (gold readings {'>1' if bucket == 'ambiguous' else '<=1'}) | {share} |")
    lines.append("")
    lines.append(
        "An empty rate near zero on **unambiguous** questions would mean the analyzer "
        "is manufacturing obligations; a high empty rate on **ambiguous** ones means it "
        "is missing real readings."
    )
    lines.append("")
    lines.append("## Analyzer prompt, verbatim")
    lines.append("")
    lines.append("```text")
    lines.append(CHECKLIST_PROMPT)
    lines.append("```")
    lines.append("")
    lines.append("## Samples")
    lines.append("")
    ambiguous = [r for r in records if len(r["gold_readings"]) > 1][:6]
    unambiguous = [r for r in records if len(r["gold_readings"]) <= 1][:6]
    for label, group in (("Ambiguous (multiple gold readings)", ambiguous),
                         ("Unambiguous (one gold reading)", unambiguous)):
        lines.append(f"### {label}")
        lines.append("")
        for record in group:
            lines.append(f"- **Q:** {record['question']}")
            lines.append(f"  - gold readings: {len(record['gold_readings'])}")
            for requirement in record["requirements"]:
                lines.append(f"  - req: {requirement}")
            if not record["requirements"]:
                lines.append("  - req: *(empty)*")
        lines.append("")

    report = "\n".join(lines)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report + "\n", encoding="utf-8")
    args.dump.parent.mkdir(parents=True, exist_ok=True)
    with args.dump.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print("\n" + report, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
