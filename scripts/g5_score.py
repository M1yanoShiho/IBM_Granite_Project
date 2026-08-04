"""Score the G5 arms with ALCE sentence-level citation metrics + STR-EM.

Judge is **MiniCheck**. TRUE is the production verifier and never judges.

Per-sentence citation convention, applied **identically to every arm**:

  * a sentence carrying the unverified annotation has **no** citations -- which is
    exactly the pre-registered rule ("citation precision is computed over cited
    claims only; annotated claims are neither numerator nor denominator"), since
    ALCE leaves uncited sentences out of the precision denominator while still
    counting them in the recall denominator. That recall cost is the intended,
    visible price of the redesign;
  * any other sentence carries the answer's citation set.

The second half is the same generous convention G3 used for the baseline, because
`GenerationResult` keeps a flat citation list and the per-sentence mapping is only
exactly recoverable in ~70% of verify-annotate answers. Applying one convention to
all three arms keeps the comparison fair; the limitation is stated in the report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from alce_metrics import ScoredExample, compute_citation_metrics  # noqa: E402, I001
from g3_baseline_comparison import str_em  # noqa: E402
from g3_sentence_rescore import remove_citations, sent_split  # noqa: E402
from evidence_rag.generator.verify_annotate import (  # noqa: E402
    is_unverified_annotation,
    strip_unverified_marker,
)

ARMS = ("baseline", "verify-only", "verify-annotate")


def build(records: list[dict[str, Any]]) -> tuple[list[ScoredExample], dict[str, Any]]:
    examples: list[ScoredExample] = []
    per_case: list[dict[str, Any]] = []
    annotated = kept = 0
    for record in records:
        answer = record["answer"]
        answered = bool(answer.strip())
        docs = {item["evidence_id"]: item["text"] for item in record["evidence"]}
        cited = tuple(c for c in record["cited_evidence_ids"] if c in docs)
        gold = tuple(tuple(a) for a in record.get("gold_answers", []))
        metrics: dict[str, float | None] = {
            "coverage": 1.0 if answered else 0.0,
            "answer_correctness": str_em(strip_unverified_marker(answer), gold),
        }
        if answered:
            sentences: list[str] = []
            citations: list[tuple[str, ...]] = []
            for raw in sent_split(answer):
                kept += 1
                if is_unverified_annotation(raw):
                    annotated += 1
                    refs: tuple[str, ...] = ()
                else:
                    refs = cited
                sentences.append(remove_citations(strip_unverified_marker(raw)).strip())
                citations.append(refs)
            if sentences:
                examples.append(
                    ScoredExample(record["query_id"], tuple(sentences), tuple(citations), docs)
                )
        per_case.append({"query_id": record["query_id"], "metrics": metrics})
    return examples, {
        "per_case": per_case,
        "annotated_sentences": annotated,
        "kept_sentences": kept,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    from evidence_rag.generator.nli import build_nli_model

    print("[judge] loading MiniCheck", flush=True)
    minicheck = build_nli_model("minicheck")

    def entails(premise: str, hypothesis: str) -> bool:
        return minicheck.classify(premise=premise, hypothesis=hypothesis) == "entailment"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {}
    for arm in ARMS:
        path = args.input_dir / f"{arm}.jsonl"
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        examples, extra = build(records)
        print(f"[score] {arm}: {len(examples)} answered of {len(records)}", flush=True)
        citation = compute_citation_metrics(examples, entails)
        # attach the per-example citation scores so paired_metric_cli can read them
        by_id = {row["example_id"]: row for row in citation.per_example}
        for case in extra["per_case"]:
            row = by_id.get(case["query_id"])
            case["metrics"]["citation_precision"] = (
                {"value": row["citation_prec"]} if row else None
            )
            case["metrics"]["citation_recall"] = {"value": row["citation_rec"]} if row else None
            for key in ("coverage", "answer_correctness"):
                value = case["metrics"][key]
                case["metrics"][key] = {"value": value} if value is not None else None
        (args.output_dir / f"{arm}-report.json").write_text(
            json.dumps({"per_case": extra["per_case"]}), encoding="utf-8"
        )

        def mean(key: str, cases: list[dict[str, Any]] = extra["per_case"]) -> float | None:
            values = [
                c["metrics"][key]["value"] for c in cases if c["metrics"].get(key) is not None
            ]
            return sum(values) / len(values) if values else None

        summary[arm] = {
            "n": len(records),
            "answered": len(examples),
            "coverage": mean("coverage"),
            "answer_correctness": mean("answer_correctness"),
            "citation_prec": citation.citation_prec / 100,
            "citation_rec": citation.citation_rec / 100,
            "annotated_sentences": extra["annotated_sentences"],
            "kept_sentences": extra["kept_sentences"],
            "overcite": citation.sent_mcite_overcite,
        }
        print(f"[score] {arm}: {json.dumps(summary[arm])}", flush=True)

    lines = ["# G5 — verify-and-annotate, three-arm calibration", ""]
    lines.append(
        "ALCE sentence-level citation metrics, judge **MiniCheck** (TRUE is the "
        "production verifier and never judges). Annotated sentences carry no "
        "citation, so they are outside the precision denominator and inside the "
        "recall denominator -- the intended, visible cost of the redesign."
    )
    lines.append("")
    lines.append("| arm | coverage | correctness (STR-EM) | cite precision | cite recall | answered |")
    lines.append("|---|---|---|---|---|---|")
    for arm in ARMS:
        s = summary[arm]

        def fmt(x: float | None) -> str:
            return f"{x:.3f}" if x is not None else "n/a"

        lines.append(
            f"| {arm} | {fmt(s['coverage'])} | {fmt(s['answer_correctness'])} | "
            f"{fmt(s['citation_prec'])} | {fmt(s['citation_rec'])} | {s['answered']}/{s['n']} |"
        )
    lines.append("")
    annotate = summary["verify-annotate"]
    if annotate["kept_sentences"]:
        lines.append(
            f"Annotation rate: **{annotate['annotated_sentences']}/{annotate['kept_sentences']}** "
            f"({annotate['annotated_sentences'] / annotate['kept_sentences']:.3f}) of kept "
            "sentences carry the unverified marker."
        )
    lines.append("")
    report = "\n".join(lines)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report + "\n", encoding="utf-8")
    print("\n" + report, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
