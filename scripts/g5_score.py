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
import re
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
from evidence_rag.contracts.models import (  # noqa: E402
    REVIEW_ANNOTATION,
    UNVERIFIED_ANNOTATION,
    ends_with_abbreviation,
    is_review_flagged,
    is_unverified_annotation,
    strip_annotations,
)

strip_unverified_marker = strip_annotations
"""Every annotation label comes off before scoring. A label left in would be
handed to the judge as part of the sentence, and would count toward STR-EM."""

ARMS = (
    "baseline",
    "verify-only",
    "verify-annotate-capped",
    "verify-annotate-open",
    "verify-annotate-nogate",
)

_MATCH_NOISE = re.compile(r"\[\d+\]|[^\w\s]")


def annotated_sent_split(answer: str) -> list[str]:
    """`sent_split`, with the annotation marker re-attached to the sentence it labels.

    The Generator writes ``"<sentence>. [unverified]"`` -- the marker sits *after*
    the terminator, so a sentence splitter carries it onto the FOLLOWING sentence
    (or strands it alone, at the end of the answer). Two consequences, both of
    which land only on the arms that annotate:

    * the marker lands on the wrong sentence, so a genuinely unverified sentence
      is scored as if it were unlabelled and its successor as if it were annotated;
    * a stranded marker counts as an extra uncited sentence, inflating ALCE's
      recall denominator by one per annotated sentence.

    Numerators are unaffected -- neither piece is ever cited -- so this only ever
    understated the annotate arms.
    """
    parts: list[str] = []
    for raw in sent_split(answer):
        text = raw.strip()
        moved = True
        while moved and parts:
            moved = False
            for label in (UNVERIFIED_ANNOTATION, REVIEW_ANNOTATION):
                if text.startswith(label):
                    parts[-1] = f"{parts[-1]} {label}"
                    text = text[len(label) :].strip()
                    moved = True
        if not text:
            continue
        # Repair the same false boundaries the contract's splitter knows about, so
        # the sentence the validator demanded a label for is the sentence scored
        # here. ALCE's tokenizer stays the primary splitter; where it is punkt this
        # is a no-op, and where it is the regex fallback it is the difference
        # between "Mount St. Helens erupted." being one sentence and two.
        if parts and ends_with_abbreviation(parts[-1]):
            parts[-1] = f"{parts[-1]} {text}"
            continue
        parts.append(text)
    return parts


def _match_key(text: str) -> str:
    """Normalised form for aligning a recorded routing entry to an answer sentence.

    Citation markers, punctuation and case are stripped because the two strings
    come from different stages: routing records the raw claim span, the answer
    sentence has been reassembled with its terminator restored.
    """
    return " ".join(_MATCH_NOISE.sub(" ", strip_unverified_marker(text)).lower().split())


def correctness(answer: str, gold: tuple[tuple[str, ...], ...], dataset: str) -> float:
    """The pre-registered correctness metric for this dataset.

    ASQA: STR-EM. QAMPARI: answer recall by containment over gold alias sets --
    computationally the same function, but named differently because the values
    are **not comparable across datasets** and must not be read as if they were.
    QAMPARI additionally reports rec@5; see the pre-registration.
    """
    if dataset == "qampari":
        from qampari_data import answer_recall

        return answer_recall(answer, list(gold))
    return str_em(answer, gold)


def build(
    records: list[dict[str, Any]], dataset: str = "asqa"
) -> tuple[list[ScoredExample], dict[str, Any]]:
    examples: list[ScoredExample] = []
    per_case: list[dict[str, Any]] = []
    annotated = kept = flagged = 0
    for record in records:
        answer = record["answer"]
        answered = bool(answer.strip())
        has_flag = False  # reset per record: a stale value would leak across records
        docs = {item["evidence_id"]: item["text"] for item in record["evidence"]}
        cited = tuple(c for c in record["cited_evidence_ids"] if c in docs)
        gold = tuple(tuple(a) for a in record.get("gold_answers", []))
        metrics: dict[str, float | None] = {
            "coverage": 1.0 if answered else 0.0,
            "answer_correctness": correctness(strip_unverified_marker(answer), gold, dataset),
        }
        if dataset == "qampari":
            from qampari_data import answer_recall

            metrics["answer_recall_at5"] = answer_recall(
                strip_unverified_marker(answer), list(gold), cap=5
            )
        if answered:
            # Exact per-sentence mapping when the arm recorded it; the flat-list
            # convention only as a fallback for arms that cannot.
            exact = {
                _match_key(r["sentence"]): r["citation"]
                for r in record.get("routing", [])
                if r.get("sentence")
            }
            sentences: list[str] = []
            citations: list[tuple[str, ...]] = []
            has_flag = False
            for raw in annotated_sent_split(answer):
                kept += 1
                if is_review_flagged(raw):
                    # Counted, then treated exactly like any other cited sentence:
                    # the label is stripped below and the citation lookup is the
                    # same one. The flag must not touch a metric.
                    flagged += 1
                    has_flag = True
                if is_unverified_annotation(raw):
                    annotated += 1
                    refs: tuple[str, ...] = ()
                elif exact:
                    # Match on normalised text, and in BOTH directions: a routing
                    # entry records the raw claim span, which stops at the claim's
                    # last word, while the answer sentence carries its terminator
                    # -- so neither string reliably contains the other.
                    key = _match_key(raw)
                    matched = next(
                        (c for s, c in exact.items() if c and s and (s in key or key in s)),
                        None,
                    )
                    refs = (matched,) if matched and matched in docs else ()
                else:
                    refs = cited
                sentences.append(remove_citations(strip_unverified_marker(raw)).strip())
                citations.append(refs)
            if sentences:
                examples.append(
                    ScoredExample(record["query_id"], tuple(sentences), tuple(citations), docs)
                )
        per_case.append(
            {
                "query_id": record["query_id"],
                "metrics": metrics,
                "has_review_flag": bool(answered and has_flag),
            }
        )
    return examples, {
        "per_case": per_case,
        "annotated_sentences": annotated,
        "kept_sentences": kept,
        "review_flagged_sentences": flagged,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        choices=("asqa", "qampari"),
        default="asqa",
        help="selects the pre-registered correctness metric; citation metrics are identical",
    )
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
        examples, extra = build(records, args.dataset)
        print(f"[score] {arm}: {len(examples)} answered of {len(records)}", flush=True)
        citation = compute_citation_metrics(examples, entails)
        # attach the per-example citation scores so paired_metric_cli can read them
        by_id = {row["example_id"]: row for row in citation.per_example}
        prec_cited: list[float] = []
        prec_flagged: list[float] = []
        prec_unflagged: list[float] = []
        for case in extra["per_case"]:
            row = by_id.get(case["query_id"])
            # ALCE scores an example that produced NO citations as precision 0. That
            # matches the reference implementation, but it contradicts this study's
            # pre-registered rule -- "annotated claims are neither numerator nor
            # denominator" -- and it only bites once an arm can emit a fully
            # annotated answer. Report both: the ALCE-convention mean for
            # comparability with published numbers, and precision over examples that
            # actually cited, which is the quantity the pre-registration named. The
            # per-case field carries the latter so paired tests compare like with
            # like; a zero-citation example simply has no precision to compare.
            has_citations = bool(row and row["citations"])
            if has_citations:
                assert row is not None
                prec_cited.append(row["citation_prec"])
                # The screening claim: does the flag concentrate the errors? This
                # is the enrichment figure, computed on THIS run rather than
                # inherited from the round that motivated the flag.
                bucket = prec_flagged if case.get("has_review_flag") else prec_unflagged
                bucket.append(row["citation_prec"])
            case["metrics"]["citation_precision"] = (
                {"value": row["citation_prec"]} if row and has_citations else None
            )
            case["metrics"]["citation_recall"] = {"value": row["citation_rec"]} if row else None
            for key in ("coverage", "answer_correctness", "answer_recall_at5"):
                if key not in case["metrics"]:
                    continue
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
            "answer_recall_at5": mean("answer_recall_at5"),
            "citation_prec": citation.citation_prec / 100,
            "citation_prec_cited_examples": (
                sum(prec_cited) / len(prec_cited) if prec_cited else None
            ),
            "examples_with_citations": len(prec_cited),
            "citation_rec": citation.citation_rec / 100,
            "annotated_sentences": extra["annotated_sentences"],
            "kept_sentences": extra["kept_sentences"],
            "review_flagged_sentences": extra["review_flagged_sentences"],
            "prec_flagged_examples": (
                sum(prec_flagged) / len(prec_flagged) if prec_flagged else None
            ),
            "n_flagged_examples": len(prec_flagged),
            "prec_unflagged_examples": (
                sum(prec_unflagged) / len(prec_unflagged) if prec_unflagged else None
            ),
            "n_unflagged_examples": len(prec_unflagged),
            "overcite": citation.sent_mcite_overcite,
        }
        print(f"[score] {arm}: {json.dumps(summary[arm])}", flush=True)

    def fmt(x: float | None) -> str:
        return f"{x:.3f}" if x is not None else "n/a"

    lines = ["# G5 — verify-and-annotate, four-arm calibration", ""]
    lines.append(
        "ALCE sentence-level citation metrics, judge **MiniCheck** (TRUE is the "
        "production verifier and never judges). Annotated sentences carry no "
        "citation, so they are outside the precision denominator and inside the "
        "recall denominator -- the intended, visible cost of the redesign."
    )
    lines.append("")
    lines.append(
        "`cite prec` is ALCE's convention, under which an example that cited nothing "
        "at all scores 0. `cite prec (cited)` restricts the mean to examples that "
        "actually produced a citation, which is what the pre-registration specified; "
        "the two differ only for an arm that can emit a fully annotated answer."
    )
    lines.append("")
    lines.append(
        "| arm | coverage | correctness (STR-EM) | cite prec | cite prec (cited) "
        "| cite recall | answered |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for arm in ARMS:
        s = summary[arm]
        lines.append(
            f"| {arm} | {fmt(s['coverage'])} | {fmt(s['answer_correctness'])} | "
            f"{fmt(s['citation_prec'])} | {fmt(s['citation_prec_cited_examples'])} "
            f"({s['examples_with_citations']}) | {fmt(s['citation_rec'])} | "
            f"{s['answered']}/{s['n']} |"
        )
    lines.append("")
    for arm in ARMS:
        s = summary[arm]
        if s["kept_sentences"] and s["annotated_sentences"]:
            lines.append(
                f"- **{arm}** — {s['annotated_sentences']}/{s['kept_sentences']} "
                f"({s['annotated_sentences'] / s['kept_sentences']:.3f}) of kept "
                "sentences carry the unverified marker."
            )
    lines.append("")
    lines.append("## Review flag — screening enrichment")
    lines.append("")
    lines.append(
        "Flagged sentences are cited and enter both metrics identically; the label "
        "is stripped before judging. A screening signal is useful when it has lift "
        "over the base rate, not when it is precise."
    )
    lines.append("")
    lines.append("| arm | flagged sentences | prec, flagged examples | prec, unflagged | lift |")
    lines.append("|---|---|---|---|---|")
    for arm in ARMS:
        s = summary[arm]
        if not s["review_flagged_sentences"]:
            continue
        flagged, unflagged = s["prec_flagged_examples"], s["prec_unflagged_examples"]
        lift = (
            f"{(1 - flagged) / (1 - unflagged):.2f}x error rate"
            if flagged is not None and unflagged is not None and unflagged < 1
            else "n/a"
        )
        lines.append(
            f"| {arm} | {s['review_flagged_sentences']} | "
            f"{fmt(flagged)} ({s['n_flagged_examples']}) | "
            f"{fmt(unflagged)} ({s['n_unflagged_examples']}) | {lift} |"
        )
    lines.append("")
    report = "\n".join(lines)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report + "\n", encoding="utf-8")
    print("\n" + report, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
