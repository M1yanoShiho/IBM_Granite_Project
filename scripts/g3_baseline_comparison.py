"""G3 -- baseline comparison: verified vs generation-time citation.

Tests the method's central claim: citations produced by post-hoc verification are
more faithful than citations the model declares while generating. Three arms on
the SAME queries, same Granite, same selected evidence -- only the
post-generation treatment varies:

  baseline       GraniteGenerator (generation-time citation + its fallback)
  verify-only    draft -> verify -> repair   (completeness/recheck disabled)
  verified-full  draft -> verify -> repair -> completeness -> recheck

Judge: **MiniCheck** (not TRUE, not TRUE-derived). MiniCheck is weaker than TRUE
(G1 recall 0.620 vs 0.747), so it is a conservative judge applied identically to
all arms -- the comparison is fair, the absolute level is a floor. TRUE and
ALCE's TRUE-based metric are deliberately not used: the verified arm's citations
were selected by TRUE, so scoring with TRUE would be circular.

Metrics, all reported together (citation precision alone would be dishonest --
the verified arm abstains and is scored only on what it answers):
  1. coverage          -- fraction of queries with a non-empty answer (baseline
                          has its own abstention path too; reported).
  2. citation precision/recall -- MiniCheck, on answered queries.
  3. answer correctness -- ASQA STR-EM against gold short answers.
Plus citation precision on the both-answered subset (abstention removed as a
confound), with paired p-value + CI via evidence_rag.evaluation.paired_metric_cli.

HOLD-OUT: ALCE/ASQA only here. HotpotQA, RGB, MuSiQue-Full never loaded.

Usage:
  PYTHONPATH=src python scripts/g3_baseline_comparison.py \
    --limit 400 --seed 13 --output-dir results/g3 \
    --report docs/generator/g3-baseline-comparison-hpc.md
"""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import verifier_triage as vt  # noqa: E402, I001  ASQA loader + text helpers
from evidence_rag.contracts.models import (  # noqa: E402
    EvidenceCandidate,
    GenerationResult,
    Query,
    QueryChecklist,
    SelectedEvidenceSet,
)
from evidence_rag.generator.draft import DraftAnswerGenerator  # noqa: E402
from evidence_rag.generator.granite import GraniteGenerator, GraniteLLMClient  # noqa: E402
from evidence_rag.generator.models import RequiredFactCoverage  # noqa: E402
from evidence_rag.generator.nli import NLIModel, build_nli_model  # noqa: E402
from evidence_rag.generator.repair import AnswerRepairer  # noqa: E402
from evidence_rag.generator.verified import VerifiedGenerator  # noqa: E402
from evidence_rag.generator.verifier import Verifier  # noqa: E402

YEAR_RE = re.compile(r"\b(1[89]\d\d|20\d\d)\b")
MAX_FACTS_PER_CASE = 3
ARMS = ("baseline", "verify-only", "verified-full")


# --------------------------------------------------------------------------
# cases (ASQA): question + selected evidence + required facts + gold answers
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class G3Case:
    query_id: str
    question: str
    required_facts: tuple[str, ...]
    constraints: tuple[str, ...]
    selected: SelectedEvidenceSet
    gold_answers: tuple[tuple[str, ...], ...]
    """One tuple of acceptable short answers per ASQA qa_pair (for STR-EM)."""


def _selected(query_id: str, docs: list[dict[str, Any]], top_k: int) -> SelectedEvidenceSet:
    evidence = tuple(
        EvidenceCandidate(
            evidence_id=f"{query_id}::d{i}",
            document_id=f"{query_id}::doc{i}",
            chunk_id=f"{query_id}::chunk{i}",
            text=doc["text"],
            source_uri=f"asqa://{query_id}/{i}",
            retrieval_score=float(len(docs) - i),
            retrieval_rank=i + 1,
        )
        for i, doc in enumerate(docs[:top_k])
        if doc.get("text")
    )
    return SelectedEvidenceSet(query_id=query_id, evidence=evidence)


def build_cases(data: list[dict[str, Any]], limit: int, rng: random.Random, top_k: int = 5) -> list[G3Case]:
    """Same construction as the G2 calibration runner (human knowledge sentences
    the gold answer states become required facts), plus the ASQA qa_pair short
    answers as correctness gold. A case is kept only when it has usable evidence,
    at least one stated fact, and at least one gold short-answer set."""
    order = list(range(len(data)))
    rng.shuffle(order)
    cases: list[G3Case] = []
    for index in order:
        if len(cases) >= limit:
            break
        sample = data[index]
        long_answer = sample.get("answer", "").strip()
        docs = sample.get("docs", [])
        question = sample.get("question") or ""
        qa_pairs = sample.get("qa_pairs", [])
        if not question and qa_pairs:
            question = qa_pairs[0].get("question", "")
        if not long_answer or len(docs) < top_k or not question:
            continue

        knowledge = [
            item["content"].strip()
            for annotation in sample.get("annotations", [])
            for item in annotation.get("knowledge", [])
            if item.get("content")
        ]
        facts: list[str] = []
        seen: set[str] = set()
        for content in knowledge:
            if content in seen or vt.coverage(content, long_answer) < 0.6:
                continue
            seen.add(content)
            facts.append(content)
            if len(facts) >= MAX_FACTS_PER_CASE:
                break
        if not facts:
            continue

        gold: list[tuple[str, ...]] = []
        for pair in qa_pairs:
            answers = tuple(a for a in pair.get("short_answers", []) if isinstance(a, str) and a.strip())
            if answers:
                gold.append(answers)
        if not gold:
            continue

        query_id = str(sample.get("sample_id", f"asqa-{index}"))
        year = (YEAR_RE.search(question) or YEAR_RE.search(facts[0]))
        constraints = (f"as of {year.group(1)}",) if year else ()
        cases.append(
            G3Case(
                query_id=query_id,
                question=question,
                required_facts=tuple(facts),
                constraints=constraints,
                selected=_selected(query_id, docs, top_k),
                gold_answers=tuple(gold),
            )
        )
    return cases


# --------------------------------------------------------------------------
# arms
# --------------------------------------------------------------------------


class _AllCovered:
    """Completeness stub for the verify-only arm: every required fact is reported
    covered, so VerifiedGenerator's completeness/recheck loop never fires and the
    answer is exactly draft -> verify -> repair."""

    def check(self, answer_text: str, checklist: QueryChecklist) -> tuple[RequiredFactCoverage, ...]:
        facts = tuple(dict.fromkeys(checklist.required_facts))
        return tuple(RequiredFactCoverage(required_fact=fact, covered=True) for fact in facts)


def build_arms(llm: GraniteLLMClient, nli: NLIModel) -> dict[str, Any]:
    baseline = GraniteGenerator(llm=llm)
    verified_full = VerifiedGenerator(llm=llm, nli=nli)
    verify_only = VerifiedGenerator(
        draft_generator=DraftAnswerGenerator(llm=llm),
        verifier=Verifier(nli, llm=llm, completeness_checker=_AllCovered()),  # type: ignore[arg-type]
        repairer=AnswerRepairer(),
        llm=llm,
    )
    return {"baseline": baseline, "verify-only": verify_only, "verified-full": verified_full}


# --------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------


def run_generation(cases: list[G3Case], arms: dict[str, Any]) -> tuple[dict[str, dict[str, GenerationResult]], dict[str, int]]:
    results: dict[str, dict[str, GenerationResult]] = {name: {} for name in arms}
    errors: dict[str, int] = {name: 0 for name in arms}
    for n, case in enumerate(cases, start=1):
        query = Query(query_id=case.query_id, text=case.question)
        checklist = QueryChecklist(
            query_id=case.query_id,
            focus=case.question,
            required_facts=case.required_facts,
            constraints=case.constraints,
        )
        for name, generator in arms.items():
            try:
                results[name][case.query_id] = generator.generate(query, checklist, case.selected)
            except Exception:  # noqa: BLE001 -- record, keep the paired sample aligned
                errors[name] += 1
        if n % 25 == 0:
            print(f"[gen] {n}/{len(cases)} cases", flush=True)
    return results, errors


# --------------------------------------------------------------------------
# scoring (MiniCheck citations + ASQA STR-EM correctness)
# --------------------------------------------------------------------------


def _normalise(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.casefold()))


def str_em(answer: str, gold_answers: tuple[tuple[str, ...], ...]) -> float | None:
    """ASQA STR-EM: fraction of gold answer-sets with a member string-present in
    the answer. Abstention (empty answer) scores 0 -- not answering is not
    correct -- so 'improved citations by deleting the answer' shows up here."""
    if not gold_answers:
        return None
    normalised = _normalise(answer)
    hits = sum(
        1
        for answer_set in gold_answers
        if any(_normalise(candidate) in normalised for candidate in answer_set if candidate)
    )
    return hits / len(gold_answers)


def _supports(judge: NLIModel, premise: str, hypothesis: str) -> bool:
    return judge.classify(premise=premise, hypothesis=hypothesis) == "entailment"


def score_arm(
    arm_results: dict[str, GenerationResult],
    cases_by_id: dict[str, G3Case],
    judge: NLIModel,
) -> dict[str, Any]:
    """Per-arm report in the shape paired_metric_cli reads: per_case with a
    metrics map. citation_* is None on abstained queries so paired comparisons
    restrict to the both-answered subset automatically."""
    per_case: list[dict[str, Any]] = []
    for query_id, result in arm_results.items():
        case = cases_by_id[query_id]
        answered = bool(result.answer.strip())
        metrics: dict[str, float | None] = {
            "coverage": 1.0 if answered else 0.0,
            "answer_correctness": str_em(result.answer, case.gold_answers),
        }
        if answered:
            cited = set(result.cited_evidence_ids)
            cited_texts = [item.text for item in case.selected.evidence if item.evidence_id in cited]
            if cited_texts:
                metrics["citation_precision"] = sum(
                    1.0 for text in cited_texts if _supports(judge, text, result.answer)
                ) / len(cited_texts)
                metrics["citation_recall"] = (
                    1.0 if _supports(judge, "\n".join(cited_texts), result.answer) else 0.0
                )
            else:  # contract makes this impossible for a non-empty answer, but guard
                metrics["citation_precision"] = None
                metrics["citation_recall"] = None
        else:
            metrics["citation_precision"] = None
            metrics["citation_recall"] = None
        per_case.append(
            {
                "query_id": query_id,
                "metrics": {
                    key: ({"value": value} if value is not None else None)
                    for key, value in metrics.items()
                },
            }
        )
    return {"per_case": per_case}


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarise(report: dict[str, Any]) -> dict[str, Any]:
    def col(metric: str) -> list[float]:
        out = []
        for case in report["per_case"]:
            entry = case["metrics"].get(metric)
            if entry is not None:
                out.append(entry["value"])
        return out

    return {
        "n": len(report["per_case"]),
        "coverage": _mean(col("coverage")),
        "answer_correctness": _mean(col("answer_correctness")),
        "citation_precision_answered": _mean(col("citation_precision")),
        "citation_recall_answered": _mean(col("citation_recall")),
        "answered": len(col("citation_precision")),
    }


# --------------------------------------------------------------------------
# human subsample dump (built into a blind packet after the run)
# --------------------------------------------------------------------------


def human_subsample_records(
    results: dict[str, dict[str, GenerationResult]],
    cases_by_id: dict[str, G3Case],
    judge: NLIModel,
) -> list[dict[str, Any]]:
    """One record per (arm, answered query): the answer, every cited chunk, and
    the hidden per-chunk MiniCheck verdict + arm. A later export step builds the
    blind ~40-item packet across both arms."""
    records: list[dict[str, Any]] = []
    for arm in ("baseline", "verified-full"):
        for query_id, result in results.get(arm, {}).items():
            if not result.answer.strip():
                continue
            case = cases_by_id[query_id]
            cited = set(result.cited_evidence_ids)
            chunks = [
                {"evidence_id": item.evidence_id, "text": item.text,
                 "minicheck_supported": _supports(judge, item.text, result.answer)}
                for item in case.selected.evidence
                if item.evidence_id in cited
            ]
            records.append(
                {
                    "arm": arm,
                    "query_id": query_id,
                    "question": case.question,
                    "answer": result.answer,
                    "cited_chunks": chunks,
                }
            )
    return records


# --------------------------------------------------------------------------
# memory proof / cleanup
# --------------------------------------------------------------------------


def _report_peak(label: str) -> None:
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return
    if torch.cuda.is_available():
        print(
            f"[mem] {label}: peak {torch.cuda.max_memory_allocated() / 1024**3:.1f}GB "
            f"of {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB",
            flush=True,
        )


def _free() -> None:
    gc.collect()
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def render_summary(summaries: dict[str, dict[str, Any]], args: argparse.Namespace) -> str:
    lines = ["# G3 baseline comparison (Generator Part B)", ""]
    lines.append(
        "Verified vs generation-time citation. Judge: MiniCheck (not TRUE). "
        f"Seed {args.seed}, {args.limit} cases requested, ASQA calibration."
    )
    lines.append("")
    lines.append("| arm | coverage | answer correctness | cite precision (answered) | cite recall (answered) | answered |")
    lines.append("|---|---|---|---|---|---|")
    for arm in ARMS:
        s = summaries[arm]
        def fmt(x: float | None) -> str:
            return f"{x:.3f}" if x is not None else "n/a"
        lines.append(
            f"| {arm} | {fmt(s['coverage'])} | {fmt(s['answer_correctness'])} | "
            f"{fmt(s['citation_precision_answered'])} | {fmt(s['citation_recall_answered'])} | {s['answered']}/{s['n']} |"
        )
    lines.append("")
    lines.append(
        "Paired citation-precision on the both-answered subset (baseline vs "
        "verified-full / verify-only) is computed separately with "
        "`evidence_rag.evaluation.paired_metric_cli` on the per-arm reports."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=400)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output-dir", type=Path, default=Path("results/g3"))
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    print("[data] loading ALCE/ASQA", flush=True)
    cases = build_cases(vt.ensure_asqa(), args.limit, rng, top_k=args.top_k)
    print(f"[data] built {len(cases)} cases", flush=True)
    if len(cases) < 300:
        print(f"[warn] only {len(cases)} cases (<300); headline undersized", flush=True)
    cases_by_id = {case.query_id: case for case in cases}

    llm = GraniteLLMClient()
    nli = build_nli_model("true")  # production verifier for the verified arms
    arms = build_arms(llm, nli)

    print("[gen] running three arms (Granite + TRUE resident)", flush=True)
    results, errors = run_generation(cases, arms)
    _report_peak("after generation")
    print(f"[gen] chain errors per arm: {errors}", flush=True)
    del nli, arms
    _free()

    print("[judge] loading MiniCheck", flush=True)
    judge = build_nli_model("minicheck")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        report = score_arm(results[arm], cases_by_id, judge)
        (args.output_dir / f"{arm}-report.json").write_text(json.dumps(report), encoding="utf-8")
        summaries[arm] = summarise(report)
        print(f"[score] {arm}: {json.dumps(summaries[arm])}", flush=True)

    human = human_subsample_records(results, cases_by_id, judge)
    (args.output_dir / "human-subsample.jsonl").write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in human) + "\n",
        encoding="utf-8",
    )
    _report_peak("after scoring")

    summary_md = render_summary(summaries, args)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(summary_md + "\n", encoding="utf-8")
    print("\n" + summary_md, flush=True)
    print(f"[done] per-arm reports + human-subsample in {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
