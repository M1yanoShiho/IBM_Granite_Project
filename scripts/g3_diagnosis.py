"""G3 follow-up -- diagnose the completeness loop before removing it.

(local/guides/G3-followup-diagnosis-guide.md)

G3 localised the damage to the completeness/recheck loop, but there is a cheap
alternative explanation: B4's own-fact coverage was 0.786 in G2, i.e. a ~21%
FALSE-GAP rate. A fact already in the draft gets flagged missing one time in
five, which alone produces both observed harms -- recheck hunts for something
already answered (weak fragment, failing citation), and a missing-fact
abstention fires that was never warranted. Same shape as the entity_check
false-veto: do not condemn a component when its FIRING RULE is what is broken.

The G3 dumps are aggregate (query_id + 4 metrics) plus final answers/citations;
they carry no per-gap, citation-origin, or abstention-reason data, and no TRUE
probabilities. This script re-runs ONLY the verified-full arm on the identical
case set (same seed, greedy decoding -> deterministic, so it reproduces the G3
behaviour) with full instrumentation, then answers Tasks 1-3 and produces the
Task 5 sweep material.

All instrumentation is script-side wrappers: `src/evidence_rag/` is untouched,
as the guide requires (diagnosis precedes remedy).

Judge is MiniCheck throughout -- never TRUE or anything TRUE-derived.

HOLD-OUT: ALCE/ASQA only. HotpotQA, RGB, MuSiQue-Full never loaded.

Usage:
  PYTHONPATH=src python scripts/g3_diagnosis.py \
    --limit 400 --seed 13 \
    --dump results/g3-diagnosis/diagnosis.jsonl \
    --report docs/generator/g3-followup-diagnosis.md
"""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import g3_baseline_comparison as g3  # noqa: E402, I001  identical case construction
from evidence_rag.contracts.models import Query, QueryChecklist  # noqa: E402
from evidence_rag.generator.attribution import EntityMismatchRecord  # noqa: E402
from evidence_rag.generator.draft import DraftAnswerGenerator  # noqa: E402
from evidence_rag.generator.entity_check import EntityConsistencyChecker  # noqa: E402
from evidence_rag.generator.evidence_recheck import EvidenceRechecker  # noqa: E402
from evidence_rag.generator.granite import GraniteLLMClient  # noqa: E402
from evidence_rag.generator.nli import build_nli_model  # noqa: E402
from evidence_rag.generator.repair import AnswerRepairer  # noqa: E402
from evidence_rag.generator.verified import VerifiedGenerator  # noqa: E402
from evidence_rag.generator.verifier import Verifier  # noqa: E402

SWEEP_THRESHOLDS = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)


# --------------------------------------------------------------------------
# instrumentation (script-side only; src/ untouched)
# --------------------------------------------------------------------------


class _RecordingNLI:
    """Wraps the TRUE backend to persist the per-pair probability the production
    path throws away, and to run the entity check on EVERY pair.

    ``classify`` derives the label from the same single ``score`` call the inner
    model would have made, so behaviour is identical and no extra forward pass is
    spent. The entity check is the cheap rule-based one; running it on pairs that
    NLI called neutral costs nothing and is what makes the Task-5 threshold sweep
    possible (a lower threshold turns some of those pairs into entailments, and
    their entity verdict has to be known to re-derive support)."""

    def __init__(self, inner: Any, threshold: float) -> None:
        self.inner = inner
        self.threshold = threshold
        self.entity_checker = EntityConsistencyChecker()
        self.pairs: list[dict[str, Any]] = []
        self._evidence_ids: dict[str, str] = {}

    def set_context(self, evidence_ids_by_text: dict[str, str]) -> None:
        self._evidence_ids = evidence_ids_by_text

    def classify(self, premise: str, hypothesis: str) -> str:
        p_entail = float(self.inner.score(premise, hypothesis))
        self.pairs.append(
            {
                "evidence_id": self._evidence_ids.get(premise, ""),
                "claim_text": hypothesis,
                "p_entail": p_entail,
                "entity_consistent": self.entity_checker.check(hypothesis, premise).consistent,
            }
        )
        return "entailment" if p_entail >= self.threshold else "neutral"


class _RecordingDraft:
    def __init__(self, inner: DraftAnswerGenerator) -> None:
        self.inner = inner
        self.last: Any = None

    def generate(self, query: Any, checklist: Any, selected: Any) -> Any:
        self.last = self.inner.generate(query, checklist, selected)
        return self.last


class _RecordingVerifier:
    def __init__(self, inner: Verifier) -> None:
        self.inner = inner
        self.last: Any = None

    def verify(self, draft: Any, selected: Any, checklist: Any) -> Any:
        self.last = self.inner.verify(draft, selected, checklist)
        return self.last


class _RecordingRepairer:
    """The repaired result carries exactly the DRAFT-ORIGIN citations; anything in
    the final result beyond these was added by recheck (Task 2's partition)."""

    def __init__(self, inner: AnswerRepairer) -> None:
        self.inner = inner
        self.last: Any = None

    def repair(self, draft: Any, report: Any, selected: Any) -> Any:
        self.last = self.inner.repair(draft, report, selected)
        return self.last


class _RecordingRechecker:
    def __init__(self, inner: EvidenceRechecker) -> None:
        self.inner = inner
        self.results: list[Any] = []

    def reset(self) -> None:
        self.results = []

    def recheck(self, coverage: Any, checklist: Any, selected: Any) -> Any:
        result = self.inner.recheck(coverage, checklist, selected)
        self.results.append(result)
        return result


# --------------------------------------------------------------------------
# generation with instrumentation
# --------------------------------------------------------------------------


def run_instrumented(cases: list[Any], llm: GraniteLLMClient, nli_inner: Any) -> list[dict[str, Any]]:
    entity_sink: list[EntityMismatchRecord] = []
    nli = _RecordingNLI(nli_inner, threshold=getattr(nli_inner, "threshold", 0.5))
    draft_gen = _RecordingDraft(DraftAnswerGenerator(llm=llm))
    verifier = _RecordingVerifier(
        Verifier(nli, llm=llm, on_entity_mismatch=entity_sink.append)  # type: ignore[arg-type]
    )
    rechecker = _RecordingRechecker(EvidenceRechecker(llm=llm))
    repairer = _RecordingRepairer(AnswerRepairer())
    generator = VerifiedGenerator(
        draft_generator=draft_gen,
        verifier=verifier,
        evidence_rechecker=rechecker,
        repairer=repairer,
    )

    records: list[dict[str, Any]] = []
    for n, case in enumerate(cases, start=1):
        query = Query(query_id=case.query_id, text=case.question)
        checklist = QueryChecklist(
            query_id=case.query_id,
            focus=case.question,
            required_facts=case.required_facts,
            constraints=case.constraints,
        )
        nli.set_context({item.text: item.evidence_id for item in case.selected.evidence})
        nli.pairs = []
        rechecker.reset()
        draft_gen.last = verifier.last = repairer.last = None

        error = ""
        try:
            final = generator.generate(query, checklist, case.selected)
        except Exception as exc:  # noqa: BLE001 -- record, keep the sample aligned
            error = f"{type(exc).__name__}: {exc}"[:200]
            final = None

        draft = draft_gen.last
        report = verifier.last
        repaired = repairer.last

        draft_citations = list(repaired.cited_evidence_ids) if repaired is not None else []
        final_citations = list(final.cited_evidence_ids) if final is not None else []
        recheck_citations = [cid for cid in final_citations if cid not in set(draft_citations)]

        gaps: list[dict[str, Any]] = []
        if report is not None:
            recheck_by_fact = {r.required_fact: r for r in rechecker.results}
            for coverage in report.fact_coverage:
                if coverage.covered:
                    continue
                result = recheck_by_fact.get(coverage.required_fact)
                gaps.append(
                    {
                        "required_fact": coverage.required_fact,
                        "gap_question": coverage.gap_question or "",
                        "recheck_found": bool(result.found) if result is not None else None,
                        "recheck_fragment": result.answer_fragment if result is not None else "",
                        "recheck_evidence_ids": list(result.evidence_ids) if result is not None else [],
                    }
                )

        answered = bool(final is not None and final.answer.strip())
        # abstention cause: VerifiedGenerator returns "" either because a required
        # fact stayed unfound after recheck, or because nothing trusted survived.
        unfound = [g for g in gaps if g["recheck_found"] is False]
        if answered or final is None:
            reason = ""
        elif unfound:
            reason = "missing_required_fact"
        else:
            reason = "no_trusted_content"

        records.append(
            {
                "query_id": case.query_id,
                "question": case.question,
                "error": error,
                "draft_answer": draft.answer_text if draft is not None else "",
                "draft_claims": (
                    [{"claim_id": c.claim_id, "text": c.text, "faithful": c.faithful_to_answer}
                     for c in draft.claims]
                    if draft is not None else []
                ),
                "verified_claims": (
                    [{"claim_id": v.claim_id, "status": v.status,
                      "supporting_evidence_ids": list(v.supporting_evidence_ids)}
                     for v in report.claims]
                    if report is not None else []
                ),
                "pairs": nli.pairs,
                "required_facts": list(case.required_facts),
                "gaps": gaps,
                "repaired_answer": repaired.answer if repaired is not None else "",
                "draft_origin_citations": draft_citations,
                "recheck_added_citations": recheck_citations,
                "final_answer": final.answer if final is not None else "",
                "final_citations": final_citations,
                "answered": answered,
                "abstention_reason": reason,
                "evidence": [
                    {"evidence_id": item.evidence_id, "text": item.text}
                    for item in case.selected.evidence
                ],
            }
        )
        if n % 25 == 0:
            print(f"[gen] {n}/{len(cases)}", flush=True)
    return records


# --------------------------------------------------------------------------
# MiniCheck judging (Tasks 1, 2, 5)
# --------------------------------------------------------------------------


def judge(records: list[dict[str, Any]], minicheck: Any) -> None:
    """Annotate records in place. Judge is MiniCheck only."""
    total = len(records)
    for n, record in enumerate(records, start=1):
        text_by_id = {item["evidence_id"]: item["text"] for item in record["evidence"]}

        # Task 1 -- was the flagged-missing fact ALREADY in the draft?
        draft_answer = record["draft_answer"]
        for gap in record["gaps"]:
            gap["minicheck_covered_by_draft"] = (
                bool(draft_answer.strip())
                and minicheck.classify(premise=draft_answer, hypothesis=gap["required_fact"])
                == "entailment"
            )
            gap["false_gap"] = gap["minicheck_covered_by_draft"]

        # Task 2 -- citation support split by origin
        final_answer = record["final_answer"]
        judged: list[dict[str, Any]] = []
        if final_answer.strip():
            draft_origin = set(record["draft_origin_citations"])
            for evidence_id in record["final_citations"]:
                chunk = text_by_id.get(evidence_id, "")
                judged.append(
                    {
                        "evidence_id": evidence_id,
                        "origin": "draft" if evidence_id in draft_origin else "recheck",
                        "minicheck_supported": bool(
                            chunk
                            and minicheck.classify(premise=chunk, hypothesis=final_answer)
                            == "entailment"
                        ),
                    }
                )
        record["judged_citations"] = judged

        # Task 5 -- threshold-independent sweep material: judge each (evidence,
        # claim) pair once, so any TRUE threshold is a pure re-aggregation.
        for pair in record["pairs"]:
            chunk = text_by_id.get(pair["evidence_id"], "")
            pair["minicheck_supported"] = bool(
                chunk
                and minicheck.classify(premise=chunk, hypothesis=pair["claim_text"]) == "entailment"
            )
        if n % 25 == 0:
            print(f"[judge] {n}/{total}", flush=True)


# --------------------------------------------------------------------------
# analysis
# --------------------------------------------------------------------------


@dataclass
class Analysis:
    gaps_total: int = 0
    gaps_false: int = 0
    cites_draft: int = 0
    cites_draft_supported: int = 0
    cites_recheck: int = 0
    cites_recheck_supported: int = 0
    abstentions: int = 0
    abstentions_missing_fact: int = 0
    abstentions_all_false_gaps: int = 0
    abstentions_any_false_gap: int = 0
    answered: int = 0
    cases: int = 0
    errors: int = 0
    sweep: list[dict[str, Any]] = field(default_factory=list)


def analyse(records: list[dict[str, Any]]) -> Analysis:
    a = Analysis()
    for record in records:
        a.cases += 1
        if record["error"]:
            a.errors += 1
        if record["answered"]:
            a.answered += 1

        for gap in record["gaps"]:
            a.gaps_total += 1
            if gap.get("false_gap"):
                a.gaps_false += 1

        for citation in record.get("judged_citations", []):
            if citation["origin"] == "draft":
                a.cites_draft += 1
                a.cites_draft_supported += int(citation["minicheck_supported"])
            else:
                a.cites_recheck += 1
                a.cites_recheck_supported += int(citation["minicheck_supported"])

        if not record["answered"] and not record["error"]:
            a.abstentions += 1
            if record["abstention_reason"] == "missing_required_fact":
                a.abstentions_missing_fact += 1
                triggering = [g for g in record["gaps"] if g["recheck_found"] is False]
                if triggering and all(g.get("false_gap") for g in triggering):
                    a.abstentions_all_false_gaps += 1
                if any(g.get("false_gap") for g in triggering):
                    a.abstentions_any_false_gap += 1

    a.sweep = sweep_verify_only(records)
    return a


def sweep_verify_only(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Task 5 -- verify-only coverage vs citation precision across TRUE thresholds.

    A claim is supported at threshold t iff some evidence has p_entail >= t AND
    passes the entity check; the answer is non-empty iff any claim is supported;
    citations are those supporting evidence, scored by the MiniCheck verdict
    already attached to each pair. Pure re-aggregation of recorded values.
    """
    rows: list[dict[str, Any]] = []
    for threshold in SWEEP_THRESHOLDS:
        answered = 0
        supported_cites = 0
        total_cites = 0
        scored_cases = 0
        for record in records:
            if record["error"]:
                continue
            scored_cases += 1
            faithful = {c["claim_id"] for c in record["draft_claims"] if c["faithful"]}
            claim_texts = {c["text"] for c in record["draft_claims"] if c["claim_id"] in faithful}
            cited: dict[str, bool] = {}
            for pair in record["pairs"]:
                if pair["claim_text"] not in claim_texts:
                    continue
                if pair["p_entail"] >= threshold and pair["entity_consistent"]:
                    cited[pair["evidence_id"]] = bool(pair["minicheck_supported"])
            if cited:
                answered += 1
                total_cites += len(cited)
                supported_cites += sum(1 for ok in cited.values() if ok)
        rows.append(
            {
                "threshold": threshold,
                "coverage": answered / scored_cases if scored_cases else None,
                "citation_precision": supported_cites / total_cites if total_cites else None,
                "cited_pairs": total_cites,
            }
        )
    return rows


def _rate(numerator: int, denominator: int) -> str:
    return f"{numerator}/{denominator} ({numerator / denominator:.3f})" if denominator else "0/0 (n/a)"


def render(a: Analysis, args: argparse.Namespace) -> str:
    lines: list[str] = []
    lines.append("# G3 follow-up — diagnosing the completeness loop")
    lines.append("")
    lines.append(
        "Re-analysis of the `verified-full` arm with per-gap / citation-origin / "
        "abstention-reason instrumentation. Judge: **MiniCheck** (never TRUE). "
        f"Seed {args.seed}, {a.cases} cases, {a.errors} chain errors."
    )
    lines.append("")
    lines.append("## Task 1 — false-gap rate on real data")
    lines.append("")
    lines.append(
        f"- gaps raised by the completeness checker: **{a.gaps_total}**\n"
        f"- of those, MiniCheck says the draft ALREADY stated the fact "
        f"(**false gap**): **{_rate(a.gaps_false, a.gaps_total)}**"
    )
    lines.append("")
    lines.append(
        "G2 measured B4 own-fact coverage 0.786 on synthetic gold statements, i.e. an "
        "implied ~0.214 false-gap rate. This is the first measurement on real data."
    )
    lines.append("")
    lines.append(
        "**Limitation (stated up front):** using a model to judge whether a gap was "
        "false is a proxy, not ground truth — MiniCheck's own recall is 0.620, so it "
        "misses genuine support and this number is *provisional*. Task 4's human "
        "packet is the ground truth for it."
    )
    lines.append("")
    lines.append("## Task 2 — citation support split by origin")
    lines.append("")
    lines.append("| citation origin | MiniCheck-supported |")
    lines.append("|---|---|")
    lines.append(f"| draft claims that survived verification | {_rate(a.cites_draft_supported, a.cites_draft)} |")
    lines.append(f"| fragments added by recheck | {_rate(a.cites_recheck_supported, a.cites_recheck)} |")
    lines.append("")
    lines.append("## Task 3 — abstentions attributable to false gaps")
    lines.append("")
    lines.append(
        f"- abstentions (non-error): **{a.abstentions}** of {a.cases} cases\n"
        f"- abstained via the completeness path (missing required fact): "
        f"**{_rate(a.abstentions_missing_fact, a.abstentions)}**\n"
        f"- of those, EVERY triggering gap was false: "
        f"**{_rate(a.abstentions_all_false_gaps, a.abstentions_missing_fact)}**\n"
        f"- of those, at least one triggering gap was false: "
        f"**{_rate(a.abstentions_any_false_gap, a.abstentions_missing_fact)}**"
    )
    lines.append("")
    lines.append("## Task 5 — verify-only coverage/precision curve (TRUE threshold sweep)")
    lines.append("")
    lines.append(
        "Pure re-aggregation of persisted per-pair TRUE probabilities; each (evidence, "
        "claim) pair carries a MiniCheck verdict, so no threshold needs a re-judge."
    )
    lines.append("")
    lines.append("| TRUE threshold | coverage | citation precision | cited pairs |")
    lines.append("|---|---|---|---|")
    for row in a.sweep:
        coverage = f"{row['coverage']:.3f}" if row["coverage"] is not None else "n/a"
        precision = f"{row['citation_precision']:.3f}" if row["citation_precision"] is not None else "n/a"
        lines.append(f"| {row['threshold']:.2f} | {coverage} | {precision} | {row['cited_pairs']} |")
    lines.append("")
    return "\n".join(lines)


def _free() -> None:
    gc.collect()
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=400)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    print("[data] loading ALCE/ASQA", flush=True)
    cases = g3.build_cases(g3.vt.ensure_asqa(), args.limit, random.Random(args.seed), top_k=args.top_k)
    print(f"[data] built {len(cases)} cases (identical construction to G3)", flush=True)

    llm = GraniteLLMClient()
    nli_inner = build_nli_model("true")
    print("[gen] instrumented verified-full (Granite + TRUE)", flush=True)
    records = run_instrumented(cases, llm, nli_inner)
    del nli_inner
    _free()

    print("[judge] loading MiniCheck", flush=True)
    minicheck = build_nli_model("minicheck")
    judge(records, minicheck)

    args.dump.parent.mkdir(parents=True, exist_ok=True)
    with args.dump.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    analysis = analyse(records)
    report = render(analysis, args)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report + "\n", encoding="utf-8")
    print("\n" + report, flush=True)
    print(
        f"[sanity] answered {analysis.answered}/{analysis.cases} "
        f"(G3 verified-full was 125/378 answered; greedy decoding should reproduce it)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
