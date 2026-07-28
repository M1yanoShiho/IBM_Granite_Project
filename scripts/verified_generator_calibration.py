"""G2 -- B4 completeness + B5 full-chain, first real-data run (calibration only).

Half 2 of the Generator-B plan, run against a real Granite generator and the
TRUE verifier promoted in G2 (docs/generator/verifier-backends.md). Two stages:

  b4  CompletenessChecker on labelled cases -- Granite ONLY. Measures coverage
      accuracy (own facts should read covered, foreign facts uncovered) and, the
      thing most likely to be wrong, gap-QUESTION quality: are the questions
      concrete and answerable from evidence, and do checklist constraints survive
      into them. Verbatim gap questions are dumped, not just counts.

  b5  VerifiedGenerator end to end (draft -> split -> verify -> repair ->
      completeness -> recheck) -- Granite AND TRUE co-resident. The chain
      re-invokes Granite (completeness, recheck) AFTER TRUE attribution, so the
      two models genuinely interleave within one query and cannot be split into
      "generate then verify" stages without reimplementing the orchestrator.
      Decision: hold both on ONE full a100 (Granite-4.1-3b ~6GB bf16 + TRUE ~21GB
      bf16 ~= 28GB, well under an a100), and PROVE it with torch peak memory
      rather than assume it. See scripts/run_verified_generator.slurm.

Nothing here mutates production code beyond what G2 already landed in nli.py /
verified.py; it imports the real components and calls them.

HOLD-OUT: HotpotQA, RGB and MuSiQue-Full are the closed final test sets and are
never loaded. Calibration data only: ALCE/ASQA. Reuses the ASQA loader and text
helpers from scripts/verifier_triage.py so the data handling matches G1.

Usage:
    PYTHONPATH=src python scripts/verified_generator_calibration.py \
        --stage all --limit 60 --seed 13 \
        --output docs/generator/verified-generator-results-hpc.md \
        --dump results/verified-generator/cases.jsonl
"""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import verifier_triage as vt  # noqa: E402, I001  ASQA loader + text helpers, shared with G1
from evidence_rag.contracts.models import (  # noqa: E402
    EvidenceCandidate,
    Query,
    QueryChecklist,
    SelectedEvidenceSet,
)
from evidence_rag.generator.attribution import EntityMismatchRecord  # noqa: E402
from evidence_rag.generator.completeness import (  # noqa: E402
    CompletenessChecker,
    fallback_gap_question,
)
from evidence_rag.generator.draft import DraftAnswerGenerator  # noqa: E402
from evidence_rag.generator.evidence_recheck import (  # noqa: E402
    EvidenceRecheckResult,
    EvidenceRechecker,
)
from evidence_rag.generator.granite import GraniteLLMClient  # noqa: E402
from evidence_rag.generator.models import (  # noqa: E402
    RequiredFactCoverage,
    VerificationReport,
)
from evidence_rag.generator.nli import NLIModel, build_nli_model  # noqa: E402
from evidence_rag.generator.repair import AnswerRepairer  # noqa: E402
from evidence_rag.generator.verified import VerifiedGenerator  # noqa: E402
from evidence_rag.generator.verifier import Verifier  # noqa: E402

YEAR_RE = re.compile(r"\b(1[89]\d\d|20\d\d)\b")
MAX_FACTS_PER_CASE = 3


# --------------------------------------------------------------------------
# case construction from ASQA (calibration only)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Case:
    """One ASQA question turned into the Generator's real inputs."""

    query_id: str
    question: str
    required_facts: tuple[str, ...]
    constraints: tuple[str, ...]
    long_answer: str
    selected: SelectedEvidenceSet
    year: str | None
    """The constraint token (a year) if one was attached, so B4 can check it
    survives into the gap question."""


def _selected_from_docs(query_id: str, docs: list[dict[str, Any]], top_k: int) -> SelectedEvidenceSet:
    evidence = tuple(
        EvidenceCandidate(
            evidence_id=f"{query_id}::d{position}",
            document_id=f"{query_id}::doc{position}",
            chunk_id=f"{query_id}::chunk{position}",
            text=doc["text"],
            source_uri=f"asqa://{query_id}/{position}",
            retrieval_score=float(len(docs) - position),
            retrieval_rank=position + 1,
        )
        for position, doc in enumerate(docs[:top_k])
        if doc.get("text")
    )
    return SelectedEvidenceSet(query_id=query_id, evidence=evidence)


def build_cases(data: list[dict[str, Any]], limit: int, rng: random.Random, top_k: int = 5) -> list[Case]:
    """Each case: question + required facts (human `knowledge` sentences the gold
    answer was written from) + top-k GTR passages as the already-selected set.

    Facts come from the human knowledge annotations rather than qa_pairs so they
    are self-contained sentences (a bare short answer like '1997' makes a useless
    required fact). A fact is kept only if the gold long answer actually states it,
    so 'own facts should read covered' is a fair expectation for B4.
    """
    order = list(range(len(data)))
    rng.shuffle(order)
    cases: list[Case] = []
    for index in order:
        if len(cases) >= limit:
            break
        sample = data[index]
        long_answer = sample.get("answer", "").strip()
        docs = sample.get("docs", [])
        question = sample.get("question") or ""
        if not question:
            qa_pairs = sample.get("qa_pairs", [])
            question = qa_pairs[0].get("question", "") if qa_pairs else ""
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
            if content in seen:
                continue
            # the gold answer must actually state it, else 'covered' is unfair
            if vt.coverage(content, long_answer) < 0.6:
                continue
            seen.add(content)
            facts.append(content)
            if len(facts) >= MAX_FACTS_PER_CASE:
                break
        if not facts:
            continue

        year_match = YEAR_RE.search(question) or YEAR_RE.search(facts[0])
        year = year_match.group(1) if year_match else None
        constraints = (f"as of {year}",) if year else ()

        cases.append(
            Case(
                query_id=str(sample.get("sample_id", f"asqa-{index}")),
                question=question,
                required_facts=tuple(facts),
                constraints=constraints,
                long_answer=long_answer,
                selected=_selected_from_docs(str(sample.get("sample_id", f"asqa-{index}")), docs, top_k),
                year=year,
            )
        )
    return cases


def checklist_for(case: Case, required_facts: tuple[str, ...]) -> QueryChecklist:
    return QueryChecklist(
        query_id=case.query_id,
        focus=case.question,
        required_facts=required_facts,
        constraints=case.constraints,
    )


# --------------------------------------------------------------------------
# B4 -- completeness on labelled cases (Granite only)
# --------------------------------------------------------------------------


@dataclass
class B4Result:
    own_facts: int = 0
    own_covered: int = 0
    foreign_facts: int = 0
    foreign_uncovered: int = 0
    constrained_gaps: int = 0
    constrained_gaps_with_year: int = 0
    generic_gaps: int = 0
    gap_examples: list[dict[str, str]] = field(default_factory=list)


def _is_generic_gap(question: str, fact: str) -> bool:
    """A gap question earns its keep only if it names something specific from the
    fact. The fallback template and pure restatements do not."""
    if question.strip() == fallback_gap_question(fact).strip():
        return True
    fact_tokens = set(vt.content_tokens(fact))
    question_tokens = set(vt.content_tokens(question))
    shared = fact_tokens & question_tokens
    # concrete if it borrows a content word from the fact that is not the whole fact
    return len(shared) == 0


def run_b4(cases: list[Case], checker: CompletenessChecker, rng: random.Random) -> B4Result:
    result = B4Result()
    for case in cases:
        # 1. own facts against the gold answer -> expect covered
        own = checker.check(case.long_answer, checklist_for(case, case.required_facts))
        for coverage_item in own:
            result.own_facts += 1
            if coverage_item.covered:
                result.own_covered += 1

        # 2. a foreign fact against the same answer -> expect uncovered, and the
        #    gap question is the interesting artefact
        others = [c for c in cases if c.query_id != case.query_id and c.required_facts]
        if not others:
            continue
        foreign_case = rng.choice(others)
        foreign_fact = rng.choice(foreign_case.required_facts)
        # borrow this case's own constraint so we can test constraint survival
        foreign_checklist = QueryChecklist(
            query_id=case.query_id,
            focus=case.question,
            required_facts=(foreign_fact,),
            constraints=case.constraints,
        )
        foreign = checker.check(case.long_answer, foreign_checklist)
        for coverage_item in foreign:
            result.foreign_facts += 1
            if not coverage_item.covered:
                result.foreign_uncovered += 1
            gap = coverage_item.gap_question or ""
            if gap:
                if _is_generic_gap(gap, foreign_fact):
                    result.generic_gaps += 1
                if case.year is not None:
                    result.constrained_gaps += 1
                    if case.year in gap:
                        result.constrained_gaps_with_year += 1
                if len(result.gap_examples) < 20:
                    result.gap_examples.append(
                        {
                            "required_fact": foreign_fact,
                            "constraints": "; ".join(case.constraints) or "none",
                            "covered": str(coverage_item.covered),
                            "gap_question": gap,
                        }
                    )
    return result


# --------------------------------------------------------------------------
# B5 -- full chain end to end (Granite + TRUE co-resident)
# --------------------------------------------------------------------------


class _RecordingDraft:
    """Wraps DraftAnswerGenerator to keep the last draft (its claim count is the
    denominator faithful claims are measured against)."""

    def __init__(self, inner: DraftAnswerGenerator) -> None:
        self.inner = inner
        self.last_draft: Any = None

    def generate(self, query: Any, checklist: Any, selected: Any) -> Any:
        draft = self.inner.generate(query, checklist, selected)
        self.last_draft = draft
        return draft


class _RecordingVerifier:
    """Wraps the real Verifier to keep the last report/draft for diagnostics."""

    def __init__(self, inner: Verifier) -> None:
        self.inner = inner
        self.last_report: VerificationReport | None = None

    def verify(self, draft: Any, selected: Any, checklist: Any) -> VerificationReport:
        report = self.inner.verify(draft, selected, checklist)
        self.last_report = report
        return report


class _RecordingRechecker:
    def __init__(self, inner: EvidenceRechecker) -> None:
        self.inner = inner
        self.found = 0
        self.not_found = 0

    def reset(self) -> None:
        self.found = 0
        self.not_found = 0

    def recheck(self, coverage: RequiredFactCoverage, checklist: Any, selected: Any) -> EvidenceRecheckResult:
        result = self.inner.recheck(coverage, checklist, selected)
        if result.found:
            self.found += 1
        else:
            self.not_found += 1
        return result


def _is_trusted(verification: Any) -> bool:
    return (
        verification.status == "supported"
        and verification.entity_consistent
        and not verification.contradicted
    )


@dataclass
class CaseStats:
    draft_claims: int
    faithful_claims: int
    supported: int
    unsupported: int
    contradicted: int
    entity_mismatch: int
    gaps_found: int
    gaps_patched: int
    repair_fired: bool
    abstained: bool
    contract_ok: bool


def summarize_case(
    draft: Any,
    report: Any,
    generation: Any,
    entity_mismatch: int,
    gaps_patched: int,
) -> CaseStats:
    """Pure per-case bookkeeping (no model calls), unit-tested in tests/generator.

    ``report.claims`` are the faithful claims only (``verify_claims`` filters);
    ``draft.claims`` is the full set A2 produced. A claim is dropped by repair
    when it is not trusted (unsupported, entity-inconsistent, or contradicted).
    """
    supported = sum(1 for c in report.claims if c.status == "supported")
    contradicted = sum(1 for c in report.claims if c.contradicted)
    dropped = sum(1 for c in report.claims if not _is_trusted(c))
    uncovered = sum(1 for fc in report.fact_coverage if not fc.covered)
    answered = bool(generation.answer.strip())
    return CaseStats(
        draft_claims=len(draft.claims),
        faithful_claims=len(report.claims),
        supported=supported,
        unsupported=len(report.claims) - supported,
        contradicted=contradicted,
        entity_mismatch=entity_mismatch,
        gaps_found=uncovered,
        gaps_patched=gaps_patched,
        repair_fired=dropped > 0,
        abstained=not answered,
        contract_ok=answered == bool(generation.cited_evidence_ids),
    )


@dataclass
class B5Result:
    attempted: int = 0
    cases: int = 0
    chain_errors: int = 0
    total_claims: int = 0
    faithful_claims: int = 0
    supported_claims: int = 0
    unsupported_claims: int = 0
    contradicted_claims: int = 0
    entity_mismatches: int = 0
    gaps_found: int = 0
    gaps_patched: int = 0
    repair_fired: int = 0
    abstentions: int = 0
    contract_ok: int = 0
    ms_total: float = 0.0
    per_case: list[dict[str, Any]] = field(default_factory=list)
    error_examples: list[dict[str, str]] = field(default_factory=list)


def run_b5(
    cases: list[Case],
    llm: GraniteLLMClient,
    nli: NLIModel,
) -> B5Result:
    result = B5Result()
    entity_sink: list[EntityMismatchRecord] = []
    draft_generator = _RecordingDraft(DraftAnswerGenerator(llm=llm))
    verifier = _RecordingVerifier(
        Verifier(nli, llm=llm, on_entity_mismatch=entity_sink.append)
    )
    rechecker = _RecordingRechecker(EvidenceRechecker(llm=llm))
    generator = VerifiedGenerator(
        draft_generator=draft_generator,
        verifier=verifier,
        evidence_rechecker=rechecker,
        repairer=AnswerRepairer(),
    )

    peak_reported = False
    for case in cases:
        query = Query(query_id=case.query_id, text=case.question)
        checklist = checklist_for(case, case.required_facts)
        entity_before = len(entity_sink)
        rechecker.reset()
        result.attempted += 1

        start = time.perf_counter()
        try:
            generation = generator.generate(query, checklist, case.selected)
        except Exception as exc:  # noqa: BLE001 -- first real run: record, don't abort
            result.chain_errors += 1
            if len(result.error_examples) < 20:
                result.error_examples.append(
                    {
                        "query_id": case.query_id,
                        "error": f"{type(exc).__name__}: {exc}"[:300],
                    }
                )
            if not peak_reported:
                _report_peak_memory("after first B5 query (Granite + TRUE resident)")
                peak_reported = True
            continue
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        report = verifier.last_report
        draft = draft_generator.last_draft
        assert report is not None and draft is not None
        stats = summarize_case(
            draft,
            report,
            generation,
            entity_mismatch=len(entity_sink) - entity_before,
            gaps_patched=rechecker.found,
        )

        result.cases += 1
        result.total_claims += stats.draft_claims
        result.faithful_claims += stats.faithful_claims
        result.supported_claims += stats.supported
        result.unsupported_claims += stats.unsupported
        result.contradicted_claims += stats.contradicted
        result.entity_mismatches += stats.entity_mismatch
        result.gaps_found += stats.gaps_found
        result.gaps_patched += stats.gaps_patched
        result.repair_fired += 1 if stats.repair_fired else 0
        result.abstentions += 1 if stats.abstained else 0
        result.contract_ok += 1 if stats.contract_ok else 0
        result.ms_total += elapsed_ms

        if len(result.per_case) < 40:
            result.per_case.append(
                {
                    "query_id": case.query_id,
                    "claims": stats.faithful_claims,
                    "supported": stats.supported,
                    "contradicted": stats.contradicted,
                    "entity_mismatch": stats.entity_mismatch,
                    "gaps_found": stats.gaps_found,
                    "gaps_patched": stats.gaps_patched,
                    "repair_fired": stats.repair_fired,
                    "abstained": stats.abstained,
                    "answer": generation.answer[:280],
                    "citations": list(generation.cited_evidence_ids),
                    "ms": round(elapsed_ms, 1),
                }
            )

        if not peak_reported:
            _report_peak_memory("after first B5 query (Granite + TRUE resident)")
            peak_reported = True

    return result


# --------------------------------------------------------------------------
# GPU memory proof (Task 2)
# --------------------------------------------------------------------------


def _report_peak_memory(label: str) -> None:
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return
    if not torch.cuda.is_available():
        print(f"[mem] {label}: CUDA not available", flush=True)
        return
    allocated = torch.cuda.max_memory_allocated() / (1024**3)
    reserved = torch.cuda.max_memory_reserved() / (1024**3)
    total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(
        f"[mem] {label}: peak allocated {allocated:.1f}GB / reserved {reserved:.1f}GB "
        f"of {total:.1f}GB on {torch.cuda.get_device_name(0)}",
        flush=True,
    )


def _free_gpu() -> None:
    gc.collect()
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def _pct(numerator: int, denominator: int) -> str:
    return f"{numerator}/{denominator} ({numerator / denominator:.3f})" if denominator else "0/0 (n/a)"


def render_report(args: argparse.Namespace, b4: B4Result | None, b5: B5Result | None) -> str:
    lines: list[str] = []
    lines.append("# Verified Generator calibration (Generator Part B, Half 2 -- G2)")
    lines.append("")
    lines.append(
        "First real-data run of B4 completeness and the B5 full chain against a real "
        "Granite generator and the TRUE verifier (docs/generator/verifier-backends.md). "
        "Measurement only; no thresholds or configs committed here."
    )
    lines.append("")
    lines.append(
        "**Hold-out respected:** HotpotQA, RGB and MuSiQue-Full are never loaded. "
        "Calibration data only -- ALCE/ASQA."
    )
    lines.append("")
    lines.append(
        f"Config: seed {args.seed}, {args.limit} cases requested, top-{args.top_k} "
        f"selected evidence, NLI backend `{os.getenv('NLI_BACKEND', 'true')}`, "
        f"Granite `{os.getenv('GRANITE_MODEL_ID', 'ibm-granite/granite-4.1-3b')}`."
    )
    lines.append("")

    if b4 is not None:
        lines.append("## B4 -- completeness (Granite only)")
        lines.append("")
        lines.append("| metric | value | reading |")
        lines.append("|---|---|---|")
        lines.append(
            f"| own facts read as covered | {_pct(b4.own_covered, b4.own_facts)} | "
            "gold answer states them, so higher is better (coverage recall) |"
        )
        lines.append(
            f"| foreign facts read as uncovered | {_pct(b4.foreign_uncovered, b4.foreign_facts)} | "
            "answer does NOT state them, so higher is better (coverage precision) |"
        )
        lines.append(
            f"| gap questions that are generic | {_pct(b4.generic_gaps, b4.foreign_facts)} | "
            "lower is better -- generic questions make evidence_recheck useless |"
        )
        lines.append(
            f"| constraint (year) survives into gap question | "
            f"{_pct(b4.constrained_gaps_with_year, b4.constrained_gaps)} | "
            "of gaps whose case carried a year constraint |"
        )
        lines.append("")
        lines.append("### Verbatim gap questions (foreign-fact trials)")
        lines.append("")
        for example in b4.gap_examples:
            lines.append(
                f"- fact `{example['required_fact']}` "
                f"(constraints: {example['constraints']}; covered={example['covered']})"
            )
            lines.append(f"  -> `{example['gap_question']}`")
        lines.append("")

    if b5 is not None:
        lines.append("## B5 -- full chain (Granite + TRUE co-resident)")
        lines.append("")
        lines.append(
            f"Attempted {b5.attempted} cases; {b5.cases} completed the chain, "
            f"{b5.chain_errors} raised (see chain-error notes below). Rates below are "
            "over the completed cases."
        )
        lines.append("")
        lines.append("| diagnostic | value |")
        lines.append("|---|---|")
        lines.append(
            f"| draft claims / faithful (verified) | {b5.total_claims} / {b5.faithful_claims} |"
        )
        lines.append(f"| supported | {_pct(b5.supported_claims, b5.faithful_claims)} |")
        lines.append(
            f"| unsupported (dropped by repair) | {_pct(b5.unsupported_claims, b5.faithful_claims)} |"
        )
        lines.append(
            f"| contradicted | {b5.contradicted_claims} "
            "(expected 0 -- TRUE is binary, no contradiction class) |"
        )
        lines.append(f"| entity mismatches caught | {b5.entity_mismatches} |")
        lines.append(f"| completeness gaps found | {b5.gaps_found} |")
        lines.append(f"| gaps patched by recheck | {_pct(b5.gaps_patched, b5.gaps_found)} |")
        lines.append(f"| repair changed the answer | {_pct(b5.repair_fired, b5.cases)} |")
        lines.append(f"| honest abstention (empty answer) | {_pct(b5.abstentions, b5.cases)} |")
        lines.append(f"| GenerationResult contract held | {_pct(b5.contract_ok, b5.cases)} |")
        lines.append(
            f"| ms/case | {b5.ms_total / b5.cases:.0f} |" if b5.cases else "| ms/case | n/a |"
        )
        lines.append("")
        if b5.error_examples:
            lines.append("### Chain-error notes (first real run)")
            lines.append("")
            for example in b5.error_examples:
                lines.append(f"- `{example['query_id']}`: {example['error']}")
            lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("b4", "b5", "all"), default="all")
    parser.add_argument("--limit", type=int, default=60, help="ASQA cases to build")
    parser.add_argument("--top-k", type=int, default=5, help="selected evidence per case")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output", type=Path, required=True, help="results markdown")
    parser.add_argument("--dump", type=Path, default=None, help="per-case jsonl")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    print("[data] loading ALCE/ASQA", flush=True)
    cases = build_cases(vt.ensure_asqa(), args.limit, rng, top_k=args.top_k)
    print(f"[data] built {len(cases)} calibration cases", flush=True)
    if not cases:
        raise SystemExit("no usable ASQA cases were built")

    b4: B4Result | None = None
    b5: B5Result | None = None

    def flush_outputs() -> None:
        report = render_report(args, b4, b5)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n", encoding="utf-8")
        if args.dump is not None:
            args.dump.parent.mkdir(parents=True, exist_ok=True)
            with args.dump.open("w", encoding="utf-8") as handle:
                if b4 is not None:
                    for example in b4.gap_examples:
                        handle.write(json.dumps({"stage": "b4", **example}) + "\n")
                if b5 is not None:
                    for record in b5.per_case:
                        handle.write(json.dumps({"stage": "b5", **record}) + "\n")

    llm = GraniteLLMClient()

    if args.stage in ("b4", "all"):
        print("[b4] completeness on labelled cases", flush=True)
        b4 = run_b4(cases, CompletenessChecker(llm=llm), random.Random(args.seed))
        _free_gpu()
        flush_outputs()  # persist B4 before B5 so a B5 failure cannot lose it

    if args.stage in ("b5", "all"):
        print("[b5] loading TRUE verifier", flush=True)
        nli = build_nli_model("true")
        print("[b5] running full chain (Granite + TRUE co-resident)", flush=True)
        b5 = run_b5(cases, llm, nli)
        _report_peak_memory("after full B5 run")

    flush_outputs()
    print("\n" + render_report(args, b4, b5), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
