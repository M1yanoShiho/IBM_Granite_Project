"""G5 -- verify-and-annotate against the delete filter and the baseline.

Three arms on the same calibration queries, same Granite, same selected evidence:

  baseline        generation-time citation (both conventions scored downstream)
  verify-only     draft -> verify -> repair, i.e. the published DELETE method
  verify-annotate the redesign: keep unverified claims, labelled

`verify-only` is kept because it is what isolates the delete-vs-annotate change;
without it any coverage recovery could not be attributed.

Emits per-arm answers plus the routing statistics the redesign is judged on:
how many model-declared citations survive verification, and how often the
mandatory fallback scan rescued a claim whose declared citation was wrong.
Scoring is done separately by the ALCE sentence-level scorer with MiniCheck --
TRUE is the production verifier and never judges.

HOLD-OUT: ALCE/ASQA only.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import g3_baseline_comparison as g3  # noqa: E402, I001  identical case construction
from evidence_rag.contracts.models import Query, QueryChecklist  # noqa: E402
from evidence_rag.generator.granite import GraniteGenerator, GraniteLLMClient  # noqa: E402
from evidence_rag.generator.nli import build_nli_model  # noqa: E402
from evidence_rag.generator.repair import AnswerRepairer  # noqa: E402
from evidence_rag.generator.verified import VerifiedGenerator  # noqa: E402
from evidence_rag.generator.verify_annotate import (  # noqa: E402
    VerifyAnnotateGenerator,
    is_unverified_annotation,
)
ARMS = ("baseline", "verify-only", "verify-annotate-capped", "verify-annotate-open")


class RecordingRepairer:
    """Captures verify-only's per-claim sentence -> citation mapping.

    Without it verify-only can only be scored under the flat-list convention,
    where every sentence carries all of the answer's citations. That inflates the
    per-sentence citation count (1.92 against verify-annotate's 1.13) and costs
    precision through ALCE's redundancy ablation, so the arms would not be scored
    under comparable conventions. Script-side wrapper: no src change.
    """

    def __init__(self, inner: AnswerRepairer) -> None:
        self.inner = inner
        self.last: list[dict[str, Any]] = []

    def repair(self, draft: Any, report: Any, selected: Any) -> Any:
        result = self.inner.repair(draft, report, selected)
        verifications = {item.claim_id: item for item in report.claims}
        self.last = []
        for claim in draft.claims:
            verification = verifications.get(claim.claim_id)
            if verification is None or verification.status != "supported":
                continue
            fragment = " ".join(draft.answer_text[claim.span.start : claim.span.end].split())
            if fragment:
                self.last.append(
                    {
                        "sentence": fragment,
                        "citation": (
                            verification.supporting_evidence_ids[0]
                            if verification.supporting_evidence_ids
                            else None
                        ),
                        "claim_id": claim.claim_id,
                        "outcome": "verified",
                    }
                )
        return result


def _empty_checklist(query_id: str, question: str) -> QueryChecklist:
    """Completeness is retired, so the checklist is inert in every arm. It stays in
    the signature because the contract is unchanged."""
    return QueryChecklist(query_id=query_id, focus=question, required_facts=())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=400)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    cases = g3.build_cases(g3.vt.ensure_asqa(), args.limit, random.Random(args.seed), args.top_k)
    print(f"[data] {len(cases)} cases", flush=True)

    llm = GraniteLLMClient()
    nli = build_nli_model("true")  # production verifier; never the judge
    repairer = RecordingRepairer(AnswerRepairer())
    arms: dict[str, Any] = {
        "baseline": GraniteGenerator(llm=llm),
        "verify-only": VerifiedGenerator(llm=llm, nli=nli, repairer=repairer),
        # the capped arm is what isolates the contract lift: same routing, old
        # wholesale abstention when nothing verified
        "verify-annotate-capped": VerifyAnnotateGenerator(
            llm=llm, nli=nli, abstain_when_unverified=True
        ),
        "verify-annotate-open": VerifyAnnotateGenerator(llm=llm, nli=nli),
    }

    results: dict[str, dict[str, Any]] = {name: {} for name in arms}
    errors: dict[str, int] = {name: 0 for name in arms}
    started = time.perf_counter()
    for n, case in enumerate(cases, start=1):
        query = Query(query_id=case.query_id, text=case.question)
        checklist = _empty_checklist(case.query_id, case.question)
        for name, generator in arms.items():
            try:
                generation = generator.generate(query, checklist, case.selected)
            except Exception as exc:  # noqa: BLE001 -- record, keep the sample aligned
                errors[name] += 1
                if errors[name] == 1:
                    # Full traceback, once per arm. Printing only the type name cost this
                    # project two diagnostic rounds on G6: 365 swallowed OSErrors and 310
                    # swallowed RuntimeErrors look identical in the log, and neither says
                    # where or why. One traceback per arm is four in a run -- cheap enough
                    # that there was never a reason not to.
                    print(f"[warn] {name} {case.query_id}: FIRST FAILURE, traceback:", flush=True)
                    traceback.print_exc()
                    sys.stderr.flush()
                elif errors[name] <= 3:
                    print(f"[warn] {name} {case.query_id}: {type(exc).__name__}: {exc}", flush=True)
                continue
            record: dict[str, Any] = {
                "query_id": case.query_id,
                "question": case.question,
                "answer": generation.answer,
                "cited_evidence_ids": list(generation.cited_evidence_ids),
                "evidence": [
                    {"evidence_id": item.evidence_id, "text": item.text}
                    for item in case.selected.evidence
                ],
                "gold_answers": [list(a) for a in case.gold_answers],
            }
            if name == "verify-only":
                record["routing"] = list(repairer.last)
            if name.startswith("verify-annotate"):
                # exact sentence -> verified citation, so citation precision does
                # not rest on the flat list; plus the entity-conflict drops, which
                # are now the only path that destroys content.
                record["routing"] = [
                    {
                        "claim_id": r.claim_id,
                        "outcome": r.outcome,
                        "sentence": r.sentence,
                        "citation": r.citation,
                        "claim_text": r.claim_text,
                        "declared_indices": list(r.declared_indices),
                        "declared_verified": r.declared_verified,
                        "rescued_by_scan": r.rescued_by_scan,
                        "conflict_evidence_id": r.conflict_evidence_id,
                        "conflict_detail": list(r.conflict_detail),
                    }
                    for r in generator.last_routings
                ]
            results[name][case.query_id] = record
        if n % 25 == 0:
            elapsed = time.perf_counter() - started
            print(f"[gen] {n}/{len(cases)}  ({elapsed / n:.1f}s/case)", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in ARMS:
        path = args.output_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for record in results[name].values():
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        answered = sum(1 for r in results[name].values() if r["answer"].strip())
        print(f"[arm] {name}: {answered}/{len(cases)} answered, {errors[name]} errors", flush=True)

    stats = arms["verify-annotate-open"].stats
    annotated_sentences = 0
    total_sentences = 0
    for record in results["verify-annotate-open"].values():
        for sentence in record["answer"].split(". "):
            if sentence.strip():
                total_sentences += 1
                annotated_sentences += int(is_unverified_annotation(sentence))
    routing = {
        "claims_routed": stats.claims,
        "verified": stats.verified,
        "unverified_annotated": stats.unverified,
        "dropped_entity_conflict": stats.dropped_entity_conflict,
        "claims_with_a_declared_citation": stats.declared_total,
        "declared_citation_verified": stats.declared_verified,
        "rescued_by_fallback_scan": stats.rescued_by_scan,
        "annotated_sentences": annotated_sentences,
        "kept_sentences": total_sentences,
        # how many annotations actually REACH an answer -- the number the contract
        # lift exists to move (it was 15 of 82 under the cap)
        "annotated_claims_reaching_an_answer": sum(
            1
            for r in results["verify-annotate-open"].values()
            if r["answer"].strip()
            for x in r.get("routing", [])
            if x["outcome"] == "unverified"
        ),
        "annotated_claims_reaching_an_answer_capped": sum(
            1
            for r in results["verify-annotate-capped"].values()
            if r["answer"].strip()
            for x in r.get("routing", [])
            if x["outcome"] == "unverified"
        ),
        "errors": errors,
    }
    (args.output_dir / "routing-stats.json").write_text(
        json.dumps(routing, indent=2), encoding="utf-8"
    )
    print("\n== routing ==")
    print(json.dumps(routing, indent=2))
    if stats.declared_total:
        print(
            f"\ndeclared citations surviving verification: "
            f"{stats.declared_verified}/{stats.declared_total} "
            f"({stats.declared_verified / stats.declared_total:.3f})"
        )
        print(
            f"claims rescued by the fallback scan: {stats.rescued_by_scan}/{stats.declared_total} "
            f"({stats.rescued_by_scan / stats.declared_total:.3f})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
