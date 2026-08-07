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
ARMS = (
    "baseline",
    "verify-only",
    "verify-annotate-capped",
    "verify-annotate-open",
    "verify-annotate-nogate",
)


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


ANNOTATE_ARMS = ("verify-annotate-capped", "verify-annotate-open", "verify-annotate-nogate")
SUBJECT_ARM = "verify-annotate-nogate"
CONTROL_ARM = "verify-annotate-open"


def _sentence_counts(records: dict[str, dict[str, Any]]) -> tuple[int, int]:
    annotated = total = 0
    for record in records.values():
        for sentence in record["answer"].split(". "):
            if sentence.strip():
                total += 1
                annotated += int(is_unverified_annotation(sentence))
    return annotated, total


def _annotations_reaching_an_answer(records: dict[str, dict[str, Any]]) -> int:
    return sum(
        1
        for r in records.values()
        if r["answer"].strip()
        for x in r.get("routing", [])
        if x["outcome"] == "unverified"
    )


def build_routing_stats(
    stats_by_arm: dict[str, Any],
    results: dict[str, dict[str, Any]],
    errors: dict[str, int],
) -> dict[str, Any]:
    """Assemble the routing report.

    Split out of ``main`` so it is unit-testable. The previous round's scoring job
    died in an equivalent reporting tail on a stale arm name *after* every number
    had been computed, which is a bad place to discover a typo: the arm outputs
    survive but this file does not, and this file carries the observe-only
    measurement the round exists for.
    """
    stats = stats_by_arm[SUBJECT_ARM]
    annotated_sentences, kept_sentences = _sentence_counts(results[SUBJECT_ARM])
    control_annotated, control_kept = _sentence_counts(results[CONTROL_ARM])
    return {
        "arm": SUBJECT_ARM,
        "claims_routed": stats.claims,
        "verified": stats.verified,
        "unverified_annotated": stats.unverified,
        "dropped_entity_conflict": stats.dropped_entity_conflict,
        # The direct measurement this round exists for: with the gate observe-only,
        # what it WOULD have destroyed, and what those claims became instead.
        "gate_would_drop": stats.gate_would_drop,
        "gate_would_drop_now_cited": stats.gate_would_drop_now_cited,
        "gate_would_drop_now_annotated": stats.gate_would_drop_now_annotated,
        "gate_would_have_picked_other_evidence": stats.gate_changed_citation,
        # Self-check: on the control arm the gate IS routing, so its observe-only
        # count must equal its actual drop count. If these ever disagree, the
        # observe-only log is not recording what the gate really does.
        "control_dropped_entity_conflict": stats_by_arm[CONTROL_ARM].dropped_entity_conflict,
        "control_gate_would_drop": stats_by_arm[CONTROL_ARM].gate_would_drop,
        "claims_with_a_declared_citation": stats.declared_total,
        "declared_citation_verified": stats.declared_verified,
        "rescued_by_fallback_scan": stats.rescued_by_scan,
        "annotated_sentences": annotated_sentences,
        "kept_sentences": kept_sentences,
        "control_annotated_sentences": control_annotated,
        "control_kept_sentences": control_kept,
        # how many annotations actually REACH an answer -- the number the contract
        # lift moved from 15/82 to 76/77
        **{
            f"annotated_claims_reaching_an_answer_{arm.rsplit('-', 1)[1]}": (
                _annotations_reaching_an_answer(results[arm])
            )
            for arm in ANNOTATE_ARMS
        },
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=400)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--max-error-rate",
        type=float,
        default=0.10,
        help=(
            "Fail the run if any arm's error rate exceeds this. A previous G6 attempt "
            "swallowed 365 errors per arm, exited 0, and wrote result files from 35 "
            "empty records -- the scoring job would have produced a well-formed report "
            "full of numbers from nothing."
        ),
    )
    args = parser.parse_args()

    cases = g3.build_cases(g3.vt.ensure_asqa(), args.limit, random.Random(args.seed), args.top_k)
    print(f"[data] {len(cases)} cases", flush=True)

    llm = GraniteLLMClient()
    nli = build_nli_model("true")  # production verifier; never the judge

    # Fail fast, in-process, before 400 cases of swallowed exceptions. A previous
    # attempt spent 76 GPU-minutes discovering the verifier could not load; every
    # arm that used it reported bare `OSError` and produced nothing. Loading is not
    # enough either -- a checkpoint whose tied embeddings were dropped rather than
    # cloned loads cleanly, runs at speed, and returns plausible-looking scores --
    # so the check is a pair whose answer is known.
    from g5_preflight import CONTRADICTING, ENTAILING  # noqa: PLC0415  fail-fast, after load

    verdicts = (
        nli.classify(premise=ENTAILING[0], hypothesis=ENTAILING[1]),
        nli.classify(premise=CONTRADICTING[0], hypothesis=CONTRADICTING[1]),
    )
    print(f"[preflight] TRUE: entailing -> {verdicts[0]}, contradicting -> {verdicts[1]}", flush=True)
    if verdicts[0] != "entailment" or verdicts[1] == "entailment":
        print("[FAIL] the verifier does not answer a known pair correctly", flush=True)
        return 1

    repairer = RecordingRepairer(AnswerRepairer())
    arms: dict[str, Any] = {
        "baseline": GraniteGenerator(llm=llm),
        "verify-only": VerifiedGenerator(llm=llm, nli=nli, repairer=repairer),
        # the capped arm is what isolates the contract lift: same routing, old
        # wholesale abstention when nothing verified
        "verify-annotate-capped": VerifyAnnotateGenerator(
            llm=llm, nli=nli, abstain_when_unverified=True
        ),
        # control: G6's routing, entity gate active
        "verify-annotate-open": VerifyAnnotateGenerator(llm=llm, nli=nli),
        # subject: entailment alone decides citation, no drop path. The entity
        # check still runs and its verdict is logged, so what the gate would have
        # destroyed is measured on this arm rather than extrapolated from an audit.
        "verify-annotate-nogate": VerifyAnnotateGenerator(llm=llm, nli=nli, entity_gate=False),
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
                # The old handler printed only `type(exc).__name__`. When TRUE's
                # weights stopped loading, that turned a one-line diagnosis into a
                # forensic exercise across two jobs: every arm reported `OSError`
                # and nothing said which file was missing. Print the message always,
                # and the first traceback per arm.
                if errors[name] == 1:
                    print(f"[error] {name} {case.query_id}: first failure", flush=True)
                    traceback.print_exc()
                elif errors[name] <= 5:
                    print(
                        f"[error] {name} {case.query_id}: {type(exc).__name__}: {exc}",
                        flush=True,
                    )
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
                        # observe-only: what the entity gate would have decided,
                        # recorded on every arm so the gated arms double as a check
                        "gated_outcome": r.gated_outcome,
                        "gated_citation": r.gated_citation,
                    }
                    for r in generator.last_routings
                ]
            results[name][case.query_id] = record
        if n % 25 == 0:
            elapsed = time.perf_counter() - started
            print(f"[gen] {n}/{len(cases)}  ({elapsed / n:.1f}s/case)", flush=True)

    # Refuse to write anything from a run that mostly failed. The previous attempt
    # swallowed 365 errors per arm and still exited 0 with result files on disk;
    # the scoring job would then have produced a well-formed, fully populated
    # report out of 35 empty records. A run this broken must be loud and empty,
    # not quiet and plausible.
    broken = {
        name: count / len(cases)
        for name, count in errors.items()
        if len(cases) and count / len(cases) > args.max_error_rate
    }
    if broken:
        for name, rate in sorted(broken.items()):
            print(
                f"[FAIL] {name}: {errors[name]}/{len(cases)} errors ({rate:.3f}) "
                f"exceeds --max-error-rate {args.max_error_rate}",
                flush=True,
            )
        print("[FAIL] no result files written -- fix the cause and re-run", flush=True)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in ARMS:
        path = args.output_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for record in results[name].values():
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        answered = sum(1 for r in results[name].values() if r["answer"].strip())
        print(f"[arm] {name}: {answered}/{len(cases)} answered, {errors[name]} errors", flush=True)

    routing = build_routing_stats(
        {name: generator.stats for name, generator in arms.items() if name in ANNOTATE_ARMS},
        results,
        errors,
    )
    stats = arms[SUBJECT_ARM].stats
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
