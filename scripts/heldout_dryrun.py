"""Held-out plumbing dry run: do the three sets load and flow through the pipeline?

**This script computes no metric and prints no answer text.** It reports schema
conformance and record counts only. Confirming that a field exists is not
observing a result; running the scorer is, and the scorer is not imported here.

The generation stage IS exercised on a tiny slice, because a loader that produces
a valid-looking record can still break the Generator -- which is precisely the
failure this is meant to catch before the one permitted run.

    PYTHONPATH=src python scripts/heldout_dryrun.py --slice 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from heldout_data import DATASETS, load  # noqa: E402

from evidence_rag.contracts.models import (  # noqa: E402
    EvidenceCandidate,
    Query,
    QueryChecklist,
    SelectedEvidenceSet,
)

REQUIRED = ("query_id", "question", "gold_answers", "passages")


def check_schema(name: str, rows: list[dict[str, Any]]) -> list[str]:
    """Structural conformance only -- shapes and counts, never content."""
    problems: list[str] = []
    if not rows:
        return [f"{name}: loaded zero records"]
    for field in REQUIRED:
        if field not in rows[0]:
            problems.append(f"{name}: missing field {field!r}")
    ids = [r["query_id"] for r in rows]
    if len(ids) != len(set(ids)):
        problems.append(f"{name}: query_id values are not unique ({len(ids) - len(set(ids))} dupes)")
    if any(not r["question"] for r in rows):
        problems.append(f"{name}: some records have an empty question")
    if any(not r["passages"] for r in rows):
        problems.append(f"{name}: some records have no passages")
    if any(not r["gold_answers"] or not r["gold_answers"][0] for r in rows):
        problems.append(f"{name}: some records have no gold answers")
    return problems


def to_selected(row: dict[str, Any], top_k: int) -> SelectedEvidenceSet:
    """Same shape the calibration runner hands the Generator."""
    evidence = tuple(
        EvidenceCandidate(
            evidence_id=f"{row['query_id']}-p{index}",
            document_id=f"{row['query_id']}-d{index}",
            chunk_id=f"{row['query_id']}-c{index}",
            text=passage["text"],
            source_uri=f"heldout://{row['query_id']}/{index}",
            retrieval_score=1.0 / (index + 1),
            retrieval_rank=index + 1,
        )
        for index, passage in enumerate(row["passages"][:top_k])
    )
    return SelectedEvidenceSet(query_id=str(row["query_id"]), evidence=evidence)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slice", type=int, default=3, help="records per set through the pipeline")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--schema-only",
        action="store_true",
        help="skip the generation stage (no GPU / no model weights needed)",
    )
    args = parser.parse_args()

    failures: list[str] = []
    summary: list[str] = []

    for name in DATASETS:
        print(f"\n=== {name} ===", flush=True)
        try:
            rows = load(name)
        except SystemExit as exc:
            failures.append(f"{name}: load failed -- {exc}")
            print(f"  LOAD FAILED: {exc}", flush=True)
            continue
        problems = check_schema(name, rows)
        failures.extend(problems)
        passages = [len(r["passages"]) for r in rows]
        print(f"  records: {len(rows)}", flush=True)
        print(
            f"  passages per record: min={min(passages)} max={max(passages)} "
            f"median={sorted(passages)[len(passages) // 2]}",
            flush=True,
        )
        print(f"  gold answer sets per record: {len(rows[0]['gold_answers'])}", flush=True)
        print(f"  schema problems: {problems or 'none'}", flush=True)
        summary.append(f"{name}: {len(rows)} records, {len(problems)} schema problems")

        # Contract conformance of the evidence objects -- still not a result.
        try:
            for row in rows[: args.slice]:
                selected = to_selected(row, args.top_k)
                Query(query_id=str(row["query_id"]), text=row["question"])
                QueryChecklist(
                    query_id=str(row["query_id"]), focus=row["question"], required_facts=()
                )
                assert len(selected.evidence) == min(args.top_k, len(row["passages"]))
            print(f"  contract objects built for {args.slice} records: ok", flush=True)
        except Exception as exc:  # noqa: BLE001 - this is the check
            failures.append(f"{name}: contract construction failed -- {type(exc).__name__}: {exc}")
            print(f"  CONTRACT FAILED: {type(exc).__name__}: {exc}", flush=True)
            continue

        if args.schema_only:
            continue

        from evidence_rag.generator.granite import GraniteGenerator, GraniteLLMClient
        from evidence_rag.generator.nli import build_nli_model
        from evidence_rag.generator.verify_annotate import VerifyAnnotateGenerator

        global _LLM, _NLI  # noqa: PLW0603 - load the weights once across datasets
        if _LLM is None:
            print("  [models] loading Granite + TRUE", flush=True)
            _LLM = GraniteLLMClient()
            _NLI = build_nli_model("true")
        arms = {
            "baseline": GraniteGenerator(llm=_LLM),
            "verify-annotate-nogate": VerifyAnnotateGenerator(
                llm=_LLM, nli=_NLI, entity_gate=False
            ),
        }
        for arm, generator in arms.items():
            ok = 0
            for row in rows[: args.slice]:
                try:
                    result = generator.generate(
                        Query(query_id=str(row["query_id"]), text=row["question"]),
                        QueryChecklist(
                            query_id=str(row["query_id"]),
                            focus=row["question"],
                            required_facts=(),
                        ),
                        to_selected(row, args.top_k),
                    )
                    # Shape only. The answer text is NOT printed and NOT scored.
                    assert result.query_id == str(row["query_id"])
                    ok += 1
                except Exception as exc:  # noqa: BLE001 - this is the check
                    failures.append(
                        f"{name}/{arm}: generation failed -- {type(exc).__name__}: {exc}"
                    )
            print(f"  {arm}: {ok}/{args.slice} records produced a valid result object", flush=True)

    print("\n=== plumbing summary ===", flush=True)
    for line in summary:
        print(f"  {line}", flush=True)
    if failures:
        print("\nPROBLEMS:", flush=True)
        for problem in failures:
            print(f"  - {problem}", flush=True)
        return 1
    print("\nplumbing OK -- no metrics computed, no outputs inspected", flush=True)
    return 0


_LLM: Any = None
_NLI: Any = None

if __name__ == "__main__":
    raise SystemExit(main())
