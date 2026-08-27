#!/usr/bin/env python3
"""Score one frozen Experiment 05 dataset-arm bundle in Goal 5."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from evidence_rag.evaluation.experiment05_data import (  # noqa: E402
    read_runtime_bundle,
    validate_sidecar_record,
)
from evidence_rag.evaluation.experiment05_generation import SystemOutput  # noqa: E402
from evidence_rag.evaluation.experiment05_io import (  # noqa: E402
    append_canonical_jsonl,
    read_jsonl,
)
from evidence_rag.evaluation.experiment05_scorer import (  # noqa: E402
    NLIEntailmentJudge,
    ScorerClaimExtractor,
    aggregate_query_scores,
    score_query,
)
from evidence_rag.generator.granite import (  # noqa: E402
    GraniteGenerationConfig,
    GraniteLLMClient,
)
from evidence_rag.generator.nli import MiniCheckNLIModel  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--generation", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--claim-traces", type=Path, required=True)
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--granite-snapshot", type=Path, required=True)
    parser.add_argument("--minicheck-snapshot", type=Path, required=True)
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()

    runtime = list(read_runtime_bundle(args.runtime, dataset=args.dataset))
    sidecars = read_jsonl(args.sidecar)
    for row in sidecars:
        validate_sidecar_record(row, dataset=args.dataset)
    outputs = [SystemOutput.model_validate(row) for row in read_jsonl(args.generation)]
    expected_ids = [str(row["query_id"]) for row in runtime]
    if (
        len(expected_ids) != 400
        or [str(row["query_id"]) for row in sidecars] != expected_ids
        or [row.query_id for row in outputs] != expected_ids
        or any(row.dataset != args.dataset or row.arm_id != args.arm for row in outputs)
    ):
        raise ValueError("runtime, sidecar, and generation identities differ")

    existing_scores = [] if not args.scores.exists() else read_jsonl(args.scores)
    existing_traces = [] if not args.claim_traces.exists() else read_jsonl(args.claim_traces)
    if len(existing_scores) != len(existing_traces):
        raise ValueError("score and claim-trace resume counts differ")
    if [str(row["query_id"]) for row in existing_scores] != expected_ids[: len(existing_scores)]:
        raise ValueError("score output is not an ordered runtime prefix")
    if [str(row["query_id"]) for row in existing_traces] != expected_ids[: len(existing_traces)]:
        raise ValueError("claim trace is not an ordered runtime prefix")

    llm = GraniteLLMClient(
        model_id=str(args.granite_snapshot),
        config=GraniteGenerationConfig(
            # Long 256-token answers can require more than 384 tokens once each
            # claim is represented by both source_text and normalized text.  The
            # cap only prevents truncating the JSON; decoding remains greedy and
            # stops at the model's ordinary end token.
            max_new_tokens=1024,
            temperature=0.0,
            top_p=1.0,
            max_input_tokens=2304,
        ),
        device="cuda",
        dtype="bfloat16",
    )
    extractor = ScorerClaimExtractor(llm)
    judge = NLIEntailmentJudge(
        MiniCheckNLIModel(model_id=str(args.minicheck_snapshot), device="cuda")
    )
    for index in range(len(existing_scores), len(outputs)):
        output = outputs[index]
        raw_output = output.model_dump(mode="json")
        if output.abstained or output.runtime_error is not None:
            claims = ()
            claim_records = ()
        else:
            claims = extractor.extract(
                str(runtime[index]["question"]), output.answer_text
            )
            claim_records = extractor.last_records
        scored = score_query(sidecars[index], raw_output, claims, judge)
        scored["schema_version"] = "experiment05.query_score.v1"
        scored["dataset"] = args.dataset
        scored["arm_id"] = args.arm
        append_canonical_jsonl(args.scores, scored)
        append_canonical_jsonl(
            args.claim_traces,
            {
                "schema_version": "experiment05.scorer_claim_trace.v1",
                "dataset": args.dataset,
                "arm_id": args.arm,
                "query_id": output.query_id,
                "claims": [asdict(claim) for claim in claims],
                "records": [asdict(record) for record in claim_records],
            },
        )
        count = index + 1
        if count % args.log_every == 0 or count == len(outputs):
            print(
                json.dumps(
                    {
                        "dataset": args.dataset,
                        "arm": args.arm,
                        "scored": count,
                        "total": len(outputs),
                    }
                ),
                flush=True,
            )

    final_scores = read_jsonl(args.scores)
    if len(final_scores) != 400 or any(row.get("scorer_error") for row in final_scores):
        raise ValueError("scorer bundle is incomplete or contains infrastructure errors")
    aggregate = aggregate_query_scores(final_scores)
    aggregate.update(
        {
            "schema_version": "experiment05.arm_aggregate.v1",
            "dataset": args.dataset,
            "arm_id": args.arm,
        }
    )
    args.aggregate.parent.mkdir(parents=True, exist_ok=True)
    args.aggregate.write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"dataset": args.dataset, "arm": args.arm, "status": "PASS"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
