#!/usr/bin/env python3
"""Prepare all Experiment 05 development arms from frozen retrieval traces."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from evidence_rag.evaluation.experiment05_data import read_runtime_bundle  # noqa: E402
from evidence_rag.evaluation.experiment05_io import (  # noqa: E402
    append_canonical_jsonl,
    read_jsonl,
)
from evidence_rag.evaluation.experiment05_runtime import PreparedQuery, prepare_query  # noqa: E402
from evidence_rag.retriever.rerank import GraniteCrossEncoderReranker  # noqa: E402
from evidence_rag.selector.dual_head import load_dual_head_checkpoint  # noqa: E402
from evidence_rag.selector.nli_dual_head import load_nli_dual_head_model  # noqa: E402
from evidence_rag.selector.provence import ProvencePassagePruner  # noqa: E402
from evidence_rag.selector.threshold_only import NliThresholdOnlySelector  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reranker-snapshot", type=Path, required=True)
    parser.add_argument("--provence-snapshot", type=Path, required=True)
    parser.add_argument("--selector-snapshot", type=Path, required=True)
    parser.add_argument("--selector-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()

    runtime = list(read_runtime_bundle(args.runtime, dataset=args.dataset))
    retrieval = read_jsonl(args.retrieval)
    expected_ids = [str(item["query_id"]) for item in runtime]
    if [str(item["query_id"]) for item in retrieval] != expected_ids:
        raise ValueError("retrieval traces do not cover the frozen runtime sequence")
    existing = [] if not args.output.exists() else read_jsonl(args.output)
    for row in existing:
        PreparedQuery.model_validate(row)
    if [str(item["query_id"]) for item in existing] != expected_ids[: len(existing)]:
        raise ValueError("prepared output is not an ordered runtime prefix")
    if len(existing) == len(runtime):
        print(json.dumps({"dataset": args.dataset, "count": len(existing), "status": "PASS"}))
        return 0

    reranker = GraniteCrossEncoderReranker(
        model_id=str(args.reranker_snapshot),
        revision="d09d3d6971b689bf9c23839e45a470874d46e13a",
        device=args.device,
        local_files_only=True,
    )
    provence = ProvencePassagePruner(
        model_id=str(args.provence_snapshot),
        revision="ef49e233e3c6e50efc476c68f1390f8a63add4d4",
        threshold=0.1,
        always_select_title=True,
        reorder=False,
        local_files_only=True,
    )
    provence.model = provence.model.to(args.device)
    provence.model.eval()
    selector_model = load_nli_dual_head_model(
        str(args.selector_snapshot),
        revision="6c749ce3425cd33b46d187e45b92bbf96ee12ec7",
        identity_model_id="cross-encoder/nli-deberta-v3-base",
        local_files_only=True,
        device=args.device,
    )
    load_dual_head_checkpoint(selector_model, args.selector_checkpoint)
    selector_model.eval()
    selector = NliThresholdOnlySelector(model=selector_model)

    for index in range(len(existing), len(runtime)):
        prepared = prepare_query(
            runtime[index],
            retrieval[index],
            reranker=reranker,
            provence=provence,
            selector=selector,
        )
        append_canonical_jsonl(args.output, prepared)
        count = index + 1
        if count % args.log_every == 0 or count == len(runtime):
            print(json.dumps({"dataset": args.dataset, "prepared": count, "total": len(runtime)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
