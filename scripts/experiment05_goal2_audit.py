#!/usr/bin/env python3
"""Audit Experiment 05 Goal 2 without reading any formal query or gold file."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from evidence_rag.contracts.models import (  # noqa: E402
    EvidenceCandidate,
    Query,
    SelectedEvidenceSet,
)
from evidence_rag.evaluation.experiment05_data import (  # noqa: E402
    read_runtime_bundle,
    validate_sidecar_record,
)
from evidence_rag.evaluation.experiment05_generation import (  # noqa: E402
    ALL_ARMS,
    DIRECT_ARMS,
    SystemOutput,
    prompt_template,
    render_prompt,
    selected_evidence_set,
)
from evidence_rag.evaluation.experiment05_io import read_jsonl  # noqa: E402
from evidence_rag.evaluation.experiment05_runtime import (  # noqa: E402
    PreparedEvidence,
    PreparedQuery,
)


class _PromptCounter:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def count(self, prompt: str) -> int:
        if getattr(self.tokenizer, "chat_template", None):
            encoded = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
        else:
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        return int(input_ids.shape[-1])

    def text_count(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False).input_ids)


def _normalized(value: str) -> str:
    return " ".join(re.findall(r"[\w]+", value.casefold()))


def _overlaps(evidence: PreparedEvidence, provenance: dict[str, Any]) -> bool:
    if evidence.source_kind != provenance.get("source_kind") or evidence.source_id != str(
        provenance.get("source_id")
    ):
        return False
    if evidence.start_unit is None or provenance.get("start_unit") is None:
        return True
    return int(evidence.start_unit) <= int(provenance["end_unit"]) and int(
        provenance["start_unit"]
    ) <= int(evidence.end_unit)


def _supports(evidence: PreparedEvidence, fact: dict[str, Any], provenance: list[dict[str, Any]]) -> bool:
    if any(
        item.get("fact_id") == fact["fact_id"] and _overlaps(evidence, item)
        for item in provenance
    ):
        return True
    text = f" {_normalized(evidence.text)} "
    return any(
        (alias := _normalized(str(raw))) and f" {alias} " in text
        for raw in fact["aliases"]
    )


def _selector_audit(prepared: list[PreparedQuery]) -> dict[str, Any]:
    checked = 0
    fail_open = 0
    all_delete = 0
    for row in prepared:
        trace = row.selector_traces["hybrid"]
        if trace.status == "fail_open_all":
            expected: tuple[str, ...] = ()
            fail_open += 1
        else:
            expected = tuple(
                item.evidence_id
                for item in trace.decisions
                if item.harm_score is not None
                and item.protect_score is not None
                and item.harm_score >= trace.threshold
                and item.protect_score <= 1.0 - trace.threshold
            )
        if trace.dropped_evidence_ids != expected:
            raise ValueError("Selector deletion set differs from the frozen predicate")
        checked += 1
        all_delete += not trace.selected_evidence_ids
    return {
        "queries_checked": checked,
        "predicate_exact": True,
        "fail_open_queries": fail_open,
        "all_delete_queries": all_delete,
        "all_delete_rate": all_delete / checked,
    }


def _diagnostics(
    prepared: list[PreparedQuery],
    runtime: list[dict[str, Any]],
    sidecars: list[dict[str, Any]],
    counter: _PromptCounter,
) -> dict[str, Any]:
    retrieved_facts = 0
    total_facts = 0
    retrieved_units = 0
    retained_units = 0
    input_tokens = 0
    selected_tokens = 0
    presented_tokens = 0
    template = prompt_template(str(runtime[0]["dataset"]))
    for row, runtime_row, sidecar in zip(prepared, runtime, sidecars, strict=True):
        full = row.arms["ours_seed13"]
        retrieved = row.arms["hybrid_rag"].selected
        selected_by_id = {item.evidence_id: item for item in full.selected}
        for fact in sidecar["reference_fact_groups"]:
            matches = [
                item
                for item in retrieved
                if _supports(item, fact, sidecar["gold_provenance"])
            ]
            total_facts += 1
            retrieved_facts += bool(matches)
            retrieved_units += len(matches)
            retained_units += sum(
                item.evidence_id in selected_by_id
                and _supports(selected_by_id[item.evidence_id], fact, sidecar["gold_provenance"])
                for item in matches
            )
        input_tokens += sum(counter.text_count(item.text) for item in retrieved)
        selected_tokens += sum(counter.text_count(item.text) for item in full.selected)
        selected_set = selected_evidence_set(row, "ours_seed13")
        query = Query(query_id=row.query_id, text=str(runtime_row["question"]))
        prompt = render_prompt(template, query, selected_set)
        while selected_set.evidence and counter.count(prompt) > 2304:
            selected_set = selected_set.model_copy(update={"evidence": selected_set.evidence[:-1]})
            prompt = render_prompt(template, query, selected_set)
        if counter.count(prompt) > 2304:
            raise ValueError("fixed prompt exceeds the formal token budget")
        presented_tokens += sum(counter.text_count(item.text) for item in selected_set.evidence)
    return {
        "support_matching": "gold_provenance_or_normalized_alias_containment",
        "er_at_10": retrieved_facts / total_facts,
        "selr": (retrieved_units - retained_units) / retrieved_units if retrieved_units else None,
        "selection_reduction_rate": 1.0 - selected_tokens / input_tokens,
        "presented_reduction_rate": 1.0 - presented_tokens / input_tokens,
        "reference_facts": total_facts,
        "retrieved_support_units": retrieved_units,
    }


def _smoke_audit(
    smoke_root: Path,
    dataset: str,
    runtime: list[dict[str, Any]],
    counter: _PromptCounter,
) -> dict[str, Any]:
    errors = 0
    abstentions = 0
    counts: dict[str, int] = {}
    by_id = {str(item["query_id"]): item for item in runtime[:5]}
    template = prompt_template(dataset)
    for arm in ALL_ARMS:
        path = smoke_root / dataset / "generations" / f"{arm}.jsonl"
        rows = [SystemOutput.model_validate(item) for item in read_jsonl(path)]
        if len(rows) != 5 or [row.query_id for row in rows] != list(by_id):
            raise ValueError(f"{dataset}/{arm} smoke IDs or count differ")
        counts[arm] = len(rows)
        for row in rows:
            errors += row.runtime_error is not None
            abstentions += row.abstained
            selected_root = (
                smoke_root.parent
                / "sealed_evidence"
                / "smoke"
                / dataset
                / ("direct" if arm in DIRECT_ARMS else "grounded")
            )
            for record in (*row.selected_evidence_records, *row.presented_evidence_records):
                path = selected_root / record.text_sha256[:2] / f"{record.text_sha256}.txt"
                if not path.is_file() or path.read_text(encoding="utf-8") != record.text:
                    raise ValueError("sealed evidence artifact differs from output record")
                if counter.text_count(record.text) != record.token_count:
                    raise ValueError("evidence token audit differs")
            query = Query(query_id=row.query_id, text=str(by_id[row.query_id]["question"]))
            evidence_set = SelectedEvidenceSet(
                query_id=row.query_id,
                evidence=tuple(
                    EvidenceCandidate(
                        evidence_id=item.evidence_id,
                        document_id=item.evidence_id,
                        chunk_id="audit",
                        text=item.text,
                        source_uri=item.artifact_uri,
                        retrieval_score=0.0,
                        retrieval_rank=item.prompt_ordinal,
                    )
                    for item in row.presented_evidence_records
                ),
            )
            prompt = render_prompt(template, query, evidence_set)
            if hashlib.sha256(prompt.encode("utf-8")).hexdigest() != row.prompt_byte_sha256:
                raise ValueError("prompt byte hash differs")
            if counter.count(prompt) != row.prompt_token_count:
                raise ValueError("prompt token count differs")
    if errors:
        raise ValueError(f"smoke contains {errors} runtime errors")
    return {"arm_counts": counts, "total_outputs": sum(counts.values()), "runtime_errors": 0, "abstentions": abstentions}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goal1-root", type=Path, required=True)
    parser.add_argument("--goal2-root", type=Path, required=True)
    parser.add_argument("--tokenizer-snapshot", type=Path, required=True)
    parser.add_argument("--formal-output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    counter = _PromptCounter(AutoTokenizer.from_pretrained(args.tokenizer_snapshot, local_files_only=True))
    datasets: dict[str, Any] = {}
    total_selector = 0
    total_smoke = 0
    for dataset in ("kilt-nq", "kilt-tqa", "alce-asqa"):
        runtime = list(
            read_runtime_bundle(
                args.goal1_root / "bundles/runtime/development" / f"{dataset}.jsonl",
                dataset=dataset,
            )
        )
        sidecars = read_jsonl(
            args.goal1_root / "bundles/scorer_only/development" / f"{dataset}.jsonl"
        )
        for sidecar in sidecars:
            validate_sidecar_record(sidecar, dataset=dataset)
        prepared = [
            PreparedQuery.model_validate(item)
            for item in read_jsonl(
                args.goal2_root / "prepared/development" / f"{dataset}.jsonl"
            )
        ]
        if len(runtime) != 120 or len(sidecars) != 120 or len(prepared) != 120:
            raise ValueError(f"{dataset} development bundle must contain 120 rows")
        selector = _selector_audit(prepared)
        smoke = _smoke_audit(args.goal2_root / "smoke", dataset, runtime, counter)
        diagnostics = _diagnostics(prepared, runtime, sidecars, counter)
        datasets[dataset] = {
            "selector": selector,
            "smoke": smoke,
            "diagnostics": diagnostics,
        }
        total_selector += selector["queries_checked"]
        total_smoke += smoke["total_outputs"]

    formal_files = (
        [path for path in args.formal_output_root.rglob("*") if path.is_file()]
        if args.formal_output_root.exists()
        else []
    )
    if formal_files:
        raise ValueError("formal output directory is not empty before Goal 2 PASS")
    if total_selector != 360 or total_smoke != 150:
        raise ValueError("Goal 2 expected exactly 360 selector queries and 150 smoke outputs")
    report = {
        "schema_version": "experiment05.goal2_audit.v1",
        "status": "PASS",
        "selector_queries": total_selector,
        "smoke_outputs": total_smoke,
        "formal_outputs": 0,
        "datasets": datasets,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "selector_queries": 360, "smoke_outputs": 150}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
