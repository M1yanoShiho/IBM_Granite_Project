"""Frozen component-ablation contracts for Experiment 04 Goal 4."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from evidence_rag.contracts.models import (
    CandidateSet,
    EvidenceCandidate,
    Query,
    SelectedEvidenceSet,
    strip_annotations,
)
from evidence_rag.evaluation.citation_metrics import CitationExample, score_citation_examples
from evidence_rag.evaluation.experiment04_goal3 import (
    CitationSentence,
    PreparedArm,
    append_canonical_jsonl,
    read_jsonl,
)
from evidence_rag.evaluation.experiment04_goal3 import (
    PreparedQuery as Goal3PreparedQuery,
)
from evidence_rag.evaluation.sealed_runtime import file_sha256, ordered_id_sha256
from evidence_rag.evaluation.system_scorer import score_bundle
from evidence_rag.selector.nli_runtime import NliRiskControlledSelector

ABLATION_ARMS = (
    "ablation_dense_retriever",
    "ablation_top10",
    "ablation_direct_generator",
)
GROUNDED_ABLATION_ARMS = ABLATION_ARMS[:2]
DIRECT_ABLATION_ARM = ABLATION_ARMS[2]
FULL_GOAL3_ARM = "ours_seed13"
PREPARED_SCHEMA_VERSION: Literal["experiment04.goal4_prepared.v1"] = (
    "experiment04.goal4_prepared.v1"
)
SYSTEM_OUTPUT_SCHEMA_VERSION: Literal["experiment04.goal4_system_output.v1"] = (
    "experiment04.goal4_system_output.v1"
)
GENERATION_MANIFEST_SCHEMA_VERSION = "experiment04.goal4_generation_manifest.v1"

NonEmpty = Annotated[str, Field(min_length=1)]


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class PreparedQuery(FrozenModel):
    schema_version: Literal["experiment04.goal4_prepared.v1"]
    dataset: NonEmpty
    query_id: NonEmpty
    arms: dict[NonEmpty, PreparedArm]

    @model_validator(mode="after")
    def exact_ablation_arm_set(self) -> PreparedQuery:
        if set(self.arms) != set(ABLATION_ARMS):
            raise ValueError("Goal 4 preparation requires exactly the three frozen ablations")
        return self


class SystemOutput(FrozenModel):
    schema_version: Literal["experiment04.goal4_system_output.v1"]
    dataset: NonEmpty
    arm_id: NonEmpty
    query_id: NonEmpty
    retrieved_unit_ids: tuple[NonEmpty, ...]
    selected_unit_ids: tuple[NonEmpty, ...]
    selected_source_ids: tuple[NonEmpty, ...]
    answer: str
    citation_indices: tuple[int, ...]
    citation_sentences: tuple[CitationSentence, ...]
    failure_stage: Literal["retrieval", "selection", "generation", "system"] | None
    error_code: str | None

    @model_validator(mode="after")
    def validate_output_contract(self) -> SystemOutput:
        if self.arm_id not in ABLATION_ARMS:
            raise ValueError("system output arm is not a Goal 4 ablation")
        if not set(self.selected_unit_ids) <= set(self.retrieved_unit_ids):
            raise ValueError("selected_unit_ids must be a subset of retrieved_unit_ids")
        if self.failure_stage is None and self.error_code is not None:
            raise ValueError("error_code requires a failure_stage")
        if self.failure_stage is not None and not self.error_code:
            raise ValueError("failure_stage requires a machine-readable error_code")
        if any(index < 1 for index in self.citation_indices):
            raise ValueError("citation indices must be positive")
        return self


def _candidate_set(prepared: Goal3PreparedQuery, arm_id: str) -> CandidateSet:
    arm = prepared.arms[arm_id]
    return CandidateSet(
        query_id=prepared.query_id,
        candidates=tuple(
            EvidenceCandidate(
                evidence_id=item.source_id,
                document_id=item.source_id,
                chunk_id="prepared",
                text=item.text,
                source_uri=(f"sealed://{prepared.dataset}/{prepared.query_id}/{item.source_id}"),
                retrieval_score=item.retrieval_score,
                retrieval_rank=item.retrieval_rank,
            )
            for item in arm.selected
        ),
    )


def prepare_query(
    goal3: Goal3PreparedQuery,
    *,
    question: str,
    selector: NliRiskControlledSelector,
) -> PreparedQuery:
    """Make three single-module substitutions from the frozen Goal 3 Full upstream."""

    query = Query(query_id=goal3.query_id, text=question)
    dense = goal3.arms["dense_rag"]
    dense_candidates = _candidate_set(goal3, "dense_rag")
    selection_status: Literal["normal", "fail_open_all"] = "normal"
    try:
        selected, _trace = selector.select_with_trace(query, dense_candidates, 10)
        selected_ids = {item.evidence_id for item in selected.items}
        dense_selected = tuple(item for item in dense.selected if item.source_id in selected_ids)
    except Exception:  # noqa: BLE001 - the frozen selector policy explicitly fails open
        selection_status = "fail_open_all"
        dense_selected = dense.selected
    dense_nli = PreparedArm(
        retrieved_source_ids=dense.retrieved_source_ids,
        retrieved_unit_ids=dense.retrieved_unit_ids,
        selected=dense_selected,
        selection_status=selection_status,
    )
    return PreparedQuery(
        schema_version=PREPARED_SCHEMA_VERSION,
        dataset=goal3.dataset,
        query_id=goal3.query_id,
        arms={
            "ablation_dense_retriever": dense_nli,
            "ablation_top10": goal3.arms["hybrid_rag"],
            "ablation_direct_generator": goal3.arms[FULL_GOAL3_ARM],
        },
    )


def validate_single_substitutions(
    prepared: PreparedQuery,
    goal3: Goal3PreparedQuery,
) -> None:
    if prepared.dataset != goal3.dataset or prepared.query_id != goal3.query_id:
        raise ValueError("Goal 4 and Goal 3 prepared identity differs")
    dense = prepared.arms["ablation_dense_retriever"]
    dense_source = goal3.arms["dense_rag"]
    if (
        dense.retrieved_source_ids != dense_source.retrieved_source_ids
        or dense.retrieved_unit_ids != dense_source.retrieved_unit_ids
        or not set(item.source_id for item in dense.selected)
        <= set(item.source_id for item in dense_source.selected)
    ):
        raise ValueError("Dense Retriever ablation changed more than frozen retrieval input")
    if prepared.arms["ablation_top10"] != goal3.arms["hybrid_rag"]:
        raise ValueError("Top-10 ablation must reuse frozen Goal 3 hybrid keep-all context")
    if prepared.arms["ablation_direct_generator"] != goal3.arms[FULL_GOAL3_ARM]:
        raise ValueError("Direct Generator ablation must reuse exact frozen Full upstream")


def selected_evidence_set(prepared: PreparedQuery, arm_id: str) -> SelectedEvidenceSet:
    arm = prepared.arms[arm_id]
    return SelectedEvidenceSet(
        query_id=prepared.query_id,
        evidence=tuple(
            EvidenceCandidate(
                evidence_id=item.source_id,
                document_id=item.source_id,
                chunk_id="prepared",
                text=item.text,
                source_uri=(f"sealed://{prepared.dataset}/{prepared.query_id}/{item.source_id}"),
                retrieval_score=item.retrieval_score,
                retrieval_rank=item.retrieval_rank,
            )
            for item in arm.selected
        ),
    )


def system_output(
    *,
    prepared: PreparedQuery,
    arm_id: str,
    selected_evidence: SelectedEvidenceSet | None = None,
    answer: str,
    citation_indices: Sequence[int],
    citation_sentences: Sequence[CitationSentence],
    failure_stage: Literal["retrieval", "selection", "generation", "system"] | None = None,
    error_code: str | None = None,
) -> SystemOutput:
    arm = prepared.arms[arm_id]
    prepared_by_source = {item.source_id: item for item in arm.selected}
    selected_source_ids = (
        tuple(item.evidence_id for item in selected_evidence.evidence)
        if selected_evidence is not None
        else tuple(prepared_by_source)
    )
    if not set(selected_source_ids) <= set(prepared_by_source):
        raise ValueError("actual generation context must be a subset of prepared selection")
    return SystemOutput(
        schema_version=SYSTEM_OUTPUT_SCHEMA_VERSION,
        dataset=prepared.dataset,
        arm_id=arm_id,
        query_id=prepared.query_id,
        retrieved_unit_ids=arm.retrieved_unit_ids,
        selected_unit_ids=tuple(
            unit_id
            for source_id in selected_source_ids
            for unit_id in prepared_by_source[source_id].unit_ids
        ),
        selected_source_ids=selected_source_ids,
        answer=answer,
        citation_indices=tuple(citation_indices),
        citation_sentences=tuple(citation_sentences),
        failure_stage=failure_stage,
        error_code=error_code,
    )


def freeze_generation_manifest(
    *,
    dataset: str,
    arm_paths: Mapping[str, Path],
    expected_query_ids: Sequence[str],
    runtime_sha256: str,
    goal3_prepared_sha256: str,
    attempt_id: str,
) -> dict[str, Any]:
    if set(arm_paths) != set(ABLATION_ARMS):
        raise ValueError("Goal 4 generation manifest requires exactly three ablation arms")
    arms: dict[str, Any] = {}
    bundle_valid = True
    for arm_id in ABLATION_ARMS:
        path = arm_paths[arm_id]
        parsed = [SystemOutput.model_validate(row) for row in read_jsonl(path)]
        observed_ids = [row.query_id for row in parsed]
        if observed_ids != list(expected_query_ids):
            raise ValueError(f"{arm_id} does not cover the exact frozen ordered IDs")
        if any(row.dataset != dataset or row.arm_id != arm_id for row in parsed):
            raise ValueError(f"{arm_id} output identity differs")
        errors = sum(row.failure_stage is not None for row in parsed)
        error_rate = errors / len(parsed)
        valid = error_rate <= 0.01
        bundle_valid &= valid
        arms[arm_id] = {
            "count": len(parsed),
            "ordered_ids_sha256": ordered_id_sha256(observed_ids),
            "file_sha256": file_sha256(path),
            "runtime_errors": errors,
            "runtime_error_rate": error_rate,
            "valid_at_1pct_guard": valid,
        }
    return {
        "schema_version": GENERATION_MANIFEST_SCHEMA_VERSION,
        "attempt_id": attempt_id,
        "dataset": dataset,
        "runtime_sha256": runtime_sha256,
        "goal3_prepared_sha256": goal3_prepared_sha256,
        "query_count": len(expected_query_ids),
        "ordered_ids_sha256": ordered_id_sha256(expected_query_ids),
        "arms": arms,
        "full_regenerated": False,
        "outputs_frozen_before_scoring": True,
        "bundle_status": "PASS" if bundle_valid else "INVALID_REQUIRES_FULL_RERUN",
    }


def score_frozen_arm(
    *,
    sidecar_records: Sequence[Mapping[str, Any]],
    outputs: Sequence[Mapping[str, Any]],
    prepared_records: Sequence[Mapping[str, Any]],
    arm_id: str,
    entails: Callable[[str, str], bool],
) -> tuple[dict[str, Any], int]:
    parsed_outputs = [SystemOutput.model_validate(row) for row in outputs]
    prepared = [PreparedQuery.model_validate(row) for row in prepared_records]
    if [row.query_id for row in parsed_outputs] != [row.query_id for row in prepared]:
        raise ValueError("prepared and frozen output ordered IDs differ")
    examples: list[CitationExample] = []
    for output, prep in zip(parsed_outputs, prepared, strict=True):
        if output.arm_id != arm_id:
            raise ValueError("frozen output contains the wrong arm")
        legal = bool(output.citation_indices) and all(
            1 <= index <= len(output.selected_source_ids) for index in output.citation_indices
        )
        if output.failure_stage is not None or not output.answer.strip() or not legal:
            continue
        selected_sources = set(output.selected_source_ids)
        documents = {
            item.source_id: item.text
            for item in prep.arms[arm_id].selected
            if item.source_id in selected_sources
        }
        examples.append(
            CitationExample(
                example_id=output.query_id,
                sentences=tuple(item.sentence for item in output.citation_sentences),
                citations=tuple(item.source_ids for item in output.citation_sentences),
                documents=documents,
            )
        )
    citation_scores: dict[str, dict[str, float | int]] = {}
    scoring_failures: set[str] = set()
    judge_calls = 0
    for example in examples:
        try:
            per_example, calls = score_citation_examples((example,), entails)
        except Exception:  # noqa: BLE001 - scorer failures stay in the denominator
            scoring_failures.add(example.example_id)
            continue
        citation_scores.update(per_example)
        judge_calls += calls
    scorer_outcomes: list[dict[str, Any]] = []
    for output in parsed_outputs:
        citation = citation_scores.get(output.query_id, {"precision": 0.0, "recall": 0.0})
        scorer_outcomes.append(
            {
                "query_id": output.query_id,
                "retrieved_unit_ids": list(output.retrieved_unit_ids),
                "selected_unit_ids": list(output.selected_unit_ids),
                "selected_source_ids": list(output.selected_source_ids),
                "answer": strip_annotations(output.answer),
                "citation_indices": list(output.citation_indices),
                "failure_stage": (
                    output.failure_stage
                    if output.failure_stage is not None
                    else "scoring"
                    if output.query_id in scoring_failures
                    else None
                ),
                "minicheck": {
                    "precision": float(citation["precision"]),
                    "recall": float(citation["recall"]),
                },
            }
        )
    return score_bundle(sidecar_records, scorer_outcomes), judge_calls


__all__ = [
    "ABLATION_ARMS",
    "DIRECT_ABLATION_ARM",
    "FULL_GOAL3_ARM",
    "GROUNDED_ABLATION_ARMS",
    "PreparedQuery",
    "SystemOutput",
    "append_canonical_jsonl",
    "freeze_generation_manifest",
    "prepare_query",
    "score_frozen_arm",
    "selected_evidence_set",
    "system_output",
    "validate_single_substitutions",
]
