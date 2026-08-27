from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

from evidence_rag.evaluation.experiment04_goal3 import (
    PRIMARY_ARMS,
    CitationSentence,
    PreparedArm,
    PreparedEvidence,
)
from evidence_rag.evaluation.experiment04_goal3 import (
    PreparedQuery as Goal3PreparedQuery,
)
from evidence_rag.evaluation.experiment04_goal4 import (
    ABLATION_ARMS,
    PreparedQuery,
    SystemOutput,
    append_canonical_jsonl,
    freeze_generation_manifest,
    prepare_query,
    score_frozen_arm,
    system_output,
    validate_single_substitutions,
)
from evidence_rag.evaluation.sealed_runtime import file_sha256
from evidence_rag.evaluation.system_scorer import SCORER_SCHEMA_VERSION


def _arm(prefix: str, selected: int = 2) -> PreparedArm:
    evidence = tuple(
        PreparedEvidence(
            source_id=f"{prefix}{index}",
            text=f"Synthetic evidence {prefix}{index}",
            unit_ids=(f"{prefix}{index}:u0",),
            retrieval_score=float(10 - index),
            retrieval_rank=index + 1,
        )
        for index in range(3)
    )
    return PreparedArm(
        retrieved_source_ids=tuple(item.source_id for item in evidence),
        retrieved_unit_ids=tuple(item.unit_ids[0] for item in evidence),
        selected=evidence[:selected],
    )


def _goal3(query_id: str = "q") -> Goal3PreparedQuery:
    dense = _arm("d", 3)
    hybrid = _arm("h", 3)
    full = _arm("h", 2)
    arms = {
        "dense_rag": dense,
        "hybrid_rag": hybrid,
        "granite_rerank_rag": _arm("r", 3),
        "provence_rag": _arm("h", 1),
        "ours_seed13": full,
        "ours_seed42": full,
        "ours_seed73": full,
    }
    assert set(arms) == set(PRIMARY_ARMS)
    return Goal3PreparedQuery(
        schema_version="experiment04.prepared.v1",
        dataset="hotpotqa",
        query_id=query_id,
        arms=arms,
    )


class _Selector:
    def select_with_trace(
        self, query: object, candidates: object, max_items: int
    ) -> tuple[object, object]:
        del query, max_items
        return SimpleNamespace(items=(candidates.candidates[0], candidates.candidates[2])), object()


def _goal4(query_id: str = "q") -> PreparedQuery:
    return prepare_query(
        _goal3(query_id),
        question="Synthetic question?",
        selector=_Selector(),  # type: ignore[arg-type]
    )


def test_prepare_query_changes_exactly_one_module_per_ablation() -> None:
    source = _goal3()
    prepared = prepare_query(
        source,
        question="Synthetic question?",
        selector=_Selector(),  # type: ignore[arg-type]
    )

    validate_single_substitutions(prepared, source)
    assert set(prepared.arms) == set(ABLATION_ARMS)
    assert tuple(item.source_id for item in prepared.arms["ablation_dense_retriever"].selected) == (
        "d0",
        "d2",
    )
    assert prepared.arms["ablation_top10"] == source.arms["hybrid_rag"]
    assert prepared.arms["ablation_direct_generator"] == source.arms["ours_seed13"]


def test_goal4_output_rejects_goal3_full_arm() -> None:
    try:
        SystemOutput(
            schema_version="experiment04.goal4_system_output.v1",
            dataset="hotpotqa",
            arm_id="ours_seed13",
            query_id="q",
            retrieved_unit_ids=("u",),
            selected_unit_ids=("u",),
            selected_source_ids=("s",),
            answer="",
            citation_indices=(),
            citation_sentences=(),
            failure_stage=None,
            error_code=None,
        )
    except ValueError as error:
        assert "Goal 4 ablation" in str(error)
    else:
        raise AssertionError("Goal 4 accepted a Full-generation output")


def test_two_percent_errors_invalidate_entire_goal4_dataset_bundle(tmp_path: Path) -> None:
    expected = [f"q{index:03d}" for index in range(100)]
    paths: dict[str, Path] = {}
    for arm in ABLATION_ARMS:
        path = tmp_path / f"{arm}.jsonl"
        paths[arm] = path
        for index, query_id in enumerate(expected):
            append_canonical_jsonl(
                path,
                system_output(
                    prepared=_goal4(query_id),
                    arm_id=arm,
                    answer="" if index < 2 else "Synthetic answer",
                    citation_indices=(),
                    citation_sentences=(),
                    failure_stage="generation" if index < 2 else None,
                    error_code="generation_error" if index < 2 else None,
                ),
            )
    runtime = tmp_path / "runtime.jsonl"
    runtime.write_text("synthetic\n", encoding="utf-8")

    manifest = freeze_generation_manifest(
        dataset="hotpotqa",
        arm_paths=paths,
        expected_query_ids=expected,
        runtime_sha256=file_sha256(runtime),
        goal3_prepared_sha256="0" * 64,
        attempt_id="synthetic",
    )

    assert manifest["bundle_status"] == "INVALID_REQUIRES_FULL_RERUN"
    assert manifest["full_regenerated"] is False
    assert all(entry["runtime_error_rate"] == 0.02 for entry in manifest["arms"].values())


def test_goal4_scoring_keeps_exact_common_denominator() -> None:
    prepared = _goal4()
    arm = "ablation_dense_retriever"
    selected_source = prepared.arms[arm].selected[0]
    output = system_output(
        prepared=prepared,
        arm_id=arm,
        answer="Synthetic answer",
        citation_indices=(1,),
        citation_sentences=(
            CitationSentence(
                sentence="Synthetic answer.",
                source_ids=(selected_source.source_id,),
            ),
        ),
    )
    text = selected_source.text
    sidecar = {
        "schema_version": SCORER_SCHEMA_VERSION,
        "dataset": "hotpotqa",
        "query_id": "q",
        "gold_answer_aliases": ["Synthetic answer"],
        "support_units": [
            {
                "unit_id": selected_source.unit_ids[0],
                "source_id": selected_source.source_id,
                "text": text,
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            }
        ],
        "component_id": "synthetic-component",
    }

    score, calls = score_frozen_arm(
        sidecar_records=[sidecar],
        outputs=[output.model_dump(mode="json")],
        prepared_records=[prepared.model_dump(mode="json")],
        arm_id=arm,
        entails=lambda premise, hypothesis: bool(premise and hypothesis),
    )

    assert calls == 1
    assert score["n_queries"] == 1
    assert score["aggregate"] == {
        "ret": 1.0,
        "sel": 1.0,
        "ans": 1.0,
        "cit": 1.0,
        "rar": 1.0,
    }
