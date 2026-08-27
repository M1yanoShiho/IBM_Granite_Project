from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

from evidence_rag.contracts.models import (
    EvidenceCandidate,
    SelectedEvidenceSet,
    SelectionItem,
    SelectionResult,
)
from evidence_rag.evaluation.experiment04_goal3 import (
    BASELINE_ARMS,
    OURS_ARMS,
    PRIMARY_ARMS,
    CitationSentence,
    PreparedArm,
    PreparedEvidence,
    PreparedQuery,
    SystemOutput,
    append_canonical_jsonl,
    citation_sentences_from_inline,
    freeze_generation_manifest,
    maximal_whole_evidence_prefix,
    ordered_prefix_count,
    paired_component_cluster_bootstrap,
    prepare_query,
    retained_unit_ids_after_pruning,
    score_frozen_arm,
    system_output,
)
from evidence_rag.evaluation.sealed_runtime import RUNTIME_SCHEMA_VERSION, file_sha256
from evidence_rag.evaluation.system_scorer import SCORER_SCHEMA_VERSION


class _Embedder:
    def embed_documents(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
        return tuple((float(index + 1), 1.0) for index, _text in enumerate(texts))

    def embed_query(self, text: str) -> tuple[float, ...]:
        assert text
        return (1.0, 1.0)


class _Reranker:
    def __init__(self) -> None:
        self.pool_sizes: list[int] = []

    def score(self, query: str, passages: tuple[str, ...]) -> tuple[float, ...]:
        assert query
        self.pool_sizes.append(len(passages))
        return tuple(float(index) for index, _passage in enumerate(passages))


class _Provence:
    def prune(self, *, question: str, title: str, text: str) -> str:
        assert question and text
        return f"{title} {text}".strip()


class _Selector:
    def select_with_trace(self, query: object, candidates: object, max_selected: int) -> object:
        assert max_selected == 10
        items = candidates.candidates[:8]
        return (
            SelectionResult(
                query_id=query.query_id,
                items=tuple(
                    SelectionItem(
                        evidence_id=item.evidence_id,
                        selection_score=1.0,
                        selection_rank=rank,
                    )
                    for rank, item in enumerate(items, start=1)
                ),
            ),
            SimpleNamespace(),
        )


def _runtime(query_id: str = "q") -> dict[str, object]:
    return {
        "schema_version": RUNTIME_SCHEMA_VERSION,
        "dataset": "hotpotqa",
        "query_id": query_id,
        "question": "Which synthetic source is relevant?",
        "candidates": [
            {
                "source_id": f"p{index:03d}",
                "title": f"Title {index}",
                "text": f"Synthetic sentence {index}.",
                "units": [
                    {
                        "unit_id": f"p{index:03d}:u000",
                        "text": f"Synthetic sentence {index}.",
                    }
                ],
            }
            for index in range(45)
        ],
    }


def _prepared(query_id: str = "q") -> PreparedQuery:
    evidence = PreparedEvidence(
        source_id="p000",
        text="Synthetic answer evidence.",
        unit_ids=("p000:u000",),
        retrieval_score=1.0,
        retrieval_rank=1,
    )
    arm = PreparedArm(
        retrieved_source_ids=("p000",),
        retrieved_unit_ids=("p000:u000",),
        selected=(evidence,),
    )
    return PreparedQuery(
        schema_version="experiment04.prepared.v1",
        dataset="hotpotqa",
        query_id=query_id,
        arms={name: arm for name in PRIMARY_ARMS},
    )


def _selected_three() -> SelectedEvidenceSet:
    return SelectedEvidenceSet(
        query_id="q",
        evidence=tuple(
            EvidenceCandidate(
                evidence_id=f"p{index:03d}",
                document_id=f"p{index:03d}",
                chunk_id="prepared",
                text=text,
                source_uri=f"fixture://p{index:03d}",
                retrieval_score=1.0,
                retrieval_rank=index + 1,
            )
            for index, text in enumerate(("aa", "bbbb", "cccccc"))
        ),
    )


def test_prompt_budget_keeps_maximal_rank_prefix_of_complete_evidence() -> None:
    selected = _selected_three()

    packed = maximal_whole_evidence_prefix(
        selected,
        render_prompt=lambda value: "|".join(item.text for item in value.evidence),
        input_token_count=len,
        max_input_tokens=7,
    )

    assert tuple(item.evidence_id for item in packed.evidence) == ("p000", "p001")
    assert tuple(item.text for item in packed.evidence) == ("aa", "bbbb")
    assert packed.evidence == selected.evidence[:2]


def test_prompt_budget_returns_empty_when_first_complete_unit_cannot_fit() -> None:
    selected = _selected_three()

    packed = maximal_whole_evidence_prefix(
        selected,
        render_prompt=lambda value: "fixed:" + "|".join(
            item.text for item in value.evidence
        ),
        input_token_count=len,
        max_input_tokens=5,
    )

    assert packed.evidence == ()


def test_prepare_query_uses_real_top40_reranker_pool_and_one_upstream_for_ours() -> None:
    reranker = _Reranker()

    prepared = prepare_query(
        _runtime(),
        embedder=_Embedder(),
        reranker=reranker,
        provence=_Provence(),
        selector=_Selector(),  # type: ignore[arg-type]
    )

    assert set(prepared.arms) == set(PRIMARY_ARMS)
    assert reranker.pool_sizes == [40]
    assert len(prepared.arms["granite_rerank_rag"].retrieved_source_ids) == 10
    assert len(prepared.arms["ours_seed13"].selected) == 8
    assert len({prepared.arms[arm].model_dump_json() for arm in OURS_ARMS}) == 1
    assert all(prepared.arms[arm].selection_status == "normal" for arm in OURS_ARMS)
    assert all(len(prepared.arms[arm].selected) == 10 for arm in BASELINE_ARMS)


def test_provence_unit_alignment_does_not_credit_title_only_output() -> None:
    candidate = {
        "title": "Synthetic title",
        "units": [
            {"unit_id": "p000:u000", "text": "First retained sentence."},
            {"unit_id": "p000:u001", "text": "Second removed sentence."},
        ],
    }

    assert retained_unit_ids_after_pruning(candidate, "Synthetic title") == ()
    assert retained_unit_ids_after_pruning(
        candidate,
        "Synthetic title First retained sentence.",
    ) == ("p000:u000",)


def test_inline_citation_parser_preserves_invalid_declared_index() -> None:
    sentences, declared = citation_sentences_from_inline(
        "A synthetic claim [1]. Another claim [9].",
        ("p000", "p001"),
    )

    assert declared == (1, 9)
    assert sentences[0].source_ids == ("p000",)
    assert sentences[1].source_ids == ()


def test_resume_requires_an_exact_ordered_prefix(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    append_canonical_jsonl(path, _prepared("q1"))
    append_canonical_jsonl(path, _prepared("q2"))

    count = ordered_prefix_count(
        path,
        ("q1", "q2", "q3"),
        PreparedQuery.model_validate,
    )

    assert count == 2


def test_scoring_occurs_after_frozen_output_and_uses_sentence_level_minicheck() -> None:
    prepared = _prepared()
    output = system_output(
        prepared=prepared,
        arm_id="dense_rag",
        answer="Synthetic answer",
        citation_indices=(1,),
        citation_sentences=(
            CitationSentence(sentence="Synthetic answer.", source_ids=("p000",)),
        ),
    )
    text = "Synthetic answer evidence."
    sidecar = {
        "schema_version": SCORER_SCHEMA_VERSION,
        "dataset": "hotpotqa",
        "query_id": "q",
        "gold_answer_aliases": ["Synthetic answer"],
        "support_units": [
            {
                "unit_id": "p000:u000",
                "source_id": "p000",
                "text": text,
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            }
        ],
        "component_id": "synthetic-component",
    }

    scored, calls = score_frozen_arm(
        sidecar_records=[sidecar],
        outputs=[output.model_dump(mode="json")],
        prepared_records=[prepared.model_dump(mode="json")],
        arm_id="dense_rag",
        entails=lambda premise, hypothesis: bool(premise and hypothesis),
    )

    assert calls == 1
    assert scored["aggregate"] == {"ret": 1.0, "sel": 1.0, "ans": 1.0, "cit": 1.0, "rar": 1.0}


def test_system_output_records_the_actual_budgeted_generation_context() -> None:
    prepared = _prepared()
    empty = SelectedEvidenceSet(query_id="q", evidence=())

    output = system_output(
        prepared=prepared,
        arm_id="dense_rag",
        selected_evidence=empty,
        answer="",
        citation_indices=(),
        citation_sentences=(),
    )

    assert output.retrieved_unit_ids == ("p000:u000",)
    assert output.selected_source_ids == ()
    assert output.selected_unit_ids == ()


def test_two_percent_runtime_errors_invalidate_the_whole_dataset_bundle(tmp_path: Path) -> None:
    expected = [f"q{index:03d}" for index in range(100)]
    paths: dict[str, Path] = {}
    for arm in PRIMARY_ARMS:
        path = tmp_path / f"{arm}.jsonl"
        paths[arm] = path
        for index, query_id in enumerate(expected):
            prepared = _prepared(query_id)
            row = system_output(
                prepared=prepared,
                arm_id=arm,
                answer="" if index < 2 else "Synthetic answer",
                citation_indices=(),
                citation_sentences=(),
                failure_stage="generation" if index < 2 else None,
                error_code="generation_error" if index < 2 else None,
            )
            append_canonical_jsonl(path, row)
    runtime = tmp_path / "runtime.jsonl"
    runtime.write_text("synthetic\n", encoding="utf-8")

    manifest = freeze_generation_manifest(
        dataset="hotpotqa",
        arm_paths=paths,
        expected_query_ids=expected,
        runtime_sha256=file_sha256(runtime),
        attempt_id="synthetic-attempt",
    )

    assert manifest["bundle_status"] == "INVALID_REQUIRES_FULL_RERUN"
    assert all(value["runtime_error_rate"] == 0.02 for value in manifest["arms"].values())


def test_paired_component_cluster_bootstrap_is_reproducible() -> None:
    result = paired_component_cluster_bootstrap(
        candidate={"q1": 1.0, "q2": 0.0, "q3": 1.0},
        baseline={"q1": 0.0, "q2": 0.0, "q3": 0.0},
        component_ids={"q1": "c1", "q2": "c1", "q3": "c2"},
        resamples=100,
        seed=13,
    )

    assert result["difference"] == 2 / 3
    assert result["n_components"] == 2
    assert result == paired_component_cluster_bootstrap(
        candidate={"q1": 1.0, "q2": 0.0, "q3": 1.0},
        baseline={"q1": 0.0, "q2": 0.0, "q3": 0.0},
        component_ids={"q1": "c1", "q2": "c1", "q3": "c2"},
        resamples=100,
        seed=13,
    )


def test_system_output_rejects_selected_units_outside_retrieval() -> None:
    try:
        SystemOutput(
            schema_version="experiment04.system_output.v1",
            dataset="hotpotqa",
            arm_id="dense_rag",
            query_id="q",
            retrieved_unit_ids=("p000:u000",),
            selected_unit_ids=("p999:u000",),
            selected_source_ids=("p999",),
            answer="",
            citation_indices=(),
            citation_sentences=(),
            failure_stage=None,
            error_code=None,
        )
    except ValueError as error:
        assert "subset" in str(error)
    else:
        raise AssertionError("invalid selected units were accepted")
