import pytest

from evidence_rag.cli.selector_beam_train import _balanced_class_weights, _sanity_cases
from evidence_rag.materializer.selector_beam_data import (
    BeamSelectorCase,
    BeamTrainingExample,
    EvidenceLabel,
)
from evidence_rag.selector.beam_three_class import BeamCandidate


def _example(label: EvidenceLabel, dataset: str) -> BeamTrainingExample:
    return BeamTrainingExample(
        query_id=f"{dataset}-{label.name}",
        question="question",
        selected_passages=(),
        candidate_passage="passage",
        label=label,
        hop=0,
        dataset=dataset,
    )


def test_class_weights_respect_equal_dataset_source_mass() -> None:
    niah = (
        _example(EvidenceLabel.IRRELEVANT, "niah"),
        _example(EvidenceLabel.REQUIRED, "niah"),
        _example(EvidenceLabel.HARMFUL, "niah"),
    )
    twowiki = (
        _example(EvidenceLabel.IRRELEVANT, "2wiki"),
        _example(EvidenceLabel.IRRELEVANT, "2wiki"),
        _example(EvidenceLabel.REQUIRED, "2wiki"),
    )
    irrelevant, required, harmful = _balanced_class_weights((niah, twowiki))
    assert irrelevant == pytest.approx((2.0 / 3.0) ** 0.5)
    assert required == pytest.approx(1.0)
    assert harmful == pytest.approx(2.0**0.5)


def test_class_weights_reject_missing_harmful_label() -> None:
    source = (_example(EvidenceLabel.REQUIRED, "only"),)
    with pytest.raises(ValueError, match="all three"):
        _balanced_class_weights((source,))


def _case(
    query_id: str,
    *,
    dataset: str,
    required_count: int,
    harmful: bool,
) -> BeamSelectorCase:
    required = tuple(f"required-{query_id}-{index}" for index in range(required_count))
    harmful_id = f"harmful-{query_id}" if harmful else None
    document_ids = (*required, *((harmful_id,) if harmful_id is not None else ()), "other")
    candidates = tuple(
        BeamCandidate(
            evidence_id=f"evidence-{index}",
            document_id=document_id,
            text=f"passage {index}",
            retrieval_rank=index + 1,
            retrieval_score=1.0 / (index + 1),
        )
        for index, document_id in enumerate(document_ids)
    )
    labels = tuple(
        EvidenceLabel.REQUIRED
        if document_id in required
        else EvidenceLabel.HARMFUL
        if document_id == harmful_id
        else EvidenceLabel.IRRELEVANT
        for document_id in document_ids
    )
    return BeamSelectorCase(
        query_id=query_id,
        question="question",
        candidates=candidates,
        labels=labels,
        dataset=dataset,
        required_document_ids=required,
        harmful_document_id=harmful_id,
    )


def test_sanity_cases_exclude_impossible_more_than_output_limit() -> None:
    impossible = _case("a-impossible", dataset="niah", required_count=11, harmful=True)
    niah = (impossible, _case("b-valid", dataset="niah", required_count=2, harmful=True))
    twowiki = (_case("wiki", dataset="2wiki", required_count=2, harmful=False),)

    selected_niah, selected_twowiki = _sanity_cases(niah, twowiki, count=1, max_selected=10)

    assert [case.query_id for case in selected_niah] == ["b-valid"]
    assert [case.query_id for case in selected_twowiki] == ["wiki"]
