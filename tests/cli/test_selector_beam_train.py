import pytest

from evidence_rag.cli.selector_beam_train import _balanced_class_weights
from evidence_rag.materializer.selector_beam_data import (
    BeamTrainingExample,
    EvidenceLabel,
)


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
    assert irrelevant == pytest.approx(2.0 / 3.0)
    assert required == pytest.approx(1.0)
    assert harmful == pytest.approx(2.0)


def test_class_weights_reject_missing_harmful_label() -> None:
    source = (_example(EvidenceLabel.REQUIRED, "only"),)
    with pytest.raises(ValueError, match="all three"):
        _balanced_class_weights((source,))
