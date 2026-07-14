import pytest

from evidence_rag.evaluation.metrics import (
    citation_validity,
    evidence_precision,
    evidence_recall,
)


def test_recall_measures_required_evidence_kept() -> None:
    assert evidence_recall(("ev-1", "ev-2"), ("ev-2", "ev-3")) == 0.5


def test_precision_measures_selected_noise() -> None:
    assert evidence_precision(("ev-1", "ev-bad"), ("ev-1",)) == 0.5


def test_empty_required_set_is_not_silently_scored() -> None:
    with pytest.raises(ValueError):
        evidence_recall(("ev-1",), ())


def test_citation_validity_uses_selected_ids() -> None:
    assert citation_validity(("ev-1", "ev-bad"), ("ev-1",)) == 0.5
