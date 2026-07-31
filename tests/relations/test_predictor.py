import pytest

from evidence_rag.relations.models import RelationLabel
from evidence_rag.relations.predictor import NLIRelationPredictor, ScoreFn, text_hash


def _scores(mapping: dict[tuple[str, str], dict[str, float]]) -> ScoreFn:
    def score(pairs):  # type: ignore[no-untyped-def]
        return [mapping[pair] for pair in pairs]

    return score


def _predictor(mapping: dict[tuple[str, str], dict[str, float]]) -> NLIRelationPredictor:
    return NLIRelationPredictor(score_fn=_scores(mapping), model_version="fake-1")


def test_argmax_picks_the_highest_scoring_class() -> None:
    prediction = _predictor(
        {("p", "h"): {"SUPPORTS": 0.7, "REFUTES": 0.2, "UNKNOWN": 0.1}}
    ).predict([("p", "h")])[0]
    assert prediction.label is RelationLabel.SUPPORTS
    assert prediction.confidence == pytest.approx(0.7)
    assert prediction.model_version == "fake-1"


def test_refutes_is_selected_when_it_leads() -> None:
    prediction = _predictor(
        {("p", "h"): {"SUPPORTS": 0.1, "REFUTES": 0.8, "UNKNOWN": 0.1}}
    ).predict([("p", "h")])[0]
    assert prediction.label is RelationLabel.REFUTES


def test_unknown_is_a_predicted_class_not_a_threshold() -> None:
    """A near-uniform distribution still yields the argmax, not an abstention — abstention has
    to be something the model predicts, or the gate acquires an absolute scale."""
    prediction = _predictor(
        {("p", "h"): {"SUPPORTS": 0.34, "REFUTES": 0.33, "UNKNOWN": 0.33}}
    ).predict([("p", "h")])[0]
    assert prediction.label is RelationLabel.SUPPORTS


def test_a_tie_resolves_to_unknown_never_to_supports() -> None:
    """A SUPPORTS edge is what makes a candidate droppable, so a coin-flip must not create one.
    The failure direction of this design is silence."""
    prediction = _predictor(
        {("p", "h"): {"SUPPORTS": 0.5, "REFUTES": 0.0, "UNKNOWN": 0.5}}
    ).predict([("p", "h")])[0]
    assert prediction.label is RelationLabel.UNKNOWN


def test_a_supports_refutes_tie_resolves_to_refutes_not_supports() -> None:
    prediction = _predictor(
        {("p", "h"): {"SUPPORTS": 0.5, "REFUTES": 0.5, "UNKNOWN": 0.0}}
    ).predict([("p", "h")])[0]
    assert prediction.label is RelationLabel.REFUTES


def test_hashes_are_recorded_for_every_edge() -> None:
    prediction = _predictor(
        {("p", "h"): {"SUPPORTS": 1.0, "REFUTES": 0.0, "UNKNOWN": 0.0}}
    ).predict([("p", "h")])[0]
    assert len(prediction.premise_hash) == 16
    assert len(prediction.hypothesis_hash) == 16
    assert prediction.premise_hash != prediction.hypothesis_hash
    assert prediction.premise_hash == text_hash("p")


def test_missing_class_in_scores_is_an_error_not_a_silent_zero() -> None:
    predictor = _predictor({("p", "h"): {"SUPPORTS": 1.0, "REFUTES": 0.0}})
    with pytest.raises(ValueError, match="missing relation class"):
        predictor.predict([("p", "h")])


def test_empty_input_returns_empty_output() -> None:
    assert _predictor({}).predict([]) == ()


def test_predictions_are_returned_in_input_order() -> None:
    predictor = _predictor(
        {
            ("p1", "h1"): {"SUPPORTS": 1.0, "REFUTES": 0.0, "UNKNOWN": 0.0},
            ("p2", "h2"): {"SUPPORTS": 0.0, "REFUTES": 1.0, "UNKNOWN": 0.0},
        }
    )
    labels = [p.label for p in predictor.predict([("p1", "h1"), ("p2", "h2")])]
    assert labels == [RelationLabel.SUPPORTS, RelationLabel.REFUTES]


def test_confidence_comes_from_the_winning_class_not_always_supports() -> None:
    """Every edge records a confidence as part of the frozen provenance, so a confidence taken
    from the wrong class corrupts the audit record silently while the label stays right."""
    refutes = _predictor(
        {("p", "h"): {"SUPPORTS": 0.1, "REFUTES": 0.8, "UNKNOWN": 0.1}}
    ).predict([("p", "h")])[0]
    assert refutes.label is RelationLabel.REFUTES
    assert refutes.confidence == pytest.approx(0.8)

    unknown = _predictor(
        {("p", "h"): {"SUPPORTS": 0.2, "REFUTES": 0.1, "UNKNOWN": 0.7}}
    ).predict([("p", "h")])[0]
    assert unknown.label is RelationLabel.UNKNOWN
    assert unknown.confidence == pytest.approx(0.7)
