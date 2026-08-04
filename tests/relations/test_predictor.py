import pytest

from evidence_rag.relations.models import RelationLabel
from evidence_rag.relations.predictor import (
    NLIRelationPredictor,
    ScoreFn,
    fingerprinted_version,
    require_fingerprinted_version,
    text_hash,
    weight_fingerprint,
)


def _scores(mapping: dict[tuple[str, str], dict[str, float]]) -> ScoreFn:
    def score(pairs):  # type: ignore[no-untyped-def]
        return [mapping[pair] for pair in pairs]

    return score


def _predictor(mapping: dict[tuple[str, str], dict[str, float]]) -> NLIRelationPredictor:
    return NLIRelationPredictor(score_fn=_scores(mapping), model_version="fake-1")


def test_argmax_picks_the_highest_scoring_class() -> None:
    prediction = _predictor(
        {("p", "h"): {"SUPPORTS": 0.7, "NOT_SUPPORTED": 0.3}}
    ).predict([("p", "h")])[0]
    assert prediction.label is RelationLabel.SUPPORTS
    assert prediction.confidence == pytest.approx(0.7)
    assert prediction.model_version == "fake-1"


def test_not_supported_is_selected_when_it_leads() -> None:
    prediction = _predictor(
        {("p", "h"): {"SUPPORTS": 0.1, "NOT_SUPPORTED": 0.9}}
    ).predict([("p", "h")])[0]
    assert prediction.label is RelationLabel.NOT_SUPPORTED


def test_not_supported_is_a_predicted_class_not_a_threshold() -> None:
    """A near-uniform distribution still yields the argmax, not an abstention — abstention has
    to be something the model predicts, or the gate acquires an absolute scale (M0 §9.5a)."""
    prediction = _predictor(
        {("p", "h"): {"SUPPORTS": 0.51, "NOT_SUPPORTED": 0.49}}
    ).predict([("p", "h")])[0]
    assert prediction.label is RelationLabel.SUPPORTS


def test_a_tie_resolves_to_not_supported_never_to_supports() -> None:
    """A SUPPORTS edge is what makes a candidate droppable, so a coin-flip must not create one.
    The failure direction of this design is silence (M0 §2.4, unchanged by A1)."""
    prediction = _predictor(
        {("p", "h"): {"SUPPORTS": 0.5, "NOT_SUPPORTED": 0.5}}
    ).predict([("p", "h")])[0]
    assert prediction.label is RelationLabel.NOT_SUPPORTED


def test_the_model_never_emits_a_schema_only_label() -> None:
    """REFUTES and UNKNOWN survive in the enum for 0B-1 gold, historical dumps and the retained
    `CLAIM_REFUTES` edge type. A scorer still speaking the pre-A1 three-class contract must fail
    loudly: silently ignoring its REFUTES/UNKNOWN mass would collapse the label in a second
    place, and the two collapses could disagree without any number looking wrong."""
    predictor = _predictor(
        {("p", "h"): {"SUPPORTS": 0.2, "NOT_SUPPORTED": 0.8, "REFUTES": 0.5, "UNKNOWN": 0.3}}
    )
    with pytest.raises(ValueError, match="not emitted by the relation model"):
        predictor.predict([("p", "h")])


def test_hashes_are_recorded_for_every_edge() -> None:
    prediction = _predictor(
        {("p", "h"): {"SUPPORTS": 1.0, "NOT_SUPPORTED": 0.0}}
    ).predict([("p", "h")])[0]
    assert len(prediction.premise_hash) == 16
    assert len(prediction.hypothesis_hash) == 16
    assert prediction.premise_hash != prediction.hypothesis_hash
    assert prediction.premise_hash == text_hash("p")


def test_missing_class_in_scores_is_an_error_not_a_silent_zero() -> None:
    predictor = _predictor({("p", "h"): {"SUPPORTS": 1.0}})
    with pytest.raises(ValueError, match="missing relation class"):
        predictor.predict([("p", "h")])


def test_empty_input_returns_empty_output() -> None:
    assert _predictor({}).predict([]) == ()


def test_predictions_are_returned_in_input_order() -> None:
    predictor = _predictor(
        {
            ("p1", "h1"): {"SUPPORTS": 1.0, "NOT_SUPPORTED": 0.0},
            ("p2", "h2"): {"SUPPORTS": 0.0, "NOT_SUPPORTED": 1.0},
        }
    )
    labels = [p.label for p in predictor.predict([("p1", "h1"), ("p2", "h2")])]
    assert labels == [RelationLabel.SUPPORTS, RelationLabel.NOT_SUPPORTED]


def test_confidence_comes_from_the_winning_class_not_always_supports() -> None:
    """Every edge records a confidence as part of the frozen provenance, so a confidence taken
    from the wrong class corrupts the audit record silently while the label stays right."""
    not_supported = _predictor(
        {("p", "h"): {"SUPPORTS": 0.1, "NOT_SUPPORTED": 0.8}}
    ).predict([("p", "h")])[0]
    assert not_supported.label is RelationLabel.NOT_SUPPORTED
    assert not_supported.confidence == pytest.approx(0.8)

    supports = _predictor(
        {("p", "h"): {"SUPPORTS": 0.7, "NOT_SUPPORTED": 0.2}}
    ).predict([("p", "h")])[0]
    assert supports.label is RelationLabel.SUPPORTS
    assert supports.confidence == pytest.approx(0.7)


def test_fingerprint_separates_checkpoints_that_differ_only_in_weight_VALUES() -> None:
    """The poison case. Two fine-tuning seeds of one architecture share every parameter name,
    shape and dtype and differ only in the numbers. A `model_version` that does not hash the
    values lets the edge cache serve seed 13's edges under seed 42's name, which would silently
    void the three-seed clause (M0 §5.4 / tracker R013-R015) with no metric able to notice."""
    seed13 = (("encoder.weight|(2, 2)|float32", b"\x01\x02\x03\x04"),)
    seed42 = (("encoder.weight|(2, 2)|float32", b"\x01\x02\x03\x05"),)
    assert weight_fingerprint(seed13) != weight_fingerprint(seed42)


def test_fingerprint_is_deterministic_and_order_independent() -> None:
    """Two runs of the same checkpoint must hit the same cache entry, and `state_dict()` order
    is not contractual."""
    parts = (("a|(1,)|float32", b"\x00"), ("b|(1,)|float32", b"\x01"))
    assert weight_fingerprint(parts) == weight_fingerprint(tuple(reversed(parts)))


def test_fingerprinted_version_keeps_the_model_id_readable() -> None:
    assert fingerprinted_version("tals/albert", "0123456789abcdef") == (
        "tals/albert@0123456789abcdef"
    )


def test_a_hand_written_model_version_is_rejected() -> None:
    """Defence in depth: the real fix is deriving the version from the weights, but a future
    caller can still hand-write a label, so anything not carrying a fingerprint is refused."""
    with pytest.raises(ValueError, match="not fingerprinted"):
        require_fingerprinted_version("relation-builder-v1")


def test_a_fingerprinted_version_is_accepted_and_returned() -> None:
    version = fingerprinted_version("tals/albert", "0123456789abcdef")
    assert require_fingerprinted_version(version) == version
