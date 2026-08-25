"""The natively-binary relation arm (M0 §3.1 control, §9.8 out-of-sample check).

Torch-free by construction. The tokenizer only ever reaches this module as an `encode` callable,
so every rule below is exercised against a fake that returns the ids the REAL tokenizer was
observed to return — see `_REAL_ENCODE`.
"""

import pytest

from evidence_rag.relations.minicheck import (
    MINICHECK_FLAN_T5_LARGE,
    VERIFIED_BINARY_PROTOCOLS,
    BinaryProtocol,
    build_input,
    protocol_for,
    resolve_label_token_ids,
    to_scores,
)

# Observed on 2026-08-04 with transformers 5.10.2 against the real checkpoint's tokenizer.
# "0" does not start with a "▁0" token — there is none in the T5 vocab — so it decomposes into
# the bare "▁" (id 3) plus "0" (id 632). "1" does have "▁1" (id 209). This asymmetry is the whole
# reason the ids are derived rather than looked up by token text.
_REAL_ENCODE = {
    "0": [3, 632, 1],
    "1": [209, 1],
}


def _fake_encode(text: str) -> list[int]:
    return _REAL_ENCODE[text]


def test_an_unregistered_checkpoint_is_refused_rather_than_guessed_at() -> None:
    """Same discipline as `gate0b.LABEL_ORDER` and `export_qa2d.INPUT_TEMPLATE`.

    A wrong prompt format or a wrong label id does not crash — it yields a plausible probability
    for the wrong question — so an unverified id must not reach the Hub at all.
    """
    with pytest.raises(ValueError, match="no verified binary protocol"):
        protocol_for("some/unknown-checkpoint")


def test_the_registered_checkpoint_is_the_one_verified_on_the_hub() -> None:
    assert MINICHECK_FLAN_T5_LARGE == "lytang/MiniCheck-Flan-T5-Large"
    assert set(VERIFIED_BINARY_PROTOCOLS) == {MINICHECK_FLAN_T5_LARGE}


def test_input_is_the_checkpoints_own_format_not_a_sentence_pair() -> None:
    """`predict: {document}</s>{claim}` as ONE sequence.

    The three-class arms hand the tokenizer a (premise, hypothesis) PAIR and let it insert the
    separator. This checkpoint is a seq2seq LM with no pair encoding: the separator is literal
    text that the tokenizer maps onto the EOS id. Feeding it a pair, or dropping the prefix,
    produces fluent nonsense rather than an error.
    """
    protocol = protocol_for(MINICHECK_FLAN_T5_LARGE)
    assert build_input(protocol, "Kennedy won.", "the answer is Kennedy") == (
        "predict: Kennedy won.</s>the answer is Kennedy"
    )


def test_label_ids_are_derived_from_the_tokenizer_and_agree_with_the_recorded_pair() -> None:
    protocol = protocol_for(MINICHECK_FLAN_T5_LARGE)
    assert resolve_label_token_ids(protocol, _fake_encode) == (3, 209)


def test_the_positive_id_is_the_first_token_of_the_string_not_the_bare_digit_token() -> None:
    """The trap this derivation exists to avoid.

    `convert_tokens_to_ids(["0", "1"])` returns (632, 536) on the real tokenizer — neither is a
    label id. Reading the decoder's first-position logits at those columns is not an error and
    not a crash; it silently scores two unrelated vocabulary entries, and every downstream number
    stays in [0, 1]. Only `encode(text)[0]` reproduces the checkpoint's own (3, 209).
    """
    protocol = protocol_for(MINICHECK_FLAN_T5_LARGE)
    negative, positive = resolve_label_token_ids(protocol, _fake_encode)
    assert (negative, positive) == (_REAL_ENCODE["0"][0], _REAL_ENCODE["1"][0])
    assert (negative, positive) != (632, 536)


def test_a_tokenizer_disagreeing_with_the_recorded_pair_fails_loudly() -> None:
    """A tokenizer revision that renumbers the vocabulary must stop the run.

    This is the tripwire the recorded ids exist for: they are not the source of truth — the
    tokenizer is — but a silent disagreement between them means the checkpoint on disk is not
    the checkpoint that was verified, and the resulting probabilities would be meaningless.
    """
    protocol = protocol_for(MINICHECK_FLAN_T5_LARGE)
    with pytest.raises(ValueError, match="does not match the verified label token ids"):
        resolve_label_token_ids(protocol, lambda text: [999, 1])


def test_support_probability_becomes_the_binary_score_dict() -> None:
    """Keys are exactly A1's output space, so `NLIRelationPredictor` needs no special case."""
    assert to_scores(0.75) == {"SUPPORTS": 0.75, "NOT_SUPPORTED": 0.25}


def test_the_two_scores_are_a_distribution_unlike_the_three_class_collapse() -> None:
    """This pair sums to 1; `gate0b._collapse_to_binary`'s deliberately does not.

    The difference is real and is not a defect on either side. The collapse takes the max of two
    surviving classes so that binary argmax == three-class argmax relabelled (§9.10a), which
    cannot sum to 1. Here the checkpoint is natively two-way, so its own softmax over the two
    label logits IS the distribution, and argmax over it is the checkpoint's own decision rule —
    identical to the `raw_prob > .5` its reference implementation applies.

    That equivalence is why this arm introduces no threshold and stays inside §9.5a: .5 is not a
    free parameter chosen against Gate results, it is where a two-way argmax already sits. A
    tuned θ would still require amendment A2.
    """
    for probability in (0.0, 0.5, 0.75, 1.0):
        scores = to_scores(probability)
        assert scores["SUPPORTS"] + scores["NOT_SUPPORTED"] == pytest.approx(1.0)


def test_protocol_records_where_each_field_was_verified() -> None:
    """The provenance is part of the datum. A future reader must be able to re-check the format
    and the ids without re-deriving them from a paper."""
    protocol = protocol_for(MINICHECK_FLAN_T5_LARGE)
    assert isinstance(protocol, BinaryProtocol)
    assert protocol.max_input_length == 2048
    assert protocol.source.startswith("https://")
