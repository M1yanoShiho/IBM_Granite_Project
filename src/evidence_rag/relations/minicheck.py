"""Natively-binary relation checkpoints (M0 §3.1 control arm, §9.8 out-of-sample check).

WHY THIS IS A SEPARATE PATH FROM `gate0b.LABEL_ORDER`. That table maps a three-class
sequence-classification head onto the frozen schema, and `gate0b._collapse_to_binary` reduces it
to A1's output space afterwards. MiniCheck has neither piece: the Hub reports
`architectures: ["T5ForConditionalGeneration"]` with no `id2label`, so
`AutoModelForSequenceClassification` cannot load it and there is no class order to record. It is
a seq2seq LM scored by reading ONE decoder step and comparing the logits of two vocabulary
entries. Inventing a three-name order for it — the obvious shortcut — would be fabricating a
class it does not have.

Under A1 (g2-proto-2) this arm needs no negated-claim second pass and no combination rule. The
pre-A1 protocol required probing twice to reach SUPPORTS/REFUTES/UNKNOWN; the binary output space
is now the amended output space exactly, which M0 §9.8 names as an independent benefit of the
amendment. This module therefore implements one pass and nothing more.

WHAT MAKES THIS DANGEROUS ENOUGH TO REGISTER. A wrong prompt prefix, a wrong separator or a wrong
pair of label ids does not raise: it returns a well-formed probability in [0, 1] for a question
nobody asked, and every metric downstream stays plausible. That is the same failure mode
`LABEL_ORDER` and `export_qa2d.INPUT_TEMPLATE` exist to block, so the same rule applies here —
the format and the ids are recorded per checkpoint with their source, and an unregistered id is
refused before anything touches the Hub.

VERIFIED 2026-08-04, three independent sources:

  * https://huggingface.co/api/models/lytang/MiniCheck-Flan-T5-Large — architectures
    ["T5ForConditionalGeneration"], model_type "t5", no id2label. This is the "MiniCheck-FT5
    (770M)" of M0 §3.1: Flan-T5-Large is 780M, and the card's "best fact-checking model with
    size < 1B" is §3.1's "LLM-AggreFact <1B SOTA". The card calls it MiniCheck-Flan-T5-Large;
    "FT5" is the protocol's abbreviation, not the Hub id.
  * `minicheck_web/inference.py` in that repo and `minicheck/inference.py` in
    https://github.com/Liyan06/MiniCheck — the prompt, the one-step decoder, `logits[:, [3, 209]]`
    under the comment "# 3 for no support and 209 for support", softmax over those two columns,
    support probability at index 1, and `max_model_len = 2048`.
  * The checkpoint's own tokenizer, loaded locally with transformers 5.10.2 (see
    `resolve_label_token_ids` for what that run established and why it matters).

Paper: Tang, Laban & Durrett, "MiniCheck: Efficient Fact-Checking of LLMs on Grounding
Documents", EMNLP 2024 (arXiv:2404.10774). Card semantics: 1 = supported, 0 = unsupported.
Per M0 §3.1 this checkpoint is a SYSTEM COMPONENT, not a label source; using it does not breach
the zero-new-annotation constraint, and Gate 0B's acceptance labels still come only from official
test splits and deterministic provenance.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from evidence_rag.relations.models import RelationLabel

Encode = Callable[[str], Sequence[int]]

MINICHECK_FLAN_T5_LARGE = "lytang/MiniCheck-Flan-T5-Large"


@dataclass(frozen=True)
class BinaryProtocol:
    """One checkpoint's verified scoring protocol.

    `negative_token_id` / `positive_token_id` are a TRIPWIRE, not the source of truth: the
    tokenizer on disk is. They are recorded so that a checkpoint whose vocabulary has been
    renumbered stops the run instead of quietly scoring different columns.
    """

    prefix: str
    separator: str
    negative_text: str
    positive_text: str
    negative_token_id: int
    positive_token_id: int
    max_input_length: int
    source: str


VERIFIED_BINARY_PROTOCOLS: dict[str, BinaryProtocol] = {
    MINICHECK_FLAN_T5_LARGE: BinaryProtocol(
        prefix="predict: ",
        # Literal text, not a tokenizer argument. The T5 tokenizer maps "</s>" in a string onto
        # the EOS id, which is what the checkpoint was trained to see between document and claim.
        separator="</s>",
        negative_text="0",
        positive_text="1",
        negative_token_id=3,
        positive_token_id=209,
        # The reference implementation's max_model_len. It also chunks longer documents and takes
        # the maximum over chunks; that is deliberately NOT reproduced here, because a 0B-2
        # premise is a single ~100-word needle passage and chunking is a no-op at that length.
        # Re-examine this before the checkpoint is ever pointed at a full retrieval pool.
        max_input_length=2048,
        source="https://huggingface.co/lytang/MiniCheck-Flan-T5-Large/blob/main/minicheck_web/inference.py",
    )
}


def protocol_for(model_id: str) -> BinaryProtocol:
    """Refuse an unverified checkpoint, naming what verification means."""
    protocol = VERIFIED_BINARY_PROTOCOLS.get(model_id)
    if protocol is None:
        raise ValueError(
            f"no verified binary protocol for {model_id!r}; record its prompt format and its "
            "label token ids in VERIFIED_BINARY_PROTOCOLS, with the source they were read from, "
            "before running the gate — a wrong format yields plausible probabilities rather "
            "than an error"
        )
    return protocol


def build_input(protocol: BinaryProtocol, premise: str, hypothesis: str) -> str:
    """One sequence, not a sentence pair.

    The three-class arms call `tokenizer(premise, hypothesis)` and let the tokenizer build the
    pair encoding. A seq2seq LM has no such encoding; the document, the separator and the claim
    are one string, and the prefix is part of what the checkpoint was tuned on.
    """
    return f"{protocol.prefix}{premise}{protocol.separator}{hypothesis}"


def resolve_label_token_ids(protocol: BinaryProtocol, encode: Encode) -> tuple[int, int]:
    """Derive (negative, positive) decoder ids from the tokenizer, then check the recorded pair.

    THE FIRST TOKEN OF THE STRING, NOT THE TOKEN NAMED BY THE STRING. Measured on the real
    tokenizer (transformers 5.10.2, 2026-08-04):

        convert_tokens_to_ids(["0", "1"])  ->  [632, 536]      <- WRONG, neither is a label id
        encode("0")                        ->  [3, 632, 1]     <- first token 3
        encode("1")                        ->  [209, 1]        <- first token 209
        convert_ids_to_tokens(3), (209)    ->  "▁", "▁1"

    The T5 vocabulary has "▁1" but no "▁0", so "0" decomposes into the bare space marker plus the
    digit while "1" does not. The reference implementation's (3, 209) are therefore the first
    decoder token of each label STRING — the only quantity a one-step decoder can be asked about.
    Looking the ids up by token text instead returns two unrelated vocabulary entries and scores
    them without complaint.

    Takes `encode` rather than a tokenizer so this stays torch-free and unit-testable.
    """
    derived = (
        encode(protocol.negative_text)[0],
        encode(protocol.positive_text)[0],
    )
    recorded = (protocol.negative_token_id, protocol.positive_token_id)
    if derived != recorded:
        raise ValueError(
            f"this tokenizer's first-token ids for "
            f"({protocol.negative_text!r}, {protocol.positive_text!r}) are {derived}, which "
            f"does not match the verified label token ids {recorded}. The checkpoint on disk is "
            "not the one this protocol was verified against; re-verify before running the gate "
            "rather than adjusting the recorded pair."
        )
    return derived


def to_scores(support_probability: float) -> dict[str, float]:
    """A1's output space, keyed off the enum so the two cannot drift apart.

    Unlike `gate0b._collapse_to_binary`, this pair sums to 1: there are only two classes to begin
    with, so the checkpoint's softmax over its two label logits already IS the distribution.
    Argmax over it coincides with the `raw_prob > .5` its reference implementation applies, which
    is why this arm introduces no threshold — .5 is where a two-way argmax sits, not a parameter
    tuned against Gate results. Introducing a real θ still requires amendment A2 (M0 §9.5a).
    """
    return {
        RelationLabel.SUPPORTS.value: support_probability,
        RelationLabel.NOT_SUPPORTED.value: 1.0 - support_probability,
    }
