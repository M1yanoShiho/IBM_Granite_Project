"""Pin the sentence rule the contract and the scorer share.

`GenerationResult` requires every sentence of an uncited answer to carry the
unverified label, so how a sentence is counted decides whether a correctly
annotated answer is accepted or destroyed. G7 lost one answer per annotate arm to
this: "Mount St. Helens erupted." counted as two sentences carrying one label.

The abbreviation list is deliberately small, so the tests here guard both
directions -- a missed abbreviation splits one sentence into two, and a
spurious entry MERGES two real sentences, which is the worse error.
"""

import pytest

from evidence_rag.contracts.models import (
    REVIEW_ANNOTATION,
    UNVERIFIED_ANNOTATION,
    count_sentences,
    ends_with_abbreviation,
    split_sentences,
)


@pytest.mark.parametrize(
    "text",
    [
        "The eruption of Mount St. Helens was a VEI 5 event.",
        "Patrick S. Castagne wrote the national anthem of Trinidad and Tobago.",
        "The Brown v. Board of Education case took place in Topeka, Kansas.",
        "John L. O'Sullivan coined the phrase \"Manifest Destiny\".",
        "Rick Kriseman won the mayor race in St. Petersburg, Florida.",
        "Francis Ouimet beat Harry Vardon and Ted Ray in the 1913 U.S. Open.",
        "The team won on Jan. 11, 1970.",
        "She works at Acme Inc. in Ohio.",
    ],
)
def test_an_abbreviation_does_not_end_a_sentence(text: str) -> None:
    """Every one of these is a real G7 answer that counted as two sentences."""
    assert count_sentences(text) == 1


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("The river ends in Peru. The Amazon is long.", 2),
        ("Alpha holds. Beta holds.", 2),
        # The claim splitter emits fragments terminated as sentences. They ARE
        # separate sentences in the answer and each carries its own label, so
        # merging them would be wrong.
        ("Spain won their first FIFA World Cup. in 2010.", 2),
        ("The Vikings hold the record. having lost four.", 2),
        ("Rome fell. Carthage fell too. Athens endured.", 3),
    ],
)
def test_real_boundaries_are_still_boundaries(text: str, expected: int) -> None:
    assert count_sentences(text) == expected


def test_a_fully_annotated_abbreviation_sentence_is_accepted() -> None:
    """The regression G7 lost an answer to, stated as the contract sees it."""
    answer = f"The eruption of Mount St. Helens was a VEI 5 event. {UNVERIFIED_ANNOTATION}"
    assert count_sentences(answer) == answer.count(UNVERIFIED_ANNOTATION)


def test_labels_are_not_counted_as_sentences() -> None:
    answer = f"Alpha holds. {UNVERIFIED_ANNOTATION} Beta holds. {REVIEW_ANNOTATION}"
    assert count_sentences(answer) == 2


def test_split_sentences_keeps_the_text() -> None:
    text = "Mount St. Helens erupted. Rome fell."
    parts = split_sentences(text)
    assert parts == ["Mount St. Helens erupted.", "Rome fell."]


@pytest.mark.parametrize(
    ("part", "expected"),
    [
        ("The eruption of Mount St.", True),
        ("Patrick S.", True),
        ("the 1913 U.S.", True),
        ("The river ends in Peru.", False),
        ("Alpha holds.", False),
        ("no trailing period", False),
        ("", False),
    ],
)
def test_ends_with_abbreviation(part: str, expected: bool) -> None:
    assert ends_with_abbreviation(part) is expected


def test_empty_and_whitespace() -> None:
    assert count_sentences("") == 0
    assert count_sentences("   ") == 0
    assert split_sentences("") == []
