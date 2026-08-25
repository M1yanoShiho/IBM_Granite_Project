"""Pin the sentence split the G5 scorer applies to annotated answers.

The Generator writes the annotation *after* the sentence terminator, so a plain
sentence splitter strands it as a fragment. That miscounts the composition and,
worse, inflates ALCE's recall denominator by one per annotated sentence -- a
penalty that lands only on the arms that annotate. These tests pin the fix.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from g5_score import annotated_sent_split  # noqa: E402

from evidence_rag.contracts.models import UNVERIFIED_ANNOTATION, count_sentences  # noqa: E402
from evidence_rag.generator.verify_annotate import is_unverified_annotation  # noqa: E402


def test_marker_stays_with_its_sentence() -> None:
    answer = f"The Mills Brothers were inducted first. {UNVERIFIED_ANNOTATION}"
    parts = annotated_sent_split(answer)
    assert len(parts) == 1
    assert is_unverified_annotation(parts[0])
    assert "Mills Brothers" in parts[0]


def test_no_sentence_is_left_bare_by_the_split() -> None:
    answer = (
        f"Alpha happened. {UNVERIFIED_ANNOTATION} "
        f"Beta happened too. {UNVERIFIED_ANNOTATION}"
    )
    parts = annotated_sent_split(answer)
    assert len(parts) == 2
    assert all(is_unverified_annotation(part) for part in parts)


def test_split_agrees_with_the_contract_sentence_count() -> None:
    """The scorer and the contract validator must count the same sentences, or one
    of them is measuring an answer the other never saw."""
    answer = (
        f"The first episode is called \"Remember\". {UNVERIFIED_ANNOTATION} "
        f"It aired in 2016. {UNVERIFIED_ANNOTATION}"
    )
    assert len(annotated_sent_split(answer)) == count_sentences(answer)


def test_unannotated_answers_are_untouched() -> None:
    answer = "Rome is the capital. Italy is in Europe."
    parts = annotated_sent_split(answer)
    assert len(parts) == 2
    assert not any(is_unverified_annotation(part) for part in parts)


def test_a_leading_stray_marker_is_not_dropped() -> None:
    """Nothing to attach it to -- keep it rather than silently losing content."""
    parts = annotated_sent_split(f"{UNVERIFIED_ANNOTATION} Alpha happened.")
    assert any(UNVERIFIED_ANNOTATION in part for part in parts)
