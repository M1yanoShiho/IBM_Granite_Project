"""Tests for the NIAH domain-adaptation half of M0 §3.8 and its sealed-600 blocker.

§3.8 carries one hard constraint on this half: "NIAH 域适配对的 parent page 必须与 sealed-600
零重叠". sealed-600 does not exist yet (§8 item 2), so the half is not legitimately runnable
today. The tests below pin that it REFUSES rather than warns or skips — a training run that
quietly omitted the overlap check would finish, save five checkpoints, and produce a Gate 0B
number that looks exactly like a valid one.
"""

import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest

from evidence_rag.materializer.provenance import MutationRecord
from evidence_rag.relations.models import RelationLabel
from evidence_rag.relations.niah_adaptation import (
    build_niah_examples,
    load_sealed_parents,
)
from evidence_rag.relations.training import family_key, page_key

_QUESTIONS = {"q1": "who won?"}
_TEXTS = {"needle": "Kennedy won", "cf::needle": "Nixon won"}
_PARENTS = {"needle": "1960 election", "cf::needle": "1960 election"}


def _record(query_id: str = "q1") -> MutationRecord:
    return MutationRecord(
        query_id=query_id,
        needle_document_id="needle",
        counterfactual_document_id="cf::needle",
        gold_value="kennedy",
        gold_alias_used="Kennedy",
        replacement_value="Nixon",
        string_class="proper_name_1",
        seed=42,
        char_span=(0, 7),
        text_hash_before="a" * 8,
        text_hash_after="b" * 8,
        answer_bank_hash="c" * 8,
    )


@dataclass(frozen=True)
class _Fingerprint:
    parent_pages: tuple[str, ...]


_TITLES: list[str] = []


@pytest.fixture(autouse=True)
def _fake_corpus_builder(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stand in for the corpus line's `materializer.sealed600`.

    That module is the other line's deliverable and is not importable here, so a fake is the
    only way to exercise the interface. It fakes the PUBLISHED contract and nothing else --
    `read_split_fingerprint(directory).parent_pages` -- because reaching for any other attribute
    would be this module inventing a field it cannot verify.
    """
    module = types.ModuleType("evidence_rag.materializer.sealed600")
    module.read_split_fingerprint = lambda directory: _Fingerprint(  # type: ignore[attr-defined]
        tuple(_TITLES)
    )
    monkeypatch.setitem(sys.modules, "evidence_rag.materializer.sealed600", module)
    _TITLES.clear()


def _sealed(tmp_path: Path, *titles: str) -> Path:
    """Set what the sealed-600 build reports, and return the directory the loader is given."""
    _TITLES[:] = titles
    return tmp_path / "sealed600"


def _raise_import_error(name: str) -> object:
    raise ImportError(f"no module named {name}")


def _build(**overrides: object):  # type: ignore[no-untyped-def]
    arguments: dict[str, object] = {
        "records": [_record()],
        "question_by_query": _QUESTIONS,
        "text_by_document": _TEXTS,
        "parent_by_document": _PARENTS,
        "twin_label": RelationLabel.REFUTES,
    }
    arguments.update(overrides)
    return build_niah_examples(**arguments)  # type: ignore[arg-type]


# --- the blocker -------------------------------------------------------------------------------


def test_the_niah_half_refuses_to_run_without_a_sealed_600_parent_set() -> None:
    """The whole point. sealed-600 is Line A's artifact and does not exist yet, so this half
    cannot be legitimately executed today — and 'cannot' has to mean raise. A warn-and-continue
    would hand back a perfectly ordinary set of training examples."""
    with pytest.raises(ValueError, match="sealed-600"):
        _build(sealed=None)


def test_an_empty_sealed_parent_set_is_refused_instead_of_passing_vacuously(
    tmp_path: Path,
) -> None:
    """Zero overlap against an empty set is true for every input. That is the exact shape of
    this project's recurring defect: the check runs, reports success, and means nothing."""
    with pytest.raises(ValueError, match="no parent pages"):
        load_sealed_parents(_sealed(tmp_path))


def test_an_overlapping_parent_page_stops_the_build(tmp_path: Path) -> None:
    """The needle article is in sealed-600, so training on it means the eval set was in the
    training data. The message has to name the pages, or the operator cannot act on it."""
    sealed = load_sealed_parents(_sealed(tmp_path, "1960 election", "Other page"))
    with pytest.raises(ValueError, match="overlap"):
        _build(sealed=sealed)


def test_a_disjoint_parent_page_builds_normally(tmp_path: Path) -> None:
    sealed = load_sealed_parents(_sealed(tmp_path, "Some other article"))
    examples = _build(sealed=sealed)
    assert len(examples) == 4


def test_a_document_missing_from_the_parent_index_stops_the_build(tmp_path: Path) -> None:
    """`ParentIndex.parent_of` falls back to the document_id for an unresolved document, and
    §3.4 argues that is safe — for VOTE COUNTING, where failing to merge only loses a merge.
    Here the direction of harm reverses: a synthetic parent can never collide with a sealed
    title, so the zero-overlap check would pass precisely for the documents it could not see.
    No fallback on this path."""
    sealed = load_sealed_parents(_sealed(tmp_path, "Some other article"))
    with pytest.raises(ValueError, match="no parent page"):
        _build(sealed=sealed, parent_by_document={"needle": "1960 election"})


def test_the_overlap_check_normalizes_titles_on_both_sides(tmp_path: Path) -> None:
    """dpr-w100 titles arrive with inconsistent casing and whitespace. Comparing raw strings
    would report zero overlap for two spellings of one article — a passing audit that checked
    nothing."""
    sealed = load_sealed_parents(_sealed(tmp_path, "1960  ELECTION"))
    with pytest.raises(ValueError, match="overlap"):
        _build(sealed=sealed)


def test_a_blank_parent_page_in_the_fingerprint_is_refused(tmp_path: Path) -> None:
    """`read_split_fingerprint` raises on an unresolvable title, so that hole is closed on the
    builder's side. This is the residue a PASSING build could still carry: a blank normalises to
    the empty string, can never match a real title, and reads as a clean row. The fingerprint is
    checked for degeneracy rather than trusted."""
    with pytest.raises(ValueError, match="blank parent page"):
        load_sealed_parents(_sealed(tmp_path, "Real page", "   "))


def test_the_half_refuses_when_the_corpus_builder_is_not_importable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Today's actual state: the corpus line's module has not landed. A missing builder must
    stop the NIAH half, not be caught and downgraded into running without the check."""
    monkeypatch.delitem(sys.modules, "evidence_rag.materializer.sealed600")
    monkeypatch.setattr(
        "evidence_rag.relations.niah_adaptation.importlib.import_module", _raise_import_error
    )
    with pytest.raises(ValueError, match="sealed-600"):
        load_sealed_parents(_sealed(tmp_path, "Any page"))


def test_load_sealed_parents_records_what_it_checked_against(tmp_path: Path) -> None:
    """The manifest has to say WHICH sealed-600 the run was checked against, or "zero overlap"
    is an unfalsifiable claim in the report."""
    sealed = load_sealed_parents(_sealed(tmp_path, "Page A", "Page B"))
    assert sealed.n_parent_pages == 2
    assert sealed.parents == frozenset({"page a", "page b"})
    assert len(sealed.parent_pages_sha256) == 64


def test_both_sides_are_normalized_by_the_one_shared_rule(tmp_path: Path) -> None:
    """Unifying on the corpus line's interface only helps if ONE function puts both sides into
    one space. normalize_parent is idempotent, so re-applying it to already-normalised builder
    output cannot corrupt it, and it guarantees the comparison never spans two spellings."""
    sealed = load_sealed_parents(_sealed(tmp_path, "Barack  OBAMA"))
    assert sealed.parents == frozenset({"barack obama"})


# --- the twin-row target, ruled REFUTES by §3.8(a) -----------------------------------


def test_twin_rows_take_the_declared_three_class_target(tmp_path: Path) -> None:
    """The probe's twin rows carry NOT_SUPPORTED, which A2 made a DERIVED label: a three-class
    head has no logit for it. §3.8(a) rules the target is REFUTES; the flag stays required with
    no default, so the ruling is visible in every manifest rather than implied."""
    sealed = load_sealed_parents(_sealed(tmp_path, "Elsewhere"))
    labels = [row.label for row in _build(sealed=sealed)]
    assert RelationLabel.NOT_SUPPORTED not in labels
    # The probe's four kinds, in order: needle_gold, cf_replacement, cf_gold, needle_replacement.
    assert labels == [
        RelationLabel.SUPPORTS,
        RelationLabel.SUPPORTS,
        RelationLabel.REFUTES,
        RelationLabel.REFUTES,
    ]


def test_unknown_remains_expressible_although_3_8_a_rules_refutes(tmp_path: Path) -> None:
    """§3.8(a) rules REFUTES. UNKNOWN stays accepted because the ruling's own reasoning turns
    on UNKNOWN being trained by VitaminC's NEI class instead, and an amendment revisiting that
    would need this path to exist rather than be re-added under time pressure."""
    sealed = load_sealed_parents(_sealed(tmp_path, "Elsewhere"))
    examples = _build(sealed=sealed, twin_label=RelationLabel.UNKNOWN)
    assert {row.label for row in examples} == {RelationLabel.SUPPORTS, RelationLabel.UNKNOWN}


def test_supports_is_refused_as_the_twin_target(tmp_path: Path) -> None:
    """Every row SUPPORTS is the degenerate arm the 0B-2 joint gate exists to block; training
    it in deliberately would be a way to reach that degenerate model on purpose."""
    sealed = load_sealed_parents(_sealed(tmp_path, "Elsewhere"))
    with pytest.raises(ValueError, match="twin"):
        _build(sealed=sealed, twin_label=RelationLabel.SUPPORTS)


def test_not_supported_is_refused_as_the_twin_target(tmp_path: Path) -> None:
    sealed = load_sealed_parents(_sealed(tmp_path, "Elsewhere"))
    with pytest.raises(ValueError, match="twin"):
        _build(sealed=sealed, twin_label=RelationLabel.NOT_SUPPORTED)


# --- shape of what comes out -------------------------------------------------------------------


def test_group_keys_carry_both_the_parent_page_and_the_synthetic_family(tmp_path: Path) -> None:
    """§3.8 groups the chain by parent page AND synthetic family, so a NIAH row has to expose
    both keys or one of the two axes silently stops constraining the folds."""
    sealed = load_sealed_parents(_sealed(tmp_path, "Elsewhere"))
    keys = set(_build(sealed=sealed)[0].group_keys)
    assert page_key("1960 election") in keys
    assert family_key("kennedy|Nixon|proper_name_1") in keys


def test_examples_use_the_frozen_hypothesis_template(tmp_path: Path) -> None:
    sealed = load_sealed_parents(_sealed(tmp_path, "Elsewhere"))
    hypotheses = {row.hypothesis for row in _build(sealed=sealed)}
    assert 'The answer to the question "who won?" is kennedy.' in hypotheses
    assert 'The answer to the question "who won?" is Nixon.' in hypotheses


def test_a_record_whose_query_is_missing_stops_the_build(tmp_path: Path) -> None:
    """`build_probe_pairs` SKIPS such a record, which is right for a probe whose denominators
    are reported. In a training set a skip silently changes what the model saw and nothing
    downstream records it."""
    sealed = load_sealed_parents(_sealed(tmp_path, "Elsewhere"))
    with pytest.raises(ValueError, match="incomplete"):
        _build(sealed=sealed, question_by_query={})


def test_every_example_is_tagged_as_the_niah_source(tmp_path: Path) -> None:
    """The manifest reports the two halves separately; a mistagged row would make a
    VitaminC-only run look like it had done domain adaptation."""
    sealed = load_sealed_parents(_sealed(tmp_path, "Elsewhere"))
    assert {row.source for row in _build(sealed=sealed)} == {"niah"}
