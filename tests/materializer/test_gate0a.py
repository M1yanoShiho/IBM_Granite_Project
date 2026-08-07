import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from evidence_rag.contracts.models import (
    CandidateSet,
    Document,
    EvidenceCandidate,
    Query,
    RetrieverProvenance,
)
from evidence_rag.infrastructure.datasets import GoldCase, JsonlDatasetAdapter
from evidence_rag.materializer.answer_bank import build_answer_bank
from evidence_rag.materializer.gate0a import (
    CheckReport,
    audit_axes,
    audit_candidates,
    audit_counterfactuals,
    audit_label_provenance,
    audit_manifest_freeze,
    audit_size_derivation,
    gate0a_report,
    not_applicable,
    unevaluated,
)
from evidence_rag.materializer.injector import find_injection_target, inject_counterfactual
from evidence_rag.materializer.provenance import MutationRecord, read_provenance
from evidence_rag.materializer.sealed600 import (
    DEV_SAMPLE_INJECTED,
    DEV_SAMPLE_QUERIES,
    DEV_SKIP_RATE,
    FROZEN_RETRIEVER,
    POOL_SIZE,
    PROTOCOL_VERSION,
    SEALED_MANIFEST_FILE,
    TARGET_INJECTED,
    CandidateFreeze,
    SealedManifest,
    SplitFingerprint,
    candidate_retrievers,
    fingerprint_bundle,
    freeze_candidate_pin,
    freeze_sealed_manifest,
    read_candidate_pin,
    read_sealed_manifest,
    sha256_file,
    sole_retriever,
    write_sealed_dataset,
)

PROTOCOL_HASH = "d" * 64
BM25 = RetrieverProvenance(
    name=FROZEN_RETRIEVER, implementation_version="bm25-v1", parameters_sha256="1" * 64
)
STRONG_BM25 = RetrieverProvenance(
    name="strong-bm25", implementation_version="strong-bm25-v1", parameters_sha256="2" * 64
)


def doc(document_id: str, title: str, body: str) -> Document:
    return Document(
        document_id=document_id,
        text=f"{title}\n\n{body}",
        source_uri=f"ir-datasets://dpr-w100/nq/document/{document_id}",
    )


def build_sealed(tmp_path: Path) -> Path:
    """A two-question sealed set built by the real injector, so its hashes are real."""
    directory = tmp_path / "sealed"
    documents = {
        "d1": doc("d1", "Titanic", "The ship sank in 1912 off Newfoundland."),
        "d2": doc("d2", "Hindenburg", "The airship burned in 1937 at Lakehurst."),
        "d3": doc("d3", "Distractor", "Nothing relevant is written here."),
    }
    gold_cases = (
        GoldCase(query_id="q1", relevant_document_ids=("d1",), reference_answers=("1912",)),
        GoldCase(query_id="q2", relevant_document_ids=("d2",), reference_answers=("1937",)),
    )
    queries = (
        Query(query_id="q1", text="When did the Titanic sink?"),
        Query(query_id="q2", text="When did the Hindenburg burn?"),
    )
    bank = build_answer_bank(["1800", "1801", "1802"], seed=42)
    twins: list[Document] = []
    records: list[MutationRecord] = []
    for gold_case in gold_cases:
        target = find_injection_target(gold_case, documents)
        assert target is not None
        injection = inject_counterfactual(target, bank, seed=42)
        assert injection is not None
        twins.append(injection[0])
        records.append(injection[1])
    hashes = write_sealed_dataset(
        directory,
        documents=tuple(documents.values()) + tuple(twins),
        queries=queries,
        gold_cases=gold_cases,
        records=tuple(records),
    )
    freeze_sealed_manifest(directory, a_manifest(artifact_sha256=hashes))
    return directory


def a_manifest(**overrides: object) -> SealedManifest:
    payload: dict[str, object] = {
        "protocol_version": "g2-proto-4",
        "protocol_document_sha256": PROTOCOL_HASH,
        "dataset_id": "niah/dpr-w100-nq-sealed600",
        "dataset_version": "sealed600-seed42",
        "split": "sealed-test",
        "source_dataset_id": "dpr-w100/natural-questions/dev",
        "seed": 42,
        "corpus_size": 3,
        "target_injected": 2,
        "dev_sample_queries": DEV_SAMPLE_QUERIES,
        "dev_sample_injected": DEV_SAMPLE_INJECTED,
        "dev_skip_rate": DEV_SKIP_RATE,
        "required_pool_size": 3,
        "pool_size": 3,
        "n_considered": 10,
        "n_injected": 2,
        "answer_bank_hash": "e" * 64,
        "answer_bank_source": "runs/niah-train-injected/manifest.json",
        "existing_splits": ("runs/niah-injected",),
        "label_provenance": {
            "gold_cases.jsonl": "official",
            "provenance.jsonl": "deterministic_rule",
        },
        "pool_rejections": {},
        "injection_rejections": {},
        "pool_query_ids": ("q1", "q2", "q3"),
        "query_ids": ("q1", "q2"),
        "artifact_sha256": {},
    }
    payload.update(overrides)
    return SealedManifest.model_validate(payload)


def load(directory: Path) -> tuple[object, tuple[MutationRecord, ...]]:
    bundle = JsonlDatasetAdapter.load(directory / "manifest.json")
    return bundle, read_provenance(directory / "provenance.jsonl")


# ---------------------------------------------------------------- the five axes


def sealed_fingerprint(directory: Path) -> SplitFingerprint:
    bundle, records = load(directory)
    return fingerprint_bundle("sealed", bundle, records)  # type: ignore[arg-type]


def full_existing(**overrides: frozenset[str]) -> SplitFingerprint:
    """An existing split that is non-empty on every axis, so a test can plant ONE collision
    without tripping the vacuity guard on the other four."""
    base: dict[str, frozenset[str]] = {
        "query_ids": frozenset({"other"}),
        "query_texts": frozenset({"other question"}),
        "parent_pages": frozenset({"other page"}),
        "answer_entities": frozenset({"other answer"}),
        "passage_hashes": frozenset({"0" * 64}),
        "synthetic_families": frozenset({"a|b|year"}),
    }
    base.update(overrides)
    return SplitFingerprint(name="dev", **base)


def test_a_clean_sealed_set_passes_all_five_axes(tmp_path: Path) -> None:
    axes = audit_axes(sealed_fingerprint(build_sealed(tmp_path)), (full_existing(),))
    assert [axis.axis for axis in axes] == [
        "query",
        "parent_page",
        "answer_entity",
        "passage_hash",
        "synthetic_family",
    ]
    assert all(axis.failure is None for axis in axes)


@pytest.mark.parametrize(
    ("axis", "overrides"),
    [
        ("query", {"query_ids": frozenset({"q1"})}),
        ("query", {"query_texts": frozenset({"when did the titanic sink"})}),
        ("parent_page", {"parent_pages": frozenset({"titanic"})}),
        ("answer_entity", {"answer_entities": frozenset({"1912"})}),
    ],
)
def test_each_axis_catches_a_planted_overlap(
    tmp_path: Path, axis: str, overrides: dict[str, frozenset[str]]
) -> None:
    axes = audit_axes(sealed_fingerprint(build_sealed(tmp_path)), (full_existing(**overrides),))
    failed = {item.axis for item in axes if item.failure is not None}
    assert failed == {axis}


def test_the_passage_hash_axis_catches_a_reused_gold_passage(tmp_path: Path) -> None:
    sealed = sealed_fingerprint(build_sealed(tmp_path))
    existing = full_existing(passage_hashes=frozenset(sealed.passage_hashes))
    failed = {item.axis for item in audit_axes(sealed, (existing,)) if item.failure is not None}
    assert failed == {"passage_hash"}


def test_the_synthetic_family_axis_catches_a_reused_contrast(tmp_path: Path) -> None:
    sealed = sealed_fingerprint(build_sealed(tmp_path))
    existing = full_existing(synthetic_families=frozenset(sealed.synthetic_families))
    failed = {item.axis for item in audit_axes(sealed, (existing,)) if item.failure is not None}
    assert failed == {"synthetic_family"}


def test_an_axis_with_nothing_on_the_existing_side_fails_instead_of_passing(
    tmp_path: Path,
) -> None:
    """The failure mode this whole module exists for. An empty comparison set never collides,
    so the axis would report zero overlap and the gate would read PASS — a plausible number
    produced by a broken input rather than by a clean set. gate0b.py takes the same line on an
    absent class: count 0 and FAIL, because that is a split error, not "nothing to measure".
    """
    axes = audit_axes(sealed_fingerprint(build_sealed(tmp_path)), (SplitFingerprint("empty"),))
    assert all(axis.failure is not None for axis in axes)
    assert "vacuous" in (axes[0].failure or "")


def test_an_axis_with_nothing_on_the_sealed_side_fails_too() -> None:
    sealed = SplitFingerprint("sealed")
    axes = audit_axes(sealed, (full_existing(),))
    assert all(axis.failure is not None for axis in axes)


def test_no_existing_split_at_all_is_refused(tmp_path: Path) -> None:
    """An audit against nothing is the cheapest way to get five green axes."""
    with pytest.raises(ValueError, match="existing"):
        audit_axes(sealed_fingerprint(build_sealed(tmp_path)), ())


def test_every_existing_split_contributes_to_the_union(tmp_path: Path) -> None:
    sealed = sealed_fingerprint(build_sealed(tmp_path))
    existing = (full_existing(), full_existing(query_ids=frozenset({"q2"})))
    failed = {item.axis for item in audit_axes(sealed, existing) if item.failure is not None}
    assert failed == {"query"}


# ---------------------------------------------------------------- size derivation


def test_the_size_derivation_is_recomputed_from_the_manifests_own_numbers() -> None:
    """M0 §6 audits the frozen manifest, not the code that wrote it. Recomputing 812 from the
    recorded dev counts catches a manifest written by an older or patched builder."""
    check = audit_size_derivation(
        a_manifest(
            required_pool_size=812, pool_size=POOL_SIZE, target_injected=TARGET_INJECTED,
            n_injected=TARGET_INJECTED, query_ids=tuple(f"q{i}" for i in range(TARGET_INJECTED)),
            pool_query_ids=tuple(f"p{i}" for i in range(POOL_SIZE)),
        )
    )
    assert check.status == "pass"


def test_a_manifest_whose_pool_lacks_headroom_fails_the_derivation() -> None:
    check = audit_size_derivation(
        a_manifest(
            target_injected=TARGET_INJECTED,
            n_injected=TARGET_INJECTED,
            pool_size=700,
            required_pool_size=812,
            query_ids=tuple(f"q{i}" for i in range(TARGET_INJECTED)),
            pool_query_ids=tuple(f"p{i}" for i in range(700)),
        )
    )
    assert check.status == "fail"


def test_a_manifest_whose_query_list_does_not_match_its_own_count_fails() -> None:
    """The list and the count are two statements of one fact; if they disagree, every
    downstream denominator has two candidate values."""
    assert audit_size_derivation(a_manifest(n_injected=3)).status == "fail"


def test_a_skip_rate_inconsistent_with_the_recorded_dev_counts_fails() -> None:
    assert audit_size_derivation(a_manifest(dev_skip_rate=0.5)).status == "fail"


# ---------------------------------------------------------------- freeze / hashes


def test_the_freeze_check_passes_on_an_untouched_directory(tmp_path: Path) -> None:
    directory = build_sealed(tmp_path)
    manifest = read_sealed_manifest(directory)
    assert audit_manifest_freeze(directory, manifest, PROTOCOL_HASH).status == "pass"


def test_the_freeze_check_fails_when_an_artefact_was_edited(tmp_path: Path) -> None:
    directory = build_sealed(tmp_path)
    manifest = read_sealed_manifest(directory)
    (directory / "queries.jsonl").write_text("", encoding="utf-8")
    assert audit_manifest_freeze(directory, manifest, PROTOCOL_HASH).status == "fail"


def test_the_freeze_check_fails_on_a_different_protocol_document(tmp_path: Path) -> None:
    """The sealed set was frozen against one text of M0. Auditing it against another means the
    rules moved after the freeze, which is the situation §0 forbids."""
    directory = build_sealed(tmp_path)
    manifest = read_sealed_manifest(directory)
    assert audit_manifest_freeze(directory, manifest, "9" * 64).status == "fail"


def test_the_freeze_check_fails_when_an_artefact_is_missing_from_the_record(
    tmp_path: Path,
) -> None:
    """Hashing only what the manifest happens to list would let an unrecorded file ride along
    unaudited, so the recorded set and the directory's data files must be the same set."""
    directory = build_sealed(tmp_path)
    manifest = read_sealed_manifest(directory)
    (directory / "extra.jsonl").write_text("{}\n", encoding="utf-8")
    assert audit_manifest_freeze(directory, manifest, PROTOCOL_HASH).status == "fail"


# ---------------------------------------------------------------- label provenance


def test_label_provenance_passes_for_official_qrels_plus_the_mutation_log(
    tmp_path: Path,
) -> None:
    directory = build_sealed(tmp_path)
    bundle, records = load(directory)
    check = audit_label_provenance(read_sealed_manifest(directory), bundle, records)
    assert check.status == "pass"


def test_a_sealed_query_without_an_official_answer_fails_provenance(tmp_path: Path) -> None:
    """A query whose gold answers are absent has no primary label at all, and M0 §6 allows
    exactly two sources — "none" is not one of them."""
    directory = build_sealed(tmp_path)
    (directory / "gold_cases.jsonl").write_text(
        GoldCase(query_id="q1", relevant_document_ids=("d1",)).model_dump_json()
        + "\n"
        + GoldCase(
            query_id="q2", relevant_document_ids=("d2",), reference_answers=("1937",)
        ).model_dump_json()
        + "\n",
        encoding="utf-8",
    )
    bundle, records = load(directory)
    assert audit_label_provenance(read_sealed_manifest(directory), bundle, records).status == "fail"


def test_a_manifest_that_forgets_to_declare_a_label_file_fails(tmp_path: Path) -> None:
    directory = build_sealed(tmp_path)
    bundle, records = load(directory)
    manifest = a_manifest(label_provenance={"gold_cases.jsonl": "official"})
    assert audit_label_provenance(manifest, bundle, records).status == "fail"


# ---------------------------------------------------------------- counterfactuals


def test_counterfactuals_are_reversible_with_no_residual_gold_alias(tmp_path: Path) -> None:
    directory = build_sealed(tmp_path)
    bundle, records = load(directory)
    assert audit_counterfactuals(bundle, records).status == "pass"


def test_a_twin_that_no_longer_reverses_to_its_needle_fails(tmp_path: Path) -> None:
    """M0 §6 requires the mutation to be reversible from the log alone. If it is not, the
    counterfactual is no longer a controlled one-span edit and the harmful label stops meaning
    "the same passage with one fact changed"."""
    directory = build_sealed(tmp_path)
    bundle, records = load(directory)
    tampered = tuple(
        Document(
            document_id=document.document_id,
            text=document.text + " Extra sentence.",
            source_uri=document.source_uri,
        )
        if document.document_id.startswith("cf::")
        else document
        for document in bundle.documents  # type: ignore[attr-defined]
    )
    assert audit_counterfactuals(bundle.model_copy(update={"documents": tampered}), records).status == "fail"  # type: ignore[attr-defined]


def test_a_twin_that_still_contains_a_gold_alias_fails(tmp_path: Path) -> None:
    """"所有 gold alias 均不残留" — a twin still carrying the gold string is not a
    counterfactual, and a selector that keeps it would be scored as harmful for holding a
    passage that is in fact correct."""
    directory = build_sealed(tmp_path)
    bundle, records = load(directory)
    residual = tuple(
        Document(
            document_id=document.document_id,
            text=document.text.replace("Newfoundland", "1912"),
            source_uri=document.source_uri,
        )
        if document.document_id.startswith("cf::q1")
        else document
        for document in bundle.documents  # type: ignore[attr-defined]
    )
    report = audit_counterfactuals(bundle.model_copy(update={"documents": residual}), records)  # type: ignore[attr-defined]
    assert report.status == "fail"
    assert "alias" in report.detail


def test_a_synthetic_document_with_no_mutation_record_fails(tmp_path: Path) -> None:
    """Every cf:: document must be named by the log. One that is not was written by something
    other than the injector, and nothing downstream would tell it apart from a real twin."""
    directory = build_sealed(tmp_path)
    bundle, records = load(directory)
    extra = (*bundle.documents, doc("cf::q9::d9", "Ghost", "Unlogged synthetic passage."))  # type: ignore[attr-defined]
    assert audit_counterfactuals(bundle.model_copy(update={"documents": extra}), records).status == "fail"  # type: ignore[attr-defined]


def test_a_sealed_query_without_a_mutation_record_fails(tmp_path: Path) -> None:
    directory = build_sealed(tmp_path)
    bundle, records = load(directory)
    assert audit_counterfactuals(bundle, records[:1]).status == "fail"


# ---------------------------------------------------------------- candidates (Top-20)


def candidates_for(
    query_id: str,
    document_ids: tuple[str, ...],
    retriever: RetrieverProvenance | None = BM25,
) -> CandidateSet:
    return CandidateSet(
        query_id=query_id,
        retriever=retriever,
        candidates=tuple(
            EvidenceCandidate(
                evidence_id=f"{query_id}:{document_id}",
                document_id=document_id,
                chunk_id=document_id,
                text="text",
                source_uri=f"fixture://{document_id}",
                retrieval_score=1.0 / (rank + 1),
                retrieval_rank=rank + 1,
            )
            for rank, document_id in enumerate(document_ids)
        ),
    )


def clean_windows(retriever: RetrieverProvenance | None = BM25) -> tuple[CandidateSet, ...]:
    return (
        candidates_for("q1", ("d1", "cf::q1::d1"), retriever),
        candidates_for("q2", ("d2", "cf::q2::d2"), retriever),
    )


CANDIDATE_HASH = "c" * 64


def a_pin(**overrides: object) -> CandidateFreeze:
    payload: dict[str, object] = {
        "protocol_version": PROTOCOL_VERSION,
        "sealed_manifest_sha256": "f" * 64,
        "candidate_file": "runs/sealed600-bm25/candidate_sets.jsonl",
        "candidate_sha256": CANDIDATE_HASH,
        "n_windows": 2,
        "top_n": 2,
        "retriever": BM25,
    }
    payload.update(overrides)
    return CandidateFreeze.model_validate(payload)


def audit(
    bundle: object,
    records: Sequence[MutationRecord],
    windows: Sequence[CandidateSet],
    *,
    top_n: int = 2,
    pin: CandidateFreeze | None = None,
    candidate_sha256: str = CANDIDATE_HASH,
) -> CheckReport:
    return audit_candidates(
        bundle,  # type: ignore[arg-type]
        records,
        windows,
        top_n=top_n,
        pin=pin,
        candidate_sha256=candidate_sha256,
    )


def test_candidate_windows_of_the_right_shape_are_still_incomplete_until_they_are_pinned(
    tmp_path: Path,
) -> None:
    """Shape was the whole check before. It is no longer sufficient: with no pin, nothing stops
    the NEXT run from handing §3.5 and §5.2 a different set of windows, and the thresholds those
    sections pre-registered were measured on one pool."""
    directory = build_sealed(tmp_path)
    bundle, records = load(directory)
    report = audit(bundle, records, clean_windows())
    assert report.status == "unevaluated"
    assert "pin" in report.detail


def test_the_check_passes_only_when_shape_provenance_and_the_pinned_hash_all_agree(
    tmp_path: Path,
) -> None:
    directory = build_sealed(tmp_path)
    bundle, records = load(directory)
    report = audit(bundle, records, clean_windows(), pin=a_pin())
    assert report.status == "pass"
    assert FROZEN_RETRIEVER in report.detail


def test_a_query_with_the_wrong_candidate_count_fails(tmp_path: Path) -> None:
    """"每题 candidate count 一致" — a short window silently changes that query's denominators
    for recall and for harmful exposure, and nothing downstream reports the window size."""
    directory = build_sealed(tmp_path)
    bundle, records = load(directory)
    windows = (candidates_for("q1", ("d1",)), candidates_for("q2", ("d2", "cf::q2::d2")))
    assert audit(bundle, records, windows, pin=a_pin()).status == "fail"


def test_a_missing_query_window_fails(tmp_path: Path) -> None:
    directory = build_sealed(tmp_path)
    bundle, records = load(directory)
    windows = (candidates_for("q1", ("d1", "cf::q1::d1")),)
    assert audit(bundle, records, windows, pin=a_pin(n_windows=1)).status == "fail"


def test_a_candidate_pointing_outside_the_sealed_corpus_fails(tmp_path: Path) -> None:
    directory = build_sealed(tmp_path)
    bundle, records = load(directory)
    windows = (
        candidates_for("q1", ("d1", "ghost")),
        candidates_for("q2", ("d2", "cf::q2::d2")),
    )
    assert audit(bundle, records, windows, pin=a_pin()).status == "fail"


def test_a_foreign_twin_in_a_window_is_reported_but_does_not_fail(tmp_path: Path) -> None:
    """Every twin lives in one shared corpus, so BM25 can return query 2's twin for query 1.
    That is pool composition, not a construction error — the harmful label keys on the query's
    OWN twin. It is counted so the number is visible rather than argued about later."""
    directory = build_sealed(tmp_path)
    bundle, records = load(directory)
    windows = (
        candidates_for("q1", ("d1", "cf::q2::d2")),
        candidates_for("q2", ("d2", "cf::q2::d2")),
    )
    report = audit(bundle, records, windows, pin=a_pin())
    assert report.status == "pass"
    assert "foreign_twin" in report.detail


# ------------------------------------------------- candidates (retriever provenance, M0 §4)


def test_a_pool_that_names_no_retriever_is_unevaluated_and_is_never_read_as_bm25(
    tmp_path: Path,
) -> None:
    """THE design question, answered where someone will hit it. A candidate file written before
    provenance existed passes every shape check identically to a bm25 pool — that is exactly the
    defect. Assuming bm25 would be the audit making the claim the audit exists to verify, so the
    item goes to `unevaluated`: the verdict becomes INCOMPLETE, which is not a pass, and the
    reason names what has to happen instead."""
    directory = build_sealed(tmp_path)
    bundle, records = load(directory)
    report = audit(bundle, records, clean_windows(retriever=None))
    assert report.status == "unevaluated"
    assert "NOT assumed" in report.detail
    assert FROZEN_RETRIEVER in report.detail


def test_an_unpinned_provenance_less_pool_never_reaches_pass(tmp_path: Path) -> None:
    """The point of the previous test, stated as the property that matters: whatever the audit
    does with an old file, it must not be able to print PASS."""
    directory = build_sealed(tmp_path)
    bundle, records = load(directory)
    axes = audit_axes(sealed_fingerprint(directory), (full_existing(),))
    report = gate0a_report(
        axes=axes, checks=(audit(bundle, records, clean_windows(retriever=None)),)
    )
    assert report.verdict == "INCOMPLETE"
    assert not report.passes


def test_a_pool_built_by_a_different_retriever_fails(tmp_path: Path) -> None:
    """M0 §4 freezes the retriever because the Graph 2.0 claim is conditional on a FIXED
    candidate pool. A strong-bm25 pool has the same shape and entirely plausible numbers, and it
    would silently move §3.5's 0.4355 baseline and §5.2's gate-off reference onto a pool they
    were never measured on."""
    directory = build_sealed(tmp_path)
    bundle, records = load(directory)
    report = audit(bundle, records, clean_windows(retriever=STRONG_BM25), pin=a_pin())
    assert report.status == "fail"
    assert "strong-bm25" in report.detail


def test_the_wrong_retriever_is_reported_even_when_the_shape_problems_fill_the_detail(
    tmp_path: Path,
) -> None:
    """The report truncates to the first few problems, and a 600-query pool from the wrong
    retriever also has 600 chances to trip a shape rule. The §4 violation is the expensive one,
    so it must not be the line that gets cut."""
    directory = build_sealed(tmp_path)
    bundle, records = load(directory)
    windows = (
        candidates_for("q1", ("d1", "ghost-a", "ghost-b"), STRONG_BM25),
        candidates_for("q2", ("d2", "ghost-c", "ghost-d"), STRONG_BM25),
    )
    report = audit(bundle, records, windows, pin=a_pin())
    assert report.status == "fail"
    assert "more)" in report.detail  # the detail really was truncated
    assert "strong-bm25" in report.detail


def test_a_pool_naming_two_retrievers_fails_rather_than_picking_one(tmp_path: Path) -> None:
    """A file whose windows disagree about their producer was assembled from two runs. Half a
    pool from each retriever is the confound §4 forbids, in its worst form: within-run pairing
    still looks internally valid."""
    directory = build_sealed(tmp_path)
    bundle, records = load(directory)
    windows = (
        candidates_for("q1", ("d1", "cf::q1::d1"), BM25),
        candidates_for("q2", ("d2", "cf::q2::d2"), STRONG_BM25),
    )
    report = audit(bundle, records, windows, pin=a_pin())
    assert report.status == "fail"
    assert "one retrieval run" in report.detail


def test_a_pool_half_of_which_names_no_retriever_fails_rather_than_being_unevaluated(
    tmp_path: Path,
) -> None:
    """Absent-everywhere is an old file; absent-in-half is a file someone edited. The second is
    a positive finding, not a gap, so it fails instead of downgrading to INCOMPLETE."""
    directory = build_sealed(tmp_path)
    bundle, records = load(directory)
    windows = (
        candidates_for("q1", ("d1", "cf::q1::d1"), BM25),
        candidates_for("q2", ("d2", "cf::q2::d2"), None),
    )
    assert audit(bundle, records, windows).status == "fail"


def test_a_pin_standing_over_a_pool_that_names_no_producer_fails(tmp_path: Path) -> None:
    """`pin_candidates` refuses to pin an unstamped pool, so this combination cannot be produced
    by the tooling. It means the pinned file was replaced by an older one."""
    directory = build_sealed(tmp_path)
    bundle, records = load(directory)
    report = audit(bundle, records, clean_windows(retriever=None), pin=a_pin())
    assert report.status == "fail"


def test_an_empty_candidate_file_fails_instead_of_reporting_nothing_wrong(
    tmp_path: Path,
) -> None:
    """The vacuity guard, on this axis. Zero windows means zero shape violations and zero
    disagreeing retrievers — a green board produced by an audit that could not look."""
    directory = build_sealed(tmp_path)
    bundle, records = load(directory)
    assert audit(bundle, records, (), pin=a_pin(n_windows=0)).status == "fail"


# ------------------------------------------------- candidates (the pinned hash, M0 §4)


def test_a_pool_whose_bytes_differ_from_the_pinned_hash_fails(tmp_path: Path) -> None:
    """Same retriever, different windows. Two runs of bm25 over two corpus builds, or a re-run
    after an unrelated fix, produce pools that both audit clean on shape and provenance — and
    the pre-registered thresholds were measured on exactly one of them."""
    directory = build_sealed(tmp_path)
    bundle, records = load(directory)
    report = audit(bundle, records, clean_windows(), pin=a_pin(), candidate_sha256="9" * 64)
    assert report.status == "fail"
    assert "sha256" in report.detail


def test_a_pin_recording_a_different_retriever_than_the_pool_fails(tmp_path: Path) -> None:
    directory = build_sealed(tmp_path)
    bundle, records = load(directory)
    report = audit(bundle, records, clean_windows(), pin=a_pin(retriever=STRONG_BM25))
    assert report.status == "fail"


def test_a_pin_recording_a_different_window_count_fails(tmp_path: Path) -> None:
    directory = build_sealed(tmp_path)
    bundle, records = load(directory)
    assert audit(bundle, records, clean_windows(), pin=a_pin(n_windows=600)).status == "fail"


def test_a_pin_recording_a_different_top_n_fails(tmp_path: Path) -> None:
    """The pin says which window size the pool was frozen at. Auditing at another one would
    compare a Top-20 claim against a Top-10 pool."""
    directory = build_sealed(tmp_path)
    bundle, records = load(directory)
    assert audit(bundle, records, clean_windows(), pin=a_pin(top_n=20)).status == "fail"


# ------------------------------------------------- the pin as a freeze record


def test_the_pin_is_written_once_and_a_second_pin_is_refused(tmp_path: Path) -> None:
    """Same reasoning as `freeze_sealed_manifest`: silently replacing the pin is what "we re-ran
    retrieval and re-pinned" looks like from the outside, and that is the confound, not the fix."""
    directory = build_sealed(tmp_path)
    freeze_candidate_pin(directory, a_pin())
    with pytest.raises(ValueError, match="already pinned"):
        freeze_candidate_pin(directory, a_pin(candidate_sha256="9" * 64))


def test_an_unpinned_sealed_set_reads_back_as_no_pin_rather_than_raising(
    tmp_path: Path,
) -> None:
    """The pre-retrieval state is legitimate and is the state the sealed 600 is in today."""
    assert read_candidate_pin(build_sealed(tmp_path)) is None


def test_the_pin_round_trips(tmp_path: Path) -> None:
    directory = build_sealed(tmp_path)
    pin = a_pin(sealed_manifest_sha256=sha256_file(directory / SEALED_MANIFEST_FILE))
    freeze_candidate_pin(directory, pin)
    assert read_candidate_pin(directory) == pin


def test_the_freeze_check_tolerates_the_pin_but_not_any_other_new_file(tmp_path: Path) -> None:
    """`candidate_freeze.json` is the one file that is SUPPOSED to appear after the freeze —
    it cannot exist before retrieval has run. It is exempt from `artifact_sha256` and audited by
    `candidate_windows` instead; nothing else gets that exemption."""
    directory = build_sealed(tmp_path)
    manifest = read_sealed_manifest(directory)
    freeze_candidate_pin(
        directory, a_pin(sealed_manifest_sha256=sha256_file(directory / SEALED_MANIFEST_FILE))
    )
    assert audit_manifest_freeze(directory, manifest, PROTOCOL_HASH).status == "pass"


def test_a_pin_bound_to_a_different_sealed_manifest_fails_the_freeze_check(
    tmp_path: Path,
) -> None:
    """A pin copied in from another sealed set would otherwise certify THIS set's pool using a
    hash measured against a different 600 questions."""
    directory = build_sealed(tmp_path)
    manifest = read_sealed_manifest(directory)
    freeze_candidate_pin(directory, a_pin(sealed_manifest_sha256="0" * 64))
    report = audit_manifest_freeze(directory, manifest, PROTOCOL_HASH)
    assert report.status == "fail"
    assert "candidate_freeze.json" in report.detail


# ------------------------------------------------- the producer resolver


def test_the_distinct_producers_of_a_pool_are_reported_in_order() -> None:
    assert candidate_retrievers(clean_windows()) == (BM25,)
    assert candidate_retrievers(clean_windows(retriever=None)) == (None,)
    assert candidate_retrievers(()) == ()


def test_sole_retriever_raises_on_a_pool_that_names_none() -> None:
    with pytest.raises(ValueError, match="names no retriever"):
        sole_retriever(clean_windows(retriever=None))


def test_sole_retriever_raises_on_an_empty_pool() -> None:
    with pytest.raises(ValueError, match="no candidate windows"):
        sole_retriever(())


# ---------------------------------------------------------------- verdict assembly


def test_an_unevaluated_check_makes_the_verdict_incomplete_not_pass(tmp_path: Path) -> None:
    """M0 §6 is a zero-violation checklist. A run that never looked at one item has not passed
    it, and printing PASS with a footnote is how a checklist rots — this project has written
    that sentence four times (§9.11, §10.8, §11.9a)."""
    axes = audit_axes(sealed_fingerprint(build_sealed(tmp_path)), (full_existing(),))
    report = gate0a_report(
        axes=axes, checks=(unevaluated("candidates", "retrieval has not run yet"),)
    )
    assert report.verdict == "INCOMPLETE"
    assert not report.passes


def test_a_not_applicable_check_still_passes_but_is_named_in_the_verdict(
    tmp_path: Path,
) -> None:
    """`utility_grade` has no producer under D1=A. That is a fact about the protocol, not a
    skipped check, so it passes — but the reason rides in the report rather than disappearing."""
    axes = audit_axes(sealed_fingerprint(build_sealed(tmp_path)), (full_existing(),))
    report = gate0a_report(
        axes=axes,
        checks=(not_applicable("utility_range", "no producer under D1=A (M0 §1)"),),
    )
    assert report.verdict == "PASS"
    assert report.passes
    assert "utility_range" in json.dumps(report.payload())


def test_a_failed_axis_outranks_an_unevaluated_check(tmp_path: Path) -> None:
    axes = audit_axes(sealed_fingerprint(build_sealed(tmp_path)), (SplitFingerprint("empty"),))
    report = gate0a_report(axes=axes, checks=(unevaluated("candidates", "not run"),))
    assert report.verdict == "FAIL"
