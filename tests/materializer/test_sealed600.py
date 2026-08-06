from hashlib import sha256
from pathlib import Path

import pytest

from evidence_rag.contracts.models import Document, Query
from evidence_rag.infrastructure.datasets import DatasetManifest, GoldCase
from evidence_rag.materializer.answer_bank import build_answer_bank
from evidence_rag.materializer.provenance import MutationRecord, write_provenance
from evidence_rag.materializer.sealed600 import (
    DEV_SAMPLE_INJECTED,
    DEV_SAMPLE_QUERIES,
    DEV_SKIP_RATE,
    POOL_SIZE,
    TARGET_INJECTED,
    SealedManifest,
    SourceQuery,
    SplitFingerprint,
    assert_pool_headroom,
    evaluate_pool,
    freeze_sealed_manifest,
    frozen_order,
    inject_pool,
    normalize_query_text,
    read_sealed_manifest,
    read_split_fingerprint,
    required_pool_size,
    select_pool,
    sha256_file,
    verify_artifacts,
)


def doc(document_id: str, title: str, body: str) -> Document:
    """dpr-w100 shape: the article title is the first paragraph (base_loader.document_text)."""
    return Document(
        document_id=document_id,
        text=f"{title}\n\n{body}",
        source_uri=f"fixture://{document_id}",
    )


def candidate(
    query_id: str,
    *,
    text: str = "who?",
    answers: tuple[str, ...] = ("1912",),
    gold: tuple[str, ...] = ("d1",),
) -> SourceQuery:
    return SourceQuery(query_id=query_id, text=text, answers=answers, gold_document_ids=gold)


def blank_split(name: str = "dev", **overrides: frozenset[str]) -> SplitFingerprint:
    return SplitFingerprint(name=name, **overrides)


def one_document() -> dict[str, Document]:
    return {"d1": doc("d1", "Titanic", "The ship sank in 1912.")}


# ---------------------------------------------------------------- size derivation (M0 §4)


def test_the_pool_size_is_derived_from_the_measured_dev_skip_rate() -> None:
    """M0 §4 fixes 900 by arithmetic, not by taste: 600/(1-.261) = 812 eligible queries.

    The number is recomputed here rather than asserted as a literal so that changing any input
    (target, skip rate) without changing the frozen pool size fails immediately instead of
    quietly shrinking the headroom.
    """
    assert required_pool_size(target=TARGET_INJECTED, skip_rate=DEV_SKIP_RATE) == 812
    assert POOL_SIZE == 900


def test_the_recorded_skip_rate_matches_the_recorded_dev_counts() -> None:
    """.261 is 1 - 1479/2000. A typo in either constant would still produce a plausible pool
    size, so the two are pinned against each other."""
    assert abs((1 - DEV_SAMPLE_INJECTED / DEV_SAMPLE_QUERIES) - DEV_SKIP_RATE) < 0.001


def test_a_pool_without_headroom_is_refused_before_any_data_is_touched() -> None:
    """The protocol forbids discovering mid-run that the pool was too small (补样本 = 看结果后
    改数据), so the arithmetic is checked up front and raises."""
    with pytest.raises(ValueError, match="headroom"):
        assert_pool_headroom(pool_size=700, target=TARGET_INJECTED, skip_rate=DEV_SKIP_RATE)


def test_the_frozen_pool_size_has_headroom_over_the_derivation() -> None:
    assert assert_pool_headroom() == 812


# ---------------------------------------------------------------- frozen ordering


def test_the_frozen_order_is_a_pure_function_of_seed_and_ids() -> None:
    """The order that decides WHICH 600 of the injected pool get sealed must reproduce on
    another machine and another interpreter. random.shuffle does not: its stream is a CPython
    implementation detail that has changed before. A hash of (seed, id) cannot."""
    ids = ["q3", "q1", "q2", "q10"]
    assert frozen_order(ids, seed=42) == frozen_order(reversed(ids), seed=42)
    assert frozen_order(ids, seed=42) != frozen_order(ids, seed=13)
    assert sorted(frozen_order(ids, seed=42)) == sorted(ids)


def test_the_frozen_order_rejects_duplicate_ids() -> None:
    """A duplicated query id would be sealed twice and counted once — every downstream paired
    denominator would then disagree with the manifest."""
    with pytest.raises(ValueError, match="duplicate"):
        frozen_order(["q1", "q1"], seed=42)


# ---------------------------------------------------------------- query text normalisation


def test_query_text_normalisation_ignores_case_spacing_and_the_question_mark() -> None:
    """Axis 1 is "query_id AND normalised text": the same question re-asked under a new id is
    exactly the leak this axis exists to catch, so punctuation must not defeat the comparison."""
    assert normalize_query_text("Who  sank the  Titanic?") == normalize_query_text(
        "who sank the titanic"
    )


# ---------------------------------------------------------------- pool eligibility, five axes


def reasons(query: SourceQuery, existing: SplitFingerprint, **kwargs: object) -> dict[str, int]:
    documents = kwargs.get("documents") or one_document()
    assert isinstance(documents, dict)
    evaluation = evaluate_pool(
        candidates=(query,), documents=documents, existing=(existing,)
    )
    return dict(evaluation.rejections)


def eligible(query: SourceQuery, existing: SplitFingerprint, **kwargs: object) -> tuple[str, ...]:
    documents = kwargs.get("documents") or one_document()
    assert isinstance(documents, dict)
    return evaluate_pool(
        candidates=(query,), documents=documents, existing=(existing,)
    ).eligible_query_ids


def test_a_query_id_already_used_by_an_existing_split_is_rejected() -> None:
    existing = blank_split(query_ids=frozenset({"q1"}))
    assert eligible(candidate("q1"), existing) == ()
    assert reasons(candidate("q1"), existing) == {"leak_query_id": 1}


def test_the_same_question_under_a_new_id_is_rejected_on_the_text_axis() -> None:
    """This is the axis a re-sampled DPR pool actually trips: NQ contains near-duplicate
    questions under different ids, and an id-only check would wave one through."""
    existing = blank_split(query_texts=frozenset({normalize_query_text("Who sank it?")}))
    assert reasons(candidate("q1", text="who  sank IT ?"), existing) == {"leak_query_text": 1}


def test_a_gold_answer_shared_with_an_existing_split_is_rejected() -> None:
    existing = blank_split(answer_entities=frozenset({"1912"}))
    assert reasons(candidate("q1", answers=("1912",)), existing) == {"leak_answer_entity": 1}


def test_a_gold_passage_from_an_already_used_article_is_rejected() -> None:
    """Axis 2 is the parent page, not the passage: dpr-w100 splits one article into many
    document_ids, so a fresh passage of an already-used article is the same source."""
    existing = blank_split(parent_pages=frozenset({"titanic"}))
    assert reasons(candidate("q1"), existing) == {"leak_parent_page": 1}


def test_a_gold_passage_whose_text_an_existing_split_already_used_is_rejected() -> None:
    digest = sha256(one_document()["d1"].text.encode("utf-8")).hexdigest()
    existing = blank_split(passage_hashes=frozenset({digest}))
    assert reasons(candidate("q1"), existing) == {"leak_passage_hash": 1}


def test_a_needle_whose_article_title_cannot_be_parsed_is_rejected_not_kept() -> None:
    """M0 §3.4 makes title resolution a hard prerequisite for axis 2. An unresolved document
    becomes its own parent, which can never collide with anything — so keeping it would make
    the parent-page audit report zero overlap because it could not look, not because there is
    nothing there. That is the "plausible number instead of a crash" failure this project keeps
    paying for, so such a query is dropped at selection time and counted by reason.
    """
    documents = {
        "d1": Document(document_id="d1", text="no title line here", source_uri="fixture://d1")
    }
    assert reasons(candidate("q1"), blank_split(), documents=documents) == {
        "unresolved_parent": 1
    }


def test_a_query_with_no_gold_document_in_the_corpus_is_rejected() -> None:
    assert reasons(candidate("q1", gold=("missing",)), blank_split()) == {"no_gold_document": 1}


def test_a_query_with_no_reference_answer_is_rejected() -> None:
    """The answer-entity axis is undefined without an answer, so such a query cannot be audited
    even though the injector would reject it a step later anyway."""
    assert reasons(candidate("q1", answers=()), blank_split()) == {"no_reference_answers": 1}


def test_a_clean_query_survives_every_axis() -> None:
    assert eligible(candidate("q1"), blank_split()) == ("q1",)
    assert reasons(candidate("q1"), blank_split()) == {}


def test_every_gold_document_is_checked_not_only_the_one_the_injector_will_pick() -> None:
    """Which gold passage becomes the needle is decided later, by the injector. Checking only
    the first would make the axis depend on that choice, so every gold passage must be clean."""
    documents = {
        "d1": doc("d1", "Titanic", "The ship sank in 1912."),
        "d2": doc("d2", "RMS Olympic", "A sister ship."),
    }
    existing = blank_split(parent_pages=frozenset({"rms olympic"}))
    assert eligible(candidate("q1", gold=("d1", "d2")), existing, documents=documents) == ()


def test_every_existing_split_is_checked_not_only_the_first() -> None:
    documents = one_document()
    evaluation = evaluate_pool(
        candidates=(candidate("q1"),),
        documents=documents,
        existing=(blank_split("dev"), blank_split("train", query_ids=frozenset({"q1"}))),
    )
    assert evaluation.eligible_query_ids == ()


# ---------------------------------------------------------------- pool freeze


def many_candidates(count: int) -> tuple[dict[str, Document], tuple[SourceQuery, ...]]:
    documents = {
        f"d{index}": doc(f"d{index}", f"Article {index}", f"Body says 19{index:02d} once.")
        for index in range(count)
    }
    candidates = tuple(
        candidate(
            f"q{index}",
            text=f"question {index}?",
            answers=(f"19{index:02d}",),
            gold=(f"d{index}",),
        )
        for index in range(count)
    )
    return documents, candidates


def test_select_pool_refuses_to_return_a_short_pool() -> None:
    """M0 §4: the pool is frozen before the run. A short pool must stop the build, because the
    only way to continue is to add samples, and adding samples after seeing the shortfall is
    changing the data on the basis of a result."""
    documents, candidates = many_candidates(3)
    with pytest.raises(ValueError, match="eligible"):
        select_pool(
            candidates=candidates,
            documents=documents,
            existing=(blank_split(),),
            seed=42,
            pool_size=900,
        )


def test_the_pool_is_capped_at_the_frozen_size_and_ordered_by_the_frozen_order() -> None:
    documents, candidates = many_candidates(10)
    selection = select_pool(
        candidates=candidates,
        documents=documents,
        existing=(blank_split(),),
        seed=42,
        pool_size=4,
    )
    assert selection.n_considered == 10
    assert selection.pool_query_ids == frozen_order(
        [item.query_id for item in candidates], seed=42
    )[:4]


# ---------------------------------------------------------------- injection


def test_injection_stops_at_the_target_and_leaves_the_rest_of_the_pool_unused() -> None:
    """The headroom exists so the target can be met; it is not a licence to seal 665 questions.
    The first `target` successes in the frozen order are sealed and the remainder is discarded,
    which is what makes the sealed set a pure function of the frozen pool."""
    documents, candidates = many_candidates(4)
    by_id = {item.query_id: item for item in candidates}
    pool = frozen_order(list(by_id), seed=42)
    injected = inject_pool(
        pool_query_ids=pool,
        candidates_by_id=by_id,
        documents=documents,
        bank=build_answer_bank(["1800", "1801", "1802", "1803"], seed=42),
        existing=(blank_split(),),
        seed=42,
        target=2,
    )
    assert injected.query_ids == pool[:2]
    assert len(injected.records) == 2
    assert {twin.document_id for twin in injected.twins} == {
        f"cf::{query_id}::d{query_id[1:]}" for query_id in pool[:2]
    }


def test_injection_refuses_to_seal_fewer_than_the_target() -> None:
    """不得跑到一半发现不够再补 — a short yield stops the build; it never tops up."""
    documents, candidates = many_candidates(1)
    with pytest.raises(ValueError, match="target"):
        inject_pool(
            pool_query_ids=("q0",),
            candidates_by_id={"q0": candidates[0]},
            documents=documents,
            bank=build_answer_bank(["1800"], seed=42),
            existing=(blank_split(),),
            seed=42,
            target=2,
        )


def test_the_shortfall_message_names_the_reasons_so_the_skip_rate_can_be_read() -> None:
    """A bare "not enough" would leave the operator guessing whether M0 §4's .261 still holds
    or whether the leakage filter ate the pool — two very different situations."""
    documents = {"d0": doc("d0", "Article", "Body says 1900 and 1900 twice.")}
    candidates = {"q0": candidate("q0", answers=("1900",), gold=("d0",))}
    with pytest.raises(ValueError, match="not_injectable"):
        inject_pool(
            pool_query_ids=("q0",),
            candidates_by_id=candidates,
            documents=documents,
            bank=build_answer_bank(["1801"], seed=42),
            existing=(blank_split(),),
            seed=42,
            target=1,
        )


def test_a_query_with_no_same_class_replacement_is_counted_as_such() -> None:
    documents = {"d0": doc("d0", "Article", "Body says 1900 once.")}
    candidates = {"q0": candidate("q0", answers=("1900",), gold=("d0",))}
    with pytest.raises(ValueError, match="no_replacement"):
        inject_pool(
            pool_query_ids=("q0",),
            candidates_by_id=candidates,
            documents=documents,
            bank=build_answer_bank(["Alice"], seed=42),
            existing=(blank_split(),),
            seed=42,
            target=1,
        )


def test_a_synthetic_family_an_existing_split_already_used_is_dropped() -> None:
    """Axis 5 can only be applied after injection, because the family triple contains the
    replacement value. A collision means the relation model may have seen this exact
    (gold, replacement, class) contrast during domain adaptation (M0 §3.8)."""
    documents = {"d0": doc("d0", "Article", "Body says 1900 once.")}
    candidates = {"q0": candidate("q0", answers=("1900",), gold=("d0",))}
    with pytest.raises(ValueError, match="leak_synthetic_family"):
        inject_pool(
            pool_query_ids=("q0",),
            candidates_by_id=candidates,
            documents=documents,
            bank=build_answer_bank(["1801"], seed=42),
            existing=(blank_split(synthetic_families=frozenset({"1900|1801|year"})),),
            seed=42,
            target=1,
        )


def test_a_pool_query_missing_from_the_candidate_index_is_a_hard_error() -> None:
    """The pool is frozen; a member that cannot be resolved means the two artefacts disagree,
    which must not be silently absorbed into the skip count."""
    documents, candidates = many_candidates(1)
    with pytest.raises(KeyError):
        inject_pool(
            pool_query_ids=("q-unknown",),
            candidates_by_id={"q0": candidates[0]},
            documents=documents,
            bank=build_answer_bank(["1800"], seed=42),
            existing=(blank_split(),),
            seed=42,
            target=1,
        )


# ---------------------------------------------------------------- split fingerprints


def write_split(directory: Path, *, injected: bool = True) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    manifest = DatasetManifest(
        dataset_id="niah/dpr-w100-nq",
        dataset_version="subsample-10",
        split="dev",
        documents_file="documents.jsonl",
        queries_file="queries.jsonl",
        gold_cases_file="gold_cases.jsonl",
    )
    (directory / "manifest.json").write_text(manifest.model_dump_json(), encoding="utf-8")
    (directory / "documents.jsonl").write_text(
        doc("d1", "Titanic", "The ship sank in 1912.").model_dump_json() + "\n",
        encoding="utf-8",
    )
    (directory / "queries.jsonl").write_text(
        Query(query_id="q1", text="When did it sink?").model_dump_json() + "\n",
        encoding="utf-8",
    )
    (directory / "gold_cases.jsonl").write_text(
        GoldCase(
            query_id="q1", relevant_document_ids=("d1",), reference_answers=("1912",)
        ).model_dump_json()
        + "\n",
        encoding="utf-8",
    )
    if injected:
        write_provenance(
            directory / "provenance.jsonl",
            (
                MutationRecord(
                    query_id="q1",
                    needle_document_id="d1",
                    counterfactual_document_id="cf::q1::d1",
                    gold_value="1912",
                    gold_alias_used="1912",
                    replacement_value="1955",
                    string_class="year",
                    seed=42,
                    char_span=(0, 4),
                    text_hash_before="a" * 64,
                    text_hash_after="b" * 64,
                    answer_bank_hash="c" * 64,
                ),
            ),
        )
    return directory


def test_a_split_fingerprint_carries_all_five_axes(tmp_path: Path) -> None:
    fingerprint = read_split_fingerprint(write_split(tmp_path / "dev"))
    assert fingerprint.query_ids == frozenset({"q1"})
    assert fingerprint.query_texts == frozenset({"when did it sink"})
    assert fingerprint.parent_pages == frozenset({"titanic"})
    assert fingerprint.answer_entities == frozenset({"1912"})
    assert len(fingerprint.passage_hashes) == 1
    assert fingerprint.synthetic_families == frozenset({"1912|1955|year"})


def test_a_split_without_a_mutation_log_cannot_be_fingerprinted(tmp_path: Path) -> None:
    """Axis 5 compares against the prior splits' mutation logs. A missing log would contribute
    an empty family set, and the axis would then report zero overlap because it had nothing to
    compare rather than because the sets are disjoint. A genuinely uninjected split says so
    with an empty file, which is a deliberate act and leaves a trace."""
    with pytest.raises(ValueError, match="provenance"):
        read_split_fingerprint(write_split(tmp_path / "dev", injected=False))


def test_an_unparsable_gold_title_in_an_existing_split_is_a_hard_error(tmp_path: Path) -> None:
    """Same failure mode as on the sealed side, opposite direction: an existing gold page that
    cannot be resolved is missing from the exclusion set, so a sealed query drawn from that very
    article would pass axis 2."""
    directory = write_split(tmp_path / "dev")
    (directory / "documents.jsonl").write_text(
        Document(
            document_id="d1", text="titleless passage", source_uri="fixture://d1"
        ).model_dump_json()
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="parent"):
        read_split_fingerprint(directory)


# ---------------------------------------------------------------- manifest freeze


def a_manifest(**overrides: object) -> SealedManifest:
    payload: dict[str, object] = {
        "protocol_version": "g2-proto-4",
        "protocol_document_sha256": "d" * 64,
        "dataset_id": "niah/dpr-w100-nq-sealed600",
        "dataset_version": "sealed600-seed42",
        "split": "sealed-test",
        "source_dataset_id": "dpr-w100/natural-questions/dev",
        "seed": 42,
        "corpus_size": 100000,
        "target_injected": TARGET_INJECTED,
        "dev_sample_queries": DEV_SAMPLE_QUERIES,
        "dev_sample_injected": DEV_SAMPLE_INJECTED,
        "dev_skip_rate": DEV_SKIP_RATE,
        "required_pool_size": 812,
        "pool_size": POOL_SIZE,
        "n_considered": 5000,
        "n_injected": TARGET_INJECTED,
        "answer_bank_hash": "e" * 64,
        "answer_bank_source": "runs/niah-train-injected/manifest.json",
        "existing_splits": ("runs/niah-injected", "runs/niah-train-injected"),
        "label_provenance": {
            "gold_cases.jsonl": "official",
            "provenance.jsonl": "deterministic_rule",
        },
        "pool_rejections": {},
        "injection_rejections": {},
        "pool_query_ids": ("q1",),
        "query_ids": ("q1",),
        "artifact_sha256": {},
    }
    payload.update(overrides)
    return SealedManifest.model_validate(payload)


def test_a_label_provenance_outside_the_two_allowed_sources_cannot_be_written() -> None:
    """M0 §6: primary labels are `official` or `deterministic_rule`, full stop. A closed set
    means an LLM-judged or hand-annotated label cannot be recorded at all, so the audit is not
    reduced to trusting a free-text string."""
    with pytest.raises(ValueError):
        a_manifest(label_provenance={"gold_cases.jsonl": "llm_judge"})


def test_a_protocol_version_other_than_the_frozen_one_cannot_be_written() -> None:
    with pytest.raises(ValueError):
        a_manifest(protocol_version="g2-proto-3")


def test_freezing_refuses_to_overwrite_an_existing_manifest(tmp_path: Path) -> None:
    """The manifest IS the freeze. Overwriting it in place is exactly what a "we just topped it
    up" run would look like from the outside — indistinguishable from the original."""
    freeze_sealed_manifest(tmp_path, a_manifest())
    with pytest.raises(ValueError, match="already frozen"):
        freeze_sealed_manifest(tmp_path, a_manifest())


def test_a_frozen_manifest_round_trips(tmp_path: Path) -> None:
    freeze_sealed_manifest(tmp_path, a_manifest())
    assert read_sealed_manifest(tmp_path).query_ids == ("q1",)


def test_verify_artifacts_detects_an_edited_file(tmp_path: Path) -> None:
    """"manifest 与 hash 在任何实验之前冻结" is only worth something if the hashes are checked
    afterwards; otherwise the freeze is a claim, not a control."""
    (tmp_path / "documents.jsonl").write_text("{}\n", encoding="utf-8")
    manifest = a_manifest(
        artifact_sha256={"documents.jsonl": sha256_file(tmp_path / "documents.jsonl")}
    )
    assert verify_artifacts(tmp_path, manifest) == ()
    (tmp_path / "documents.jsonl").write_text('{"tampered": true}\n', encoding="utf-8")
    assert verify_artifacts(tmp_path, manifest) != ()


def test_verify_artifacts_reports_a_missing_file(tmp_path: Path) -> None:
    manifest = a_manifest(artifact_sha256={"documents.jsonl": "f" * 64})
    assert verify_artifacts(tmp_path, manifest) != ()
