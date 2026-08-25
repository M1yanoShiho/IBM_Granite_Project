from __future__ import annotations

import json
from pathlib import Path

import pytest

from evidence_rag.evaluation.experiment05_data import (
    Goal1DataError,
    RuntimeIsolationError,
    audit_bundle_pair,
    materialize_asqa_record,
    materialize_kilt_record,
    read_asqa_records,
    read_kilt_records,
    read_runtime_bundle,
    read_triviaqa_question_lookup,
    scan_exposure_paths,
    select_asqa_ids,
    select_kilt_ids,
    validate_runtime_record,
    validate_sidecar_record,
)


def test_kilt_selection_uses_frozen_hash_order_and_excludes_exposure() -> None:
    selection = select_kilt_ids(
        dataset="kilt-nq",
        train_ids=("d1", "d2", "d3", "d4"),
        formal_pool_ids=("f1", "f2", "f3", "f4", "f5"),
        exposure_ids=frozenset({"f2"}),
        development_count=2,
        formal_count=3,
    )

    assert selection.development_ids == ("d4", "d1")
    assert selection.formal_ids == ("f5", "f3", "f4")
    assert selection.development_ordered_ids_sha256 == (
        "49bda741404b66d4e6f1a7c92ef5c0fb11d8c5e3a07c76238c09f9950ece544e"
    )
    assert selection.formal_ordered_ids_sha256 == (
        "c97f57bd7efb3e19e602b8635f2d47ce3c67c9bbdacdd899ad93f97f69fbb1fb"
    )
    assert set(selection.formal_ids).isdisjoint({"f2"})


def test_asqa_selection_prefers_exposed_development_then_excludes_it_from_formal() -> None:
    selection = select_asqa_ids(
        dataset="alce-asqa",
        all_ids=("a1", "a2", "a3", "a4", "a5", "a6"),
        exposure_ids=frozenset({"a1"}),
        development_count=3,
        formal_count=2,
    )

    assert selection.development_ids == ("a1", "a5", "a3")
    assert selection.formal_ids == ("a2", "a4")
    assert selection.development_ordered_ids_sha256 == (
        "2d988d9cbb31c0a48b4303c009516149b92e9d61a3c5f07e800578a8a9f129c1"
    )
    assert selection.formal_ordered_ids_sha256 == (
        "07bfc9d7ef511371a39e80b6565adb82b61608c3334f5624f7ddbfef4c227152"
    )
    assert set(selection.development_ids).isdisjoint(selection.formal_ids)


def test_selection_refuses_to_shrink_an_insufficient_formal_pool() -> None:
    with pytest.raises(Goal1DataError, match="requires 3 unexposed formal IDs; found 2"):
        select_kilt_ids(
            dataset="kilt-tqa",
            train_ids=("d1", "d2"),
            formal_pool_ids=("f1", "f2", "f3"),
            exposure_ids=frozenset({"f2"}),
            development_count=2,
            formal_count=3,
        )


def test_selection_rejects_duplicate_canonical_ids() -> None:
    with pytest.raises(Goal1DataError, match="duplicate canonical IDs"):
        select_asqa_ids(
            dataset="alce-asqa",
            all_ids=("a1", "a1", "a2"),
            exposure_ids=frozenset(),
            development_count=1,
            formal_count=1,
        )


def _runtime(query_id: str) -> dict[str, object]:
    return {
        "schema_version": "experiment05.runtime.v1",
        "dataset": "kilt-nq",
        "query_id": query_id,
        "question": "Which city is the capital of France?",
        "corpus_snapshot_id": "kilt-wikipedia-20190801",
        "bm25_index_id": "bm25-sha256:abc",
        "dense_index_id": "granite-dense-sha256:def",
    }


def _sidecar(query_id: str) -> dict[str, object]:
    return {
        "schema_version": "experiment05.scorer.v1",
        "dataset": "kilt-nq",
        "query_id": query_id,
        "reference_fact_groups": [
            {
                "fact_id": "fact-1",
                "fact_question": "Which city is the capital of France?",
                "aliases": ["Paris"],
            }
        ],
        "gold_provenance": [
            {
                "fact_id": "fact-1",
                "source_kind": "kilt-paragraph",
                "source_id": "22989",
                "start_unit": 1,
                "end_unit": 1,
            }
        ],
    }


def test_runtime_contract_accepts_only_gold_free_index_access_fields() -> None:
    validate_runtime_record(_runtime("q1"), dataset="kilt-nq")

    poisoned = _runtime("q1")
    poisoned["gold_answer"] = "Paris"
    with pytest.raises(RuntimeIsolationError, match="unexpected=.*gold_answer"):
        validate_runtime_record(poisoned, dataset="kilt-nq")


def test_sidecar_contract_requires_reference_facts_and_aliases() -> None:
    validate_sidecar_record(_sidecar("q1"), dataset="kilt-nq")

    malformed = _sidecar("q1")
    malformed["reference_fact_groups"] = []
    with pytest.raises(RuntimeIsolationError, match="reference_fact_groups"):
        validate_sidecar_record(malformed, dataset="kilt-nq")


def test_sidecar_accepts_asqa_title_provenance_without_fake_paragraph_spans() -> None:
    sidecar = _sidecar("q1")
    sidecar["dataset"] = "alce-asqa"
    sidecar["gold_provenance"] = [
        {
            "fact_id": "fact-1",
            "source_kind": "wikipedia-title",
            "source_id": "Paris",
            "start_unit": None,
            "end_unit": None,
        }
    ]

    validate_sidecar_record(sidecar, dataset="alce-asqa")


def test_runtime_reader_cannot_accept_a_scorer_sidecar(tmp_path) -> None:
    sidecar_path = tmp_path / "scorer_only.jsonl"
    sidecar_path.write_text(json.dumps(_sidecar("q1")) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeIsolationError, match="runtime record keys differ"):
        read_runtime_bundle(sidecar_path, dataset="kilt-nq")


def test_bundle_audit_requires_physical_separation_and_identical_order(tmp_path) -> None:
    runtime_dir = tmp_path / "runtime"
    sidecar_dir = tmp_path / "scorer_only"
    runtime_dir.mkdir()
    sidecar_dir.mkdir()
    runtime_path = runtime_dir / "kilt-nq.jsonl"
    sidecar_path = sidecar_dir / "kilt-nq.jsonl"
    runtime_path.write_text(
        "".join(json.dumps(_runtime(query_id)) + "\n" for query_id in ("q1", "q2")),
        encoding="utf-8",
    )
    sidecar_path.write_text(
        "".join(json.dumps(_sidecar(query_id)) + "\n" for query_id in ("q1", "q2")),
        encoding="utf-8",
    )

    audit = audit_bundle_pair(
        runtime_path,
        sidecar_path,
        dataset="kilt-nq",
        expected_query_ids=("q1", "q2"),
    )

    assert audit == {
        "count": 2,
        "ordered_ids_sha256": "548668e92cb97abb3f2c0385fe088c6875b9e493e1e202531b7eca4515666e4b",
        "runtime_gold_field_count": 0,
        "physical_parent_directories_separate": True,
        "runtime_schema_version": "experiment05.runtime.v1",
        "sidecar_schema_version": "experiment05.scorer.v1",
    }


def test_kilt_materialization_keeps_question_runtime_only_and_groups_aliases() -> None:
    raw = {
        "id": "nq-1",
        "input": "Which city is the capital of France?",
        "output": [
            {
                "answer": "Paris",
                "provenance": [
                    {
                        "wikipedia_id": 22989,
                        "start_paragraph_id": 1,
                        "end_paragraph_id": 1,
                    }
                ],
            },
            {
                "answer": "City of Paris",
                "provenance": [
                    {
                        "wikipedia_id": 22989,
                        "start_paragraph_id": 1,
                        "end_paragraph_id": 1,
                    }
                ],
            },
            {
                "answer": "Paris",
                "provenance": [],
            },
        ],
    }

    runtime, sidecar = materialize_kilt_record(
        raw,
        dataset="kilt-nq",
        corpus_snapshot_id="kilt-wikipedia-20190801",
        bm25_index_id="bm25-sha256:abc",
        dense_index_id="granite-dense-sha256:def",
    )

    assert runtime == _runtime("nq-1")
    assert sidecar["reference_fact_groups"] == [
        {
            "fact_id": "fact-0001",
            "fact_question": "Which city is the capital of France?",
            "aliases": ["Paris", "City of Paris"],
        }
    ]
    assert sidecar["gold_provenance"] == [
        {
            "fact_id": "fact-0001",
            "source_kind": "kilt-paragraph",
            "source_id": "22989",
            "start_unit": 1,
            "end_unit": 1,
        }
    ]


def test_asqa_materialization_uses_qa_pairs_but_never_the_supplied_top100_docs() -> None:
    raw = {
        "sample_id": "asqa-1",
        "question": "What can Mercury refer to?",
        "qa_pairs": [
            {
                "question": "Which planet is Mercury?",
                "short_answers": ["the closest planet to the Sun", "Mercury"],
                "wikipage": "Mercury (planet)",
            },
            {
                "question": "Which element uses Hg?",
                "short_answers": ["mercury"],
                "wikipage": None,
            },
        ],
        "docs": [{"title": "LEAK", "text": "Oracle/top-100 text must not enter runtime."}],
        "answer": "A gold long answer that must remain scorer-only.",
        "annotations": [{"knowledge": [{"content": "Gold support text."}]}],
        "wikipages": [{"title": "Mercury", "url": "https://example.invalid"}],
    }

    runtime, sidecar = materialize_asqa_record(
        raw,
        corpus_snapshot_id="dpr-wikipedia-psgs-w100",
        bm25_index_id="bm25-sha256:ghi",
        dense_index_id="granite-dense-sha256:jkl",
    )

    assert runtime == {
        "schema_version": "experiment05.runtime.v1",
        "dataset": "alce-asqa",
        "query_id": "asqa-1",
        "question": "What can Mercury refer to?",
        "corpus_snapshot_id": "dpr-wikipedia-psgs-w100",
        "bm25_index_id": "bm25-sha256:ghi",
        "dense_index_id": "granite-dense-sha256:jkl",
    }
    assert "docs" not in runtime
    assert "answer" not in runtime
    assert sidecar["reference_fact_groups"] == [
        {
            "fact_id": "fact-0001",
            "fact_question": "Which planet is Mercury?",
            "aliases": ["the closest planet to the Sun", "Mercury"],
        },
        {
            "fact_id": "fact-0002",
            "fact_question": "Which element uses Hg?",
            "aliases": ["mercury"],
        },
    ]
    assert sidecar["gold_provenance"] == [
        {
            "fact_id": "fact-0001",
            "source_kind": "wikipedia-title",
            "source_id": "Mercury (planet)",
            "start_unit": None,
            "end_unit": None,
        }
    ]


def test_exposure_scan_reads_only_query_id_fields_and_intersects_canonical_ids(tmp_path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "query_id": "q1",
                "nested": {"ordered_query_ids": ["q2", "not-canonical"]},
                "id": "q3",
                "wikipedia_id": "q3",
            }
        ),
        encoding="utf-8",
    )
    rows = tmp_path / "rows.jsonl"
    rows.write_text(
        json.dumps({"case_id": "q2"}) + "\n" + json.dumps({"document_id": "q3"}) + "\n",
        encoding="utf-8",
    )

    registry = scan_exposure_paths((manifest, rows), canonical_ids=frozenset({"q1", "q2", "q3"}))

    assert registry.exposure_ids == ("q1", "q2")
    assert [source["matched_count"] for source in registry.sources] == [2, 1]
    assert all(len(source["sha256"]) == 64 for source in registry.sources)


def test_exposure_scan_accepts_legacy_json_files_that_are_actually_jsonl(tmp_path) -> None:
    source = tmp_path / "legacy.json"
    source.write_text(
        json.dumps({"query_id": "q1"}) + "\n" + json.dumps({"query_id": "q2"}) + "\n",
        encoding="utf-8",
    )

    registry = scan_exposure_paths((source,), canonical_ids={"q1", "q2", "q3"})

    assert registry.exposure_ids == ("q1", "q2")


def test_kilt_reader_joins_missing_triviaqa_questions_by_canonical_id(tmp_path) -> None:
    source = tmp_path / "triviaqa-kilt.jsonl"
    source.write_text(
        json.dumps({"id": "t1", "output": [{"answer": "A"}]})
        + "\n"
        + json.dumps({"id": "t2", "output": [{"answer": "B"}]})
        + "\n",
        encoding="utf-8",
    )

    records = read_kilt_records(source, question_lookup={"t1": "Question one?", "t2": "Q2?"})

    assert [record["id"] for record in records] == ["t1", "t2"]
    assert [record["input"] for record in records] == ["Question one?", "Q2?"]


def test_kilt_reader_refuses_missing_question_or_duplicate_id(tmp_path) -> None:
    source = tmp_path / "bad-kilt.jsonl"
    source.write_text(
        json.dumps({"id": "t1", "output": []})
        + "\n"
        + json.dumps({"id": "t1", "output": []})
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(Goal1DataError, match="duplicate canonical ID"):
        read_kilt_records(source, question_lookup={"t1": "Question?"})

    source.write_text(json.dumps({"id": "t2", "output": []}) + "\n", encoding="utf-8")
    with pytest.raises(Goal1DataError, match="missing a TriviaQA question"):
        read_kilt_records(source, question_lookup={"t1": "Question?"})


def test_kilt_reader_skips_unselected_missing_question(tmp_path: Path) -> None:
    source = tmp_path / "tqa.jsonl"
    source.write_text(
        json.dumps({"id": "selected", "output": []})
        + "\n"
        + json.dumps({"id": "unselected-missing", "output": []})
        + "\n",
        encoding="utf-8",
    )

    records = read_kilt_records(
        source,
        question_lookup={"selected": "Selected question?"},
        allowed_ids={"selected"},
    )

    assert [row["id"] for row in records] == ["selected"]


def test_triviaqa_parquet_lookup_is_strict_and_stable(tmp_path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"
    pq.write_table(
        pa.table({"question_id": ["t1", "t2"], "question": ["Q1?", "Q2?"]}),
        first,
    )
    pq.write_table(
        pa.table({"question_id": ["t2", "t3"], "question": ["Q2?", "Q3?"]}),
        second,
    )

    assert read_triviaqa_question_lookup((first, second)) == {
        "t1": "Q1?",
        "t2": "Q2?",
        "t3": "Q3?",
    }

    pq.write_table(pa.table({"question_id": ["t1"], "question": ["DIFFERENT"]}), second)
    with pytest.raises(Goal1DataError, match="conflicting question text"):
        read_triviaqa_question_lookup((first, second))


def test_asqa_reader_requires_a_json_list_and_unique_sample_ids(tmp_path) -> None:
    source = tmp_path / "asqa.json"
    source.write_text(
        json.dumps([{"sample_id": "a1"}, {"sample_id": "a2"}]), encoding="utf-8"
    )
    assert [record["sample_id"] for record in read_asqa_records(source)] == ["a1", "a2"]

    source.write_text(json.dumps([{"sample_id": "a1"}, {"sample_id": "a1"}]), encoding="utf-8")
    with pytest.raises(Goal1DataError, match="duplicate canonical ID"):
        read_asqa_records(source)
