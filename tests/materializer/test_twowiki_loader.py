import json
from pathlib import Path

from evidence_rag.infrastructure.datasets import JsonlDatasetAdapter
from evidence_rag.materializer.twowiki_loader import materialize_2wiki


def _rows() -> list[dict]:
    # context / supporting_facts arrive as JSON strings in the real parquet.
    return [
        {
            "_id": "q1",
            "question": "Which country is Paris the capital of?",
            "answer": "France",
            "context": json.dumps(
                [["Paris", ["Paris is the capital of France."]], ["France", ["France is in Europe."]]]
            ),
            "supporting_facts": json.dumps([["Paris", 0], ["France", 0]]),
        },
        {
            "_id": "q2",
            "question": "Where is the Eiffel Tower?",
            "answer": "Paris",
            "context": json.dumps(
                [["Paris", ["Paris is the capital of France."]], ["Eiffel Tower", ["The tower is in Paris."]]]
            ),
            "supporting_facts": json.dumps([["Eiffel Tower", 0]]),
        },
        {
            # gold title absent from context -> unreachable label -> dropped
            "_id": "q3",
            "question": "bad",
            "answer": "x",
            "context": json.dumps([["A", ["a body"]]]),
            "supporting_facts": json.dumps([["Missing", 0]]),
        },
    ]


def test_dedup_gold_and_answers(tmp_path: Path) -> None:
    result = materialize_2wiki(_rows(), tmp_path, query_limit=10, seed=1)
    bundle = JsonlDatasetAdapter.load(result.manifest_path)

    # q3 dropped; the two shared "Paris" paragraphs collapse to one document.
    assert {q.query_id for q in bundle.queries} == {"q1", "q2"}
    ids = [d.document_id for d in bundle.documents]
    assert sorted(ids) == ["Eiffel Tower", "France", "Paris"]
    assert ids.count("Paris") == 1

    gold = {g.query_id: g for g in bundle.gold_cases}
    assert gold["q1"].relevant_document_ids == ("France", "Paris")
    assert gold["q1"].reference_answers == ("France",)
    assert gold["q2"].relevant_document_ids == ("Eiffel Tower",)

    paris = next(d for d in bundle.documents if d.document_id == "Paris")
    assert "Paris" in paris.text and "capital of France" in paris.text


def test_materialization_is_deterministic(tmp_path: Path) -> None:
    a = materialize_2wiki(_rows(), tmp_path / "a", query_limit=10, seed=7)
    b = materialize_2wiki(_rows(), tmp_path / "b", query_limit=10, seed=7)
    ids_a = [d.document_id for d in JsonlDatasetAdapter.load(a.manifest_path).documents]
    ids_b = [d.document_id for d in JsonlDatasetAdapter.load(b.manifest_path).documents]
    assert ids_a == ids_b


def test_materialization_records_requested_split(tmp_path: Path) -> None:
    result = materialize_2wiki(
        _rows(), tmp_path, query_limit=10, seed=7, split="train"
    )
    bundle = JsonlDatasetAdapter.load(result.manifest_path)
    assert bundle.manifest.split == "train"
    assert bundle.manifest.dataset_version == "train-subsample-10"
    assert all(document.source_uri.startswith("2wiki://train/") for document in bundle.documents)


def test_cli_uses_injected_rows(tmp_path: Path) -> None:
    from evidence_rag.materializer.twowiki_cli import main

    out = tmp_path / "twowiki"
    exit_code = main(("--output", str(out), "--query-limit", "10", "--seed", "1"), rows=_rows())
    assert exit_code == 0
    bundle = JsonlDatasetAdapter.load(out / "manifest.json")
    assert {q.query_id for q in bundle.queries} == {"q1", "q2"}


def test_cli_excludes_existing_query_ids(tmp_path: Path) -> None:
    from evidence_rag.materializer.twowiki_cli import main

    excluded = tmp_path / "excluded.jsonl"
    excluded.write_text('{"query_id":"q1"}\n', encoding="utf-8")
    out = tmp_path / "twowiki-heldout"
    exit_code = main(
        (
            "--output",
            str(out),
            "--query-limit",
            "10",
            "--seed",
            "1",
            "--source-split",
            "dev",
            "--output-split",
            "heldout",
            "--exclude-queries",
            str(excluded),
        ),
        rows=_rows(),
    )
    assert exit_code == 0
    bundle = JsonlDatasetAdapter.load(out / "manifest.json")
    assert bundle.manifest.split == "heldout"
    assert {query.query_id for query in bundle.queries} == {"q2"}
