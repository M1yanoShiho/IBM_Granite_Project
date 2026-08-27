from __future__ import annotations

import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "scripts", ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import experiment05_goal1 as g1  # noqa: E402


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _sources(root: Path) -> None:
    _write_jsonl(
        root / "nq-train-kilt.jsonl",
        [{"id": f"nd{i}", "input": f"NQ dev {i}", "output": []} for i in range(4)],
    )
    _write_jsonl(
        root / "nq-dev-kilt.jsonl",
        [{"id": f"nf{i}", "input": f"NQ formal {i}", "output": []} for i in range(5)],
    )
    _write_jsonl(
        root / "triviaqa-train_id-kilt.jsonl",
        [{"id": f"td{i}", "output": []} for i in range(4)],
    )
    _write_jsonl(
        root / "triviaqa-dev_id-kilt.jsonl",
        [{"id": f"tf{i}", "output": []} for i in range(5)],
    )
    pq.write_table(
        pa.table(
            {
                "question_id": [f"td{i}" for i in range(4)],
                "question": [f"TQA dev {i}" for i in range(4)],
            }
        ),
        root / "triviaqa-unfiltered-nocontext-train.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "question_id": [f"tf{i}" for i in range(5)],
                "question": [f"TQA formal {i}" for i in range(5)],
            }
        ),
        root / "triviaqa-unfiltered-nocontext-validation.parquet",
    )
    alce = root / "ALCE-data"
    alce.mkdir()
    (alce / "asqa_eval_gtr_top100.json").write_text(
        json.dumps([{"sample_id": f"af{i}"} for i in range(7)]), encoding="utf-8"
    )


def test_inventory_writes_only_ids_counts_and_hashes(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "inventory"
    source.mkdir()
    _sources(source)

    manifest = g1.inventory_stage(source, output)

    assert manifest["schema_version"] == "experiment05.data_inventory.v1"
    assert manifest["datasets"]["kilt-nq"]["train"]["count"] == 4
    assert manifest["datasets"]["kilt-tqa"]["formal_pool"]["count"] == 5
    assert manifest["datasets"]["alce-asqa"]["all"]["count"] == 7
    ids_payload = json.loads((output / "kilt-tqa.formal_pool.ids.json").read_text())
    assert ids_payload["canonical_ids"] == ["tf0", "tf1", "tf2", "tf3", "tf4"]
    assert "TQA formal" not in json.dumps(manifest)
    assert "TQA formal" not in json.dumps(ids_payload)


def test_inventory_does_not_drop_kilt_ids_when_question_conversion_is_incomplete(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _sources(source)
    pq.write_table(
        pa.table({"question_id": ["td0"], "question": ["TQA dev 0"]}),
        source / "triviaqa-unfiltered-nocontext-train.parquet",
    )

    manifest = g1.inventory_stage(source, tmp_path / "inventory")

    row = manifest["datasets"]["kilt-tqa"]["train"]
    assert row["count"] == 4
    assert row["question_lookup_matched_count"] == 1
    assert row["question_lookup_missing_count"] == 3


def test_selection_merges_exposure_and_freezes_exact_counts(tmp_path: Path) -> None:
    inventory = tmp_path / "inventory"
    inventory.mkdir()
    payloads = {
        "kilt-nq.train.ids.json": ["nd0", "nd1", "nd2", "nd3"],
        "kilt-nq.formal_pool.ids.json": ["nf0", "nf1", "nf2", "nf3", "nf4"],
        "kilt-tqa.train.ids.json": ["td0", "td1", "td2", "td3"],
        "kilt-tqa.formal_pool.ids.json": ["tf0", "tf1", "tf2", "tf3", "tf4"],
        "alce-asqa.all.ids.json": ["af0", "af1", "af2", "af3", "af4", "af5", "af6"],
    }
    for name, ids in payloads.items():
        (inventory / name).write_text(json.dumps({"canonical_ids": ids}), encoding="utf-8")
    exposures = tmp_path / "exposures"
    exposures.mkdir()
    (exposures / "nq-local.json").write_text(
        json.dumps({"dataset": "kilt-nq", "exposure_ids": ["nf1"]}), encoding="utf-8"
    )
    (exposures / "nq-remote.json").write_text(
        json.dumps({"dataset": "kilt-nq", "exposure_ids": ["nf2"]}), encoding="utf-8"
    )
    (exposures / "tqa.json").write_text(
        json.dumps({"dataset": "kilt-tqa", "exposure_ids": []}), encoding="utf-8"
    )
    (exposures / "asqa.json").write_text(
        json.dumps({"dataset": "alce-asqa", "exposure_ids": ["af0"]}), encoding="utf-8"
    )

    manifest = g1.selection_stage(
        inventory,
        {
            "kilt-nq": (exposures / "nq-local.json", exposures / "nq-remote.json"),
            "kilt-tqa": (exposures / "tqa.json",),
            "alce-asqa": (exposures / "asqa.json",),
        },
        tmp_path / "selection",
        development_count=2,
        formal_count=2,
    )

    assert manifest["status"] == "FROZEN"
    assert manifest["datasets"]["kilt-nq"]["exposure_count"] == 2
    for dataset in ("kilt-nq", "kilt-tqa", "alce-asqa"):
        row = manifest["datasets"][dataset]
        assert row["development_count"] == 2
        assert row["formal_count"] == 2
        assert row["development_formal_disjoint"] is True
        assert row["formal_exposure_disjoint"] is True


def test_historical_asqa_exposure_is_replayed_with_the_recorded_sampler(tmp_path: Path) -> None:
    source = tmp_path / "asqa.json"
    rows = []
    for index in range(5):
        rows.append(
            {
                "sample_id": f"a{index}",
                "question": f"Question {index}?",
                "answer": f"Answer {index}.",
                "docs": [{"text": "evidence"}] * 5,
                "annotations": [{"knowledge": [{"content": f"Answer {index}."}]}],
                "qa_pairs": [{"short_answers": [f"Answer {index}"]}],
            }
        )
    source.write_text(json.dumps(rows), encoding="utf-8")

    payload = g1.reconstruct_asqa_exposure_stage(
        source,
        tmp_path / "exposure.json",
        limit=3,
        seed=13,
        top_k=5,
    )

    assert payload["dataset"] == "alce-asqa"
    assert payload["exposure_count"] == 3
    assert len(payload["exposure_ids"]) == 3
    assert payload["reconstruction"]["method"] == "exact_historical_g3_build_cases_replay"
    assert len(payload["reconstruction"]["sampler_script_sha256"]) == 64


def test_materialize_writes_gold_free_runtime_and_separate_scorer_sidecars(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    selection = tmp_path / "selection"
    output = tmp_path / "bundles"
    (source / "ALCE-data").mkdir(parents=True)
    selection.mkdir()

    def kilt(query_id: str, question: str, answer: str) -> dict[str, object]:
        return {
            "id": query_id,
            "input": question,
            "output": [
                {
                    "answer": answer,
                    "provenance": [
                        {
                            "wikipedia_id": 7,
                            "start_paragraph_id": 0,
                            "end_paragraph_id": 0,
                        }
                    ],
                }
            ],
        }

    _write_jsonl(
        source / "nq-train-kilt.jsonl",
        [kilt("nd0", "NQ development question?", "NQ development answer")],
    )
    _write_jsonl(
        source / "nq-dev-kilt.jsonl",
        [kilt("nf0", "NQ formal question?", "NQ formal answer")],
    )
    _write_jsonl(
        source / "triviaqa-train_id-kilt.jsonl",
        [kilt("td0", "TQA development question?", "TQA development answer")],
    )
    _write_jsonl(
        source / "triviaqa-dev_id-kilt.jsonl",
        [kilt("tf0", "TQA formal question?", "TQA formal answer")],
    )
    pq.write_table(
        pa.table({"question_id": ["td0"], "question": ["TQA development question?"]}),
        source / "triviaqa-unfiltered-nocontext-train.parquet",
    )
    pq.write_table(
        pa.table({"question_id": ["tf0"], "question": ["TQA formal question?"]}),
        source / "triviaqa-unfiltered-nocontext-validation.parquet",
    )
    asqa_rows = [
        {
            "sample_id": query_id,
            "question": question,
            "qa_pairs": [
                {
                    "question": fact_question,
                    "short_answers": [answer],
                    "wikipage": "Example",
                }
            ],
        }
        for query_id, question, fact_question, answer in (
            ("ad0", "ASQA development question?", "Development fact?", "development"),
            ("af0", "ASQA formal question?", "Formal fact?", "formal"),
        )
    ]
    (source / "ALCE-data/asqa_eval_gtr_top100.json").write_text(
        json.dumps(asqa_rows), encoding="utf-8"
    )
    for dataset, development_id, formal_id in (
        ("kilt-nq", "nd0", "nf0"),
        ("kilt-tqa", "td0", "tf0"),
        ("alce-asqa", "ad0", "af0"),
    ):
        for split, query_id in (("development", development_id), ("formal", formal_id)):
            (selection / f"{dataset}.{split}.ids.json").write_text(
                json.dumps({"canonical_ids": [query_id]}), encoding="utf-8"
            )

    manifest = g1.materialize_stage(
        source,
        selection,
        output,
        index_identities={
            "kilt": {
                "corpus_snapshot_id": "kilt-snapshot",
                "bm25_index_id": "bm25-sha256:kilt",
                "dense_index_id": "dense-sha256:kilt",
            },
            "dpr": {
                "corpus_snapshot_id": "dpr-snapshot",
                "bm25_index_id": "bm25-sha256:dpr",
                "dense_index_id": "dense-sha256:dpr",
            },
        },
    )

    assert manifest["status"] == "FROZEN_RUNTIME_SIDECAR_ISOLATION_PASS"
    assert manifest["datasets"]["kilt-nq"]["formal"]["count"] == 1
    runtime = json.loads((output / "runtime/formal/kilt-nq.jsonl").read_text())
    assert set(runtime) == {
        "schema_version",
        "dataset",
        "query_id",
        "question",
        "corpus_snapshot_id",
        "bm25_index_id",
        "dense_index_id",
    }
    assert (output / "runtime/formal").resolve() != (
        output / "scorer_only/formal"
    ).resolve()
    assert (output / "scorer_only").stat().st_mode & 0o777 == 0o700
