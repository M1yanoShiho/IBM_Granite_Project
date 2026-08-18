from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import full_flow_g200_v2 as g200_v2  # noqa: E402
from full_flow_g200 import QA2D_MODEL_ID, QA2D_REVISION  # noqa: E402


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _candidate(
    evidence_id: str,
    document_id: str,
    rank: int,
    text: str,
    *,
    harmful: bool = False,
) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "evidence_id": evidence_id,
        "document_id": f"cf::{document_id}" if harmful else document_id,
        "chunk_id": f"chunk-{evidence_id}",
        "text": text,
        "source_uri": f"synthetic://cf/{document_id}" if harmful else f"memory://{document_id}",
        "retrieval_score": float(100 - rank),
        "retrieval_rank": rank,
    }


def _top10(query_id: str, support_docs: list[tuple[str, str]]) -> dict[str, object]:
    candidates: list[dict[str, object]] = []
    rank = 1
    for document_id, text in support_docs:
        candidates.append(_candidate(f"{query_id}-s{rank}", document_id, rank, text))
        rank += 1
    candidates.append(_candidate(f"{query_id}-h", "harm", rank, "Wrong injected answer.", harmful=True))
    rank += 1
    while rank <= 10:
        candidates.append(
            _candidate(
                f"{query_id}-b{rank}",
                f"benign-{rank}",
                rank,
                f"Benign context {rank}.",
            )
        )
        rank += 1
    return {"schema_version": "1.0", "query_id": query_id, "candidates": candidates}


def test_prepare_niah_modelval_excludes_old_parent_overlap(tmp_path: Path) -> None:
    queries = _write_jsonl(
        tmp_path / "queries.jsonl",
        [
            {"schema_version": "1.0", "query_id": "old", "text": "old?"},
            {"schema_version": "1.0", "query_id": "new-ok", "text": "new?"},
            {"schema_version": "1.0", "query_id": "new-overlap", "text": "overlap?"},
        ],
    )
    gold = _write_jsonl(
        tmp_path / "gold.jsonl",
        [
            {"query_id": "old", "reference_answers": ["2001"], "relevant_document_ids": ["old-doc"]},
            {"query_id": "new-ok", "reference_answers": ["2009"], "relevant_document_ids": ["new-doc"]},
            {
                "query_id": "new-overlap",
                "reference_answers": ["2010"],
                "relevant_document_ids": ["overlap-doc"],
            },
        ],
    )
    pool = _write_jsonl(
        tmp_path / "candidate_sets.jsonl",
        [
            _top10("old", [("old-doc", "The answer is 2001.")]),
            _top10("new-ok", [("new-doc", "The answer is 2009.")]),
            _top10("new-overlap", [("overlap-doc", "The answer is 2010.")]),
        ],
    )
    source_parent = _write_jsonl(
        tmp_path / "source_parent.jsonl",
        [
            {"document_id": "old-doc", "source_parent_id": "old-parent"},
            {"document_id": "new-doc", "source_parent_id": "new-parent"},
            {"document_id": "overlap-doc", "source_parent_id": "old-parent"},
        ],
    )
    old_roles = _write_jsonl(
        tmp_path / "old_roles.jsonl",
        [{"query_id": "old", "role": "train-fit", "component_id": "old-component"}],
    )
    old_components = _write_jsonl(
        tmp_path / "old_components.jsonl",
        [
            {
                "query_id": "old",
                "component_id": "old-component",
                "allowed_keys": [{"axis": "parent", "value": "old-parent"}],
            }
        ],
    )

    summary = g200_v2.prepare_niah_modelval(
        queries_path=queries,
        candidate_pool_path=pool,
        gold_path=gold,
        source_parent_path=source_parent,
        old_role_assignments_path=old_roles,
        old_component_map_path=old_components,
        output_dir=tmp_path / "prepared",
    )

    assert summary["selected_queries"] == 1
    assert summary["ordered_query_ids"] == ["new-ok"]
    assert summary["counts"]["excluded_old_parent_overlap"] == 1
    role_rows = [
        json.loads(line)
        for line in (tmp_path / "prepared/role_assignments.jsonl").read_text().splitlines()
    ]
    assert role_rows[0]["role"] == "train-modelval"


def test_materialize_builds_g200_v2_manifest_and_unsupported_rows(tmp_path: Path) -> None:
    old_train = _write_jsonl(
        tmp_path / "old_train.jsonl",
        [
            {
                "schema_version": "full-flow-g200-case-v1",
                "query_id": "old-train",
                "role": "train-fit",
                "component_id": "old-component",
                "question": "When?",
                "answer": "2001",
                "semantic_target": "It happened in 2001.",
                "semantic_target_sha256": "sha",
                "variants": {
                    "support_only": {
                        "evidence_ids": ["old-e1"],
                        "support_evidence_id": "old-e1",
                        "prompt": "Question?",
                        "target": "It happened in 2001 [1].",
                    }
                },
            }
        ],
    )
    niah_queries = _write_jsonl(
        tmp_path / "niah_queries.jsonl",
        [{"schema_version": "1.0", "query_id": "niah-val", "text": "When did it happen?"}],
    )
    niah_gold = _write_jsonl(
        tmp_path / "niah_gold.jsonl",
        [
            {
                "query_id": "niah-val",
                "reference_answers": ["2009"],
                "relevant_document_ids": ["niah-doc"],
            }
        ],
    )
    niah_pool = _write_jsonl(
        tmp_path / "niah_pool.jsonl",
        [_top10("niah-val", [("niah-doc", "The event happened in 2009.")])],
    )
    niah_roles = _write_jsonl(
        tmp_path / "niah_roles.jsonl",
        [{"query_id": "niah-val", "role": "train-modelval", "component_id": "niah-val-component"}],
    )
    niah_components = _write_jsonl(
        tmp_path / "niah_components.jsonl",
        [{"query_id": "niah-val", "component_id": "niah-val-component"}],
    )
    niah_qa2d = _write_jsonl(
        tmp_path / "niah_qa2d.jsonl",
        [
            {
                "query_id": "niah-val",
                "answer": "2009",
                "declarative": "The event happened in 2009.",
                "answer_preserved": True,
                "model": QA2D_MODEL_ID,
                "revision": QA2D_REVISION,
            }
        ],
    )

    tw_queries = _write_jsonl(
        tmp_path / "tw_queries.jsonl",
        [
            {"schema_version": "1.0", "query_id": "tw-train", "text": "Who directed it?"},
            {"schema_version": "1.0", "query_id": "tw-val", "text": "Who wrote it?"},
        ],
    )
    tw_gold = _write_jsonl(
        tmp_path / "tw_gold.jsonl",
        [
            {
                "query_id": "tw-train",
                "reference_answers": ["Alice"],
                "relevant_document_ids": ["Film", "Alice"],
            },
            {
                "query_id": "tw-val",
                "reference_answers": ["Bob"],
                "relevant_document_ids": ["Book", "Bob"],
            },
        ],
    )
    tw_pool = _write_jsonl(
        tmp_path / "tw_pool.jsonl",
        [
            _top10("tw-train", [("Film", "Film was directed by Alice."), ("Alice", "Alice is French.")]),
            _top10("tw-val", [("Book", "Book was written by Bob."), ("Bob", "Bob is Canadian.")]),
        ],
    )
    tw_roles = _write_jsonl(
        tmp_path / "tw_roles.jsonl",
        [
            {"query_id": "tw-train", "role": "train-fit", "component_id": "tw-train-component"},
            {"query_id": "tw-val", "role": "train-modelval", "component_id": "tw-val-component"},
        ],
    )
    tw_components = _write_jsonl(
        tmp_path / "tw_components.jsonl",
        [
            {"query_id": "tw-train", "component_id": "tw-train-component"},
            {"query_id": "tw-val", "component_id": "tw-val-component"},
        ],
    )
    official = _write_jsonl(
        tmp_path / "tw_official.jsonl",
        [
            {
                "_id": "tw-train",
                "context": [["Film", ["Film was directed by Alice."]], ["Alice", ["Alice is French."]]],
                "supporting_facts": [["Film", 0], ["Alice", 0]],
                "evidences": [["Film", "director", "Alice"], ["Alice", "country", "French"]],
                "answer": "Alice",
            },
            {
                "_id": "tw-val",
                "context": [["Book", ["Book was written by Bob."]], ["Bob", ["Bob is Canadian."]]],
                "supporting_facts": [["Book", 0], ["Bob", 0]],
                "evidences": [["Book", "author", "Bob"], ["Bob", "country", "Canadian"]],
                "answer": "Bob",
            },
        ],
    )

    manifest = g200_v2.materialize(
        old_niah_train_cases_path=old_train,
        niah_queries_path=niah_queries,
        niah_candidate_pool_path=niah_pool,
        niah_gold_path=niah_gold,
        niah_modelval_roles_path=niah_roles,
        niah_modelval_components_path=niah_components,
        niah_modelval_qa2d_targets_path=niah_qa2d,
        twowiki_queries_path=tw_queries,
        twowiki_candidate_pool_path=tw_pool,
        twowiki_gold_path=tw_gold,
        twowiki_roles_path=tw_roles,
        twowiki_components_path=tw_components,
        twowiki_official_rows_path=official,
        output_dir=tmp_path / "data",
        min_niah_train_groups=1,
        min_niah_modelval_groups=1,
        min_twowiki_train_groups=1,
        min_twowiki_modelval_groups=1,
    )

    assert manifest["status"] == "PRE_AUDIT"
    assert manifest["stage"] == "G200R"
    assert manifest["target_construction"]["2wiki"] == "support_sentence_aligned_v1"
    assert manifest["gates"]["split_component_overlap_zero"] is True
    assert 0.10 <= manifest["unsupported_update_ratio"] <= 0.15
    train_rows = [
        json.loads(line)
        for line in (tmp_path / "data/train_cases.jsonl").read_text().splitlines()
    ]
    validation_rows = [
        json.loads(line)
        for line in (tmp_path / "data/validation_cases.jsonl").read_text().splitlines()
    ]
    assert any(row["case_id"] == "unsupported::2wiki::tw-train" for row in train_rows)
    assert any(row["case_id"] == "2wiki::tw-val" for row in validation_rows)
    twiki_train = next(row for row in train_rows if row["case_id"] == "2wiki::tw-train")
    assert "Film was directed by Alice" in twiki_train["variants"]["topk"]["target"]
    assert twiki_train["semantic_sentences"] == ["Film was directed by Alice.", "Alice is French."]
    assert twiki_train["target_construction"] == "support_sentence_aligned_v1"
