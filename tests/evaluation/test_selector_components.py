import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

import evidence_rag.evaluation.selector_components as component_module
from evidence_rag.evaluation.selector_components import (
    CHAIN_REPRESENTATIVES_FILE,
    COMPONENT_MAP_FILE,
    MANIFEST_FILE,
    RECALL_REPRESENTATIVES_FILE,
    REPORT_FILE,
    ROLE_ASSIGNMENTS_FILE,
    AllowedKey,
    assign_query_balanced_folds,
    build_allowed_key_components,
    build_selector_component_artifacts,
    component_id_for_queries,
    freeze_selector_component_artifacts,
    verify_selector_component_artifacts,
)
from evidence_rag.evaluation.selector_risk import (
    COMPONENT_REPRESENTATIVE_SEED,
    ComponentCandidate,
    representative_digest,
)
from evidence_rag.infrastructure.datasets import JsonlDatasetAdapter


@dataclass(frozen=True)
class _Fixture:
    manifest: Path
    parents: Path
    candidates: Path
    assignment: Path | None = None


def _jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _candidate_docs(query_id: str, *leading: str) -> list[str]:
    values = list(leading)
    index = 0
    while len(values) < 20:
        document_id = f"{query_id}-noise-{index}"
        index += 1
        if document_id not in values:
            values.append(document_id)
    return values


def _fixture(
    tmp_path: Path,
    *,
    cases: dict[str, tuple[tuple[str, ...], list[str]]],
    parent_overrides: dict[str, str] | None = None,
    assignments: list[dict[str, object]] | None = None,
    dataset_id: str = "toy-dataset",
) -> _Fixture:
    root = tmp_path / "dataset"
    root.mkdir()
    parent_overrides = parent_overrides or {}
    document_ids = {
        document_id
        for gold, candidates in cases.values()
        for document_id in (*gold, *candidates)
    }
    if assignments is not None:
        document_ids.update(str(row["harmful_document_id"]) for row in assignments)

    _jsonl(
        root / "documents.jsonl",
        [
            {
                "schema_version": "1.0",
                "document_id": document_id,
                "text": f"{parent_overrides.get(document_id, f'parent-{document_id}')}\n\ntext",
                "source_uri": f"toy://{document_id}",
            }
            for document_id in sorted(document_ids)
        ],
    )
    _jsonl(
        root / "queries.jsonl",
        [
            {"schema_version": "1.0", "query_id": query_id, "text": f"Question {query_id}?"}
            for query_id in sorted(cases)
        ],
    )
    _jsonl(
        root / "gold_cases.jsonl",
        [
            {
                "query_id": query_id,
                "relevant_document_ids": list(cases[query_id][0]),
            }
            for query_id in sorted(cases)
        ],
    )
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "dataset_id": dataset_id,
                "dataset_version": "toy-v1",
                "split": "declared-source-split",
                "documents_file": "documents.jsonl",
                "queries_file": "queries.jsonl",
                "gold_cases_file": "gold_cases.jsonl",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    parents = _jsonl(
        root / "source_parent.jsonl",
        [
            {
                "document_id": document_id,
                "source_parent_id": parent_overrides.get(
                    document_id, f"parent-{document_id}"
                ),
            }
            for document_id in sorted(document_ids)
        ],
    )
    retriever = {
        "schema_version": "1.0",
        "name": "hybrid",
        "implementation_version": "hybrid-v1",
        "parameters_sha256": "a" * 64,
    }
    candidates = _jsonl(
        root / "candidate_sets.jsonl",
        [
            {
                "schema_version": "1.0",
                "query_id": query_id,
                "retriever": retriever,
                "candidates": [
                    {
                        "schema_version": "1.0",
                        "evidence_id": f"{query_id}-evidence-{rank}",
                        "document_id": document_id,
                        "chunk_id": f"{query_id}-chunk-{rank}",
                        "text": f"candidate {document_id}",
                        "source_uri": f"toy://{document_id}",
                        "retrieval_score": float(21 - rank),
                        "retrieval_rank": rank,
                    }
                    for rank, document_id in enumerate(cases[query_id][1], start=1)
                ],
            }
            for query_id in sorted(cases)
        ],
    )
    assignment_path = (
        _jsonl(root / "niah_assignments.jsonl", assignments)
        if assignments is not None
        else None
    )
    return _Fixture(manifest, parents, candidates, assignment_path)


def _rows(payload: bytes) -> list[dict[str, object]]:
    return [json.loads(line) for line in payload.decode("utf-8").splitlines()]


def _projection_hash(rows: list[dict[str, object]]) -> str:
    payload = "".join(
        json.dumps(row, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n"
        for row in rows
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def test_allowed_key_dsu_is_transitive_and_component_id_is_exact() -> None:
    keys: dict[str, set[AllowedKey]] = {
        "q1": {("query", "q1"), ("parent", "parent-shared")},
        "q2": {
            ("query", "q2"),
            ("parent", "parent-shared"),
            ("family", "family-shared"),
        },
        "q3": {("query", "q3"), ("family", "family-shared")},
        "q4": {("query", "q4"), ("parent", "parent-alone")},
    }

    components = build_allowed_key_components(keys)
    reversed_components = build_allowed_key_components(dict(reversed(tuple(keys.items()))))
    by_query = {
        query_id: component.component_id
        for component in components
        for query_id in component.query_ids
    }

    assert components == reversed_components
    assert by_query["q1"] == by_query["q2"] == by_query["q3"]
    assert by_query["q4"] != by_query["q1"]
    assert by_query["q1"] == hashlib.sha256(b"q1\nq2\nq3\n").hexdigest()
    assert by_query["q1"] == component_id_for_queries({"q3", "q1", "q2"})
    connected = next(component for component in components if "q1" in component.query_ids)
    assert connected.smallest_key == "family:family-shared"


def test_largest_first_fold_assignment_is_query_balanced_and_deterministic() -> None:
    keys: dict[str, set[AllowedKey]] = {}
    for query_id in ("a1", "a2", "a3"):
        keys[query_id] = {("query", query_id), ("parent", "parent-a")}
    for query_id in ("b1", "b2"):
        keys[query_id] = {("query", query_id), ("parent", "parent-b")}
    keys["c1"] = {("query", "c1")}
    keys["d1"] = {("query", "d1")}
    components = build_allowed_key_components(keys)

    folds, loads = assign_query_balanced_folds(components, n_folds=2)
    reversed_folds, reversed_loads = assign_query_balanced_folds(
        tuple(reversed(components)), n_folds=2
    )
    component_for = {
        query_id: component.component_id
        for component in components
        for query_id in component.query_ids
    }

    assert folds == reversed_folds
    assert loads == reversed_loads == (4, 3)
    assert folds[component_for["a1"]] == 0
    assert folds[component_for["b1"]] == 1
    assert folds[component_for["c1"]] == 1
    assert folds[component_for["d1"]] == 0


def test_twowiki_distractor_parent_never_creates_a_component_edge(tmp_path: Path) -> None:
    q1_candidates = _candidate_docs("q1", "gold-a", "shared-distractor")
    q2_candidates = _candidate_docs("q2", "gold-b", "shared-distractor")
    fixture = _fixture(
        tmp_path,
        cases={
            "q1": (("gold-a",), q1_candidates),
            "q2": (("gold-b",), q2_candidates),
        },
        parent_overrides={
            "gold-a": "official-parent-a",
            "gold-b": "official-parent-b",
            "shared-distractor": "same-distractor-parent",
        },
    )

    artifacts = build_selector_component_artifacts(
        dataset_kind="2wiki",
        source_split="dev",
        dataset_manifest_path=fixture.manifest,
        source_parent_path=fixture.parents,
        candidate_pool_path=fixture.candidates,
    )
    rows = _rows(artifacts.files[COMPONENT_MAP_FILE])
    by_query = {str(row["query_id"]): row for row in rows}

    assert by_query["q1"]["component_id"] != by_query["q2"]["component_id"]
    assert all(
        key["value"] != "same-distractor-parent"
        for row in rows
        for key in row["allowed_keys"]  # type: ignore[union-attr]
    )


def test_niah_any_allowed_key_connects_and_train_chain_structure_is_not_crc(
    tmp_path: Path,
) -> None:
    cases = {
        query_id: ((f"{query_id}-gold",), _candidate_docs(query_id, f"{query_id}-gold"))
        for query_id in ("q1", "q2", "q3", "q-unused")
    }
    parent_overrides = {
        "q1-gold": "parent-shared",
        "q1-harm": "parent-shared",
        "q2-gold": "parent-shared",
        "q2-harm": "parent-shared",
        "q3-gold": "parent-three",
        "q3-harm": "parent-three",
    }
    assignments: list[dict[str, object]] = [
        {
            "query_id": "q1",
            "required_document_ids": ["q1-gold"],
            "harmful_document_id": "q1-harm",
            "source_parent_ids": ["parent-shared"],
            "synthetic_family": "family-one",
        },
        {
            "query_id": "q2",
            "required_document_ids": ["q2-gold"],
            "harmful_document_id": "q2-harm",
            "source_parent_ids": ["parent-shared"],
            "synthetic_family": "family-link",
        },
        {
            "query_id": "q3",
            "required_document_ids": ["q3-gold"],
            "harmful_document_id": "q3-harm",
            "source_parent_ids": ["parent-three"],
            "synthetic_family": "family-link",
        },
    ]
    fixture = _fixture(
        tmp_path,
        cases=cases,
        parent_overrides=parent_overrides,
        assignments=assignments,
        dataset_id="niah-toy",
    )

    artifacts = build_selector_component_artifacts(
        dataset_kind="niah",
        source_split="train",
        dataset_manifest_path=fixture.manifest,
        source_parent_path=fixture.parents,
        candidate_pool_path=fixture.candidates,
        assignment_path=fixture.assignment,
    )
    component_rows = _rows(artifacts.files[COMPONENT_MAP_FILE])
    role_rows = _rows(artifacts.files[ROLE_ASSIGNMENTS_FILE])
    recall_rows = _rows(artifacts.files[RECALL_REPRESENTATIVES_FILE])
    chain_rows = _rows(artifacts.files[CHAIN_REPRESENTATIVES_FILE])

    assert len({row["component_id"] for row in component_rows}) == 1
    assert {row["query_id"] for row in component_rows} == {"q1", "q2", "q3"}
    assert {row["role"] for row in role_rows} == {"train-modelval"}
    assert all(row["chain_eligible_topk10"] is True for row in role_rows)
    assert recall_rows == []
    assert chain_rows == []
    report = json.loads(artifacts.files[REPORT_FILE])
    assert report["roles"]["train-modelval"]["chain_eligible_queries"] == 3
    assert report["evaluation_semantics"]["train_modelval_chain"].endswith("never CRC")


def test_topk10_chain_eligibility_and_crc_representative_binding(tmp_path: Path) -> None:
    c_candidates = ["c-gold-1", *[f"c-noise-{index}" for index in range(9)], "c-gold-2"]
    c_candidates = _candidate_docs("c", *c_candidates)
    fixture = _fixture(
        tmp_path,
        cases={
            "a": (("a-gold",), _candidate_docs("a", "a-gold")),
            "b": (("b-gold",), _candidate_docs("b", "b-gold")),
            "c": (("c-gold-1", "c-gold-2"), c_candidates),
            "d": (("d-gold",), _candidate_docs("d", "d-gold")),
        },
        dataset_id="twowiki-toy",
    )

    artifacts = build_selector_component_artifacts(
        dataset_kind="2wiki",
        source_split="dev",
        dataset_manifest_path=fixture.manifest,
        source_parent_path=fixture.parents,
        candidate_pool_path=fixture.candidates,
    )
    component_rows = _rows(artifacts.files[COMPONENT_MAP_FILE])
    role_rows = _rows(artifacts.files[ROLE_ASSIGNMENTS_FILE])
    recall_rows = _rows(artifacts.files[RECALL_REPRESENTATIVES_FILE])
    chain_rows = _rows(artifacts.files[CHAIN_REPRESENTATIVES_FILE])
    component_by_query = {
        str(row["query_id"]): str(row["component_id"]) for row in component_rows
    }
    role_by_query = {str(row["query_id"]): row for row in role_rows}

    assert role_by_query["c"]["chain_eligible_topk10"] is False
    assert {row["query_id"] for row in recall_rows} == {"a", "c"}
    assert {row["query_id"] for row in chain_rows} == {"a"}
    for row in (*recall_rows, *chain_rows):
        query_id = str(row["query_id"])
        assert row["component_id"] == component_by_query[query_id]
        candidate = ComponentCandidate(
            dataset_id=str(row["dataset_id"]),
            risk_name=str(row["risk_name"]),
            component_id=str(row["component_id"]),
            query_id=query_id,
        )
        assert row["representative_digest"] == representative_digest(
            candidate, seed=COMPONENT_REPRESENTATIVE_SEED
        )


def test_artifacts_are_sorted_hashed_write_once_and_recomputed_for_verify(
    tmp_path: Path,
) -> None:
    fixture = _fixture(
        tmp_path,
        cases={
            "z": (("z-gold",), _candidate_docs("z", "z-gold")),
            "a": (("a-gold",), _candidate_docs("a", "a-gold")),
        },
    )
    arguments = {
        "dataset_kind": "2wiki",
        "source_split": "dev",
        "dataset_manifest_path": fixture.manifest,
        "source_parent_path": fixture.parents,
        "candidate_pool_path": fixture.candidates,
    }
    first = build_selector_component_artifacts(**arguments)  # type: ignore[arg-type]
    second = build_selector_component_artifacts(**arguments)  # type: ignore[arg-type]

    assert first.files == second.files
    assert [row["query_id"] for row in _rows(first.files[COMPONENT_MAP_FILE])] == ["a", "z"]
    assert [row["query_id"] for row in _rows(first.files[ROLE_ASSIGNMENTS_FILE])] == ["a", "z"]
    manifest = json.loads(first.files[MANIFEST_FILE])
    report = json.loads(first.files[REPORT_FILE])
    assert manifest["inputs"]["dataset_manifest"]["sha256"] == hashlib.sha256(
        fixture.manifest.read_bytes()
    ).hexdigest()
    assert manifest["dataset_signature"]
    assert manifest["candidate_pool_status"] == "UNVERIFIED_TEST_INPUT"
    assert manifest["crossing"] == {
        "components_across_folds": 0,
        "components_across_roles": 0,
    }
    for filename in (
        COMPONENT_MAP_FILE,
        ROLE_ASSIGNMENTS_FILE,
        RECALL_REPRESENTATIVES_FILE,
        CHAIN_REPRESENTATIVES_FILE,
        REPORT_FILE,
    ):
        assert manifest["outputs"][filename]["sha256"] == hashlib.sha256(
            first.files[filename]
        ).hexdigest()

    component_rows = _rows(first.files[COMPONENT_MAP_FILE])
    role_rows = _rows(first.files[ROLE_ASSIGNMENTS_FILE])
    recall_rows = _rows(first.files[RECALL_REPRESENTATIVES_FILE])
    chain_rows = _rows(first.files[CHAIN_REPRESENTATIVES_FILE])
    component_projection = [
        {
            "component_id": row["component_id"],
            "dataset_id": row["dataset_id"],
            "query_id": row["query_id"],
        }
        for row in sorted(component_rows, key=lambda row: (row["dataset_id"], row["query_id"]))
    ]
    role_projection = [
        {
            "component_id": row["component_id"],
            "dataset_id": row["dataset_id"],
            "query_id": row["query_id"],
            "role": row["role"],
        }
        for row in sorted(role_rows, key=lambda row: (row["dataset_id"], row["query_id"]))
    ]

    def rep_projection(rows: list[dict[str, object]]) -> list[dict[str, object]]:
        return [
            {
                "component_id": row["component_id"],
                "dataset_id": row["dataset_id"],
                "query_id": row["query_id"],
                "risk_name": row["risk_name"],
                "role": row["source_role"],
            }
            for row in sorted(rows, key=lambda row: (row["component_id"], row["query_id"]))
        ]

    fingerprints = manifest["canonical_projection_fingerprints"]
    assert fingerprints == report["canonical_projection_fingerprints"]
    assert fingerprints["component_rows"]["sha256"] == _projection_hash(
        component_projection
    )
    assert fingerprints["role_rows"]["sha256"] == _projection_hash(role_projection)
    assert fingerprints["recall_representative_rows"]["sha256"] == _projection_hash(
        rep_projection(recall_rows)
    )
    assert fingerprints["chain_representative_rows"]["sha256"] == _projection_hash(
        rep_projection(chain_rows)
    )
    for role in ("crc-calibration", "decision-dev"):
        expected = [row for row in role_projection if row["role"] == role]
        assert fingerprints["role_rows_by_role"][role]["sha256"] == _projection_hash(
            expected
        )

    output = tmp_path / "frozen"
    freeze_selector_component_artifacts(output, first)
    verify_selector_component_artifacts(output_directory=output, **arguments)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="refusing to overwrite"):
        freeze_selector_component_artifacts(output, first)

    (output / COMPONENT_MAP_FILE).write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="differs from recomputation"):
        verify_selector_component_artifacts(output_directory=output, **arguments)  # type: ignore[arg-type]


def test_pool_directory_is_verified_against_v2_manifest_and_dataset_signature(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(
        tmp_path,
        cases={"q": (("gold",), _candidate_docs("q", "gold"))},
    )
    pool = tmp_path / "pool"
    pool.mkdir()
    (pool / "candidate_sets.jsonl").write_bytes(fixture.candidates.read_bytes())
    frozen_manifest = pool / "selector_candidate_pool_manifest_v2.json"
    frozen_manifest.write_text("{}\n", encoding="utf-8")
    dataset_signature = JsonlDatasetAdapter.load(fixture.manifest).dataset_signature
    calls: list[tuple[Path, Path]] = []

    def verify(pool_path: Path, dataset_path: Path) -> SimpleNamespace:
        calls.append((pool_path, dataset_path))
        return SimpleNamespace(data_role="2wiki-dev", dataset_signature=dataset_signature)

    monkeypatch.setattr(component_module, "verify_selector_candidate_pool_v2", verify)
    artifacts = build_selector_component_artifacts(
        dataset_kind="2wiki",
        source_split="dev",
        dataset_manifest_path=fixture.manifest,
        source_parent_path=fixture.parents,
        candidate_pool_path=pool,
    )
    manifest = json.loads(artifacts.files[MANIFEST_FILE])

    assert calls == [(pool, fixture.manifest)]
    assert manifest["candidate_pool_status"] == "VERIFIED_SELECTOR_POOL_V2"
    assert manifest["inputs"]["selector_candidate_pool_manifest_v2"][
        "sha256"
    ] == hashlib.sha256(frozen_manifest.read_bytes()).hexdigest()


def test_strict_pool_query_alignment_is_required(tmp_path: Path) -> None:
    fixture = _fixture(
        tmp_path,
        cases={
            "q1": (("q1-gold",), _candidate_docs("q1", "q1-gold")),
            "q2": (("q2-gold",), _candidate_docs("q2", "q2-gold")),
        },
    )
    lines = fixture.candidates.read_text(encoding="utf-8").splitlines()
    fixture.candidates.write_text(lines[0] + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="dataset-query/candidate-pool keys differ"):
        build_selector_component_artifacts(
            dataset_kind="2wiki",
            source_split="dev",
            dataset_manifest_path=fixture.manifest,
            source_parent_path=fixture.parents,
            candidate_pool_path=fixture.candidates,
        )
