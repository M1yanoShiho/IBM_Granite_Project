import dataclasses
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import evidence_rag.materializer.selector_labels as label_module
from evidence_rag.evaluation.selector_components import (
    OUTPUT_FILES as COMPONENT_OUTPUT_FILES,
)
from evidence_rag.evaluation.selector_components import (
    ROLE_ASSIGNMENTS_FILE,
    SelectorComponentArtifacts,
)
from evidence_rag.infrastructure.datasets import JsonlDatasetAdapter
from evidence_rag.materializer.selector_labels import (
    LABELS_FILE,
    MANIFEST_FILE,
    REPORT_FILE,
    TextPair,
    build_selector_label_artifacts,
    freeze_selector_label_artifacts,
    project_text_pair,
    text_pair_sha256,
    verify_selector_label_artifacts,
)
from evidence_rag.materializer.selector_pool import (
    CANDIDATE_FILE,
    SELECTOR_POOL_MANIFEST_FILE,
)


@dataclass(frozen=True)
class _Fixture:
    dataset_kind: str
    manifest: Path
    source_parent: Path
    pool: Path
    components: Path
    assignment: Path | None
    provenance: Path | None
    clean_text: str | None = None
    counterfactual_text: str | None = None


def _jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    return path


def _candidate(query_id: str, document_id: str, rank: int, text: str) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "evidence_id": f"ev-{query_id}-{rank:02d}",
        "document_id": document_id,
        "chunk_id": f"chunk-{query_id}-{rank:02d}",
        "text": text,
        "source_uri": f"source-secret://{document_id}",
        "metadata": {
            "schema_version": "1.0",
            "source_type": "html",
            "file_name": f"provenance-secret-{document_id}",
        },
        "retrieval_score": 1.0 / rank,
        "retrieval_rank": rank,
    }


def _component_bundle(
    directory: Path,
    *,
    dataset_kind: str,
    dataset_id: str,
    dataset_signature: str,
    query_id: str,
    role: str,
) -> SelectorComponentArtifacts:
    role_bytes = (
        json.dumps(
            {
                "schema_version": "1.0",
                "dataset_id": dataset_id,
                "query_id": query_id,
                "component_id": f"component-{query_id}",
                "fold": 0 if role == "train-modelval" else 1,
                "role": role,
                "chain_eligible_topk10": True,
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode()
    files: dict[str, bytes] = {}
    directory.mkdir()
    for filename in COMPONENT_OUTPUT_FILES:
        payload = role_bytes if filename == ROLE_ASSIGNMENTS_FILE else b"{}\n"
        (directory / filename).write_bytes(payload)
        files[filename] = payload
    return SelectorComponentArtifacts(
        files=files,
        report={},
        manifest={
            "candidate_pool_manifest_verified": True,
            "source_split": "train",
            "dataset_kind": dataset_kind,
            "dataset_signature": dataset_signature,
        },
    )


def _make_fixture(
    tmp_path: Path, *, dataset_kind: str
) -> tuple[_Fixture, SelectorComponentArtifacts]:
    root = tmp_path / "dataset"
    root.mkdir(parents=True)
    query_id = "q-1"
    if dataset_kind == "niah":
        dataset_id = "niah/dpr-w100-nq"
        clean_text = "Page\n\nAlice won the controlled contest."
        counterfactual_text = "Page\n\nBob won the controlled contest."
        documents: dict[str, str] = {
            "needle": clean_text,
            "cf::q-1::needle": counterfactual_text,
            "required-2": "Second page\n\nAnother official fact.",
            "cf::other-query::other-needle": "Other page\n\nAnother query's twin.",
        }
        while len(documents) < 20:
            index = len(documents)
            documents[f"noise-{index:02d}"] = f"Noise {index}\n\nUnjudged passage {index}."
        candidate_ids = list(documents)
        gold_ids = ["needle", "required-2"]
    else:
        dataset_id = "2wiki/multihop"
        clean_text = None
        counterfactual_text = None
        documents = {"supporting": "Gold title\n\nOfficial supporting paragraph."}
        while len(documents) < 20:
            index = len(documents)
            documents[f"unjudged-{index:02d}"] = f"Title {index}\n\nUnjudged paragraph {index}."
        candidate_ids = list(documents)
        gold_ids = ["supporting"]

    _jsonl(
        root / "documents.jsonl",
        [
            {
                "schema_version": "1.0",
                "document_id": document_id,
                "text": text,
                "source_uri": f"dataset://{document_id}",
            }
            for document_id, text in documents.items()
        ],
    )
    _jsonl(
        root / "queries.jsonl",
        [{"schema_version": "1.0", "query_id": query_id, "text": "Who won?"}],
    )
    _jsonl(
        root / "gold_cases.jsonl",
        [
            {
                "query_id": query_id,
                "relevant_document_ids": gold_ids,
                **({"reference_answers": ["Alice"]} if dataset_kind == "niah" else {}),
            }
        ],
    )
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "dataset_id": dataset_id,
                "dataset_version": "toy-v1",
                "split": "train",
                "documents_file": "documents.jsonl",
                "queries_file": "queries.jsonl",
                "gold_cases_file": "gold_cases.jsonl",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    source_parent = _jsonl(
        root / "source_parent.jsonl",
        [
            {"document_id": document_id, "source_parent_id": f"parent-{document_id}"}
            for document_id in documents
        ],
    )

    pool = tmp_path / "pool"
    pool.mkdir()
    (pool / SELECTOR_POOL_MANIFEST_FILE).write_text("{}\n", encoding="utf-8")
    _jsonl(
        pool / CANDIDATE_FILE,
        [
            {
                "schema_version": "1.0",
                "query_id": query_id,
                "retriever": {
                    "schema_version": "1.0",
                    "name": "hybrid",
                    "implementation_version": "hybrid-v1",
                    "parameters_sha256": "a" * 64,
                },
                "candidates": [
                    _candidate(query_id, document_id, rank, documents[document_id])
                    for rank, document_id in enumerate(candidate_ids, start=1)
                ],
            }
        ],
    )

    assignment: Path | None = None
    provenance: Path | None = None
    if dataset_kind == "niah":
        assignment = _jsonl(
            root / "assignments.jsonl",
            [
                {
                    "query_id": query_id,
                    "required_document_ids": gold_ids,
                    "harmful_document_id": "cf::q-1::needle",
                    "source_parent_ids": ["page", "second page"],
                    "synthetic_family": "alice|bob|name-1",
                }
            ],
        )
        assert clean_text is not None and counterfactual_text is not None
        start = clean_text.index("Alice")
        provenance = _jsonl(
            root / "provenance.jsonl",
            [
                {
                    "query_id": query_id,
                    "needle_document_id": "needle",
                    "counterfactual_document_id": "cf::q-1::needle",
                    "gold_value": "alice",
                    "gold_alias_used": "Alice",
                    "replacement_value": "Bob",
                    "string_class": "name-1",
                    "seed": 42,
                    "char_span": [start, start + len("Alice")],
                    "text_hash_before": hashlib.sha256(clean_text.encode()).hexdigest(),
                    "text_hash_after": hashlib.sha256(counterfactual_text.encode()).hexdigest(),
                    "answer_bank_hash": "b" * 64,
                }
            ],
        )

    bundle = JsonlDatasetAdapter.load(manifest)
    components = tmp_path / "components"
    component_artifacts = _component_bundle(
        components,
        dataset_kind=dataset_kind,
        dataset_id=dataset_id,
        dataset_signature=bundle.dataset_signature,
        query_id=query_id,
        role="train-modelval" if dataset_kind == "niah" else "train-fit",
    )
    return (
        _Fixture(
            dataset_kind=dataset_kind,
            manifest=manifest,
            source_parent=source_parent,
            pool=pool,
            components=components,
            assignment=assignment,
            provenance=provenance,
            clean_text=clean_text,
            counterfactual_text=counterfactual_text,
        ),
        component_artifacts,
    )


def _install_upstream_verifiers(
    monkeypatch: pytest.MonkeyPatch,
    fixture: _Fixture,
    components: SelectorComponentArtifacts,
) -> dict[str, Any]:
    bundle = JsonlDatasetAdapter.load(fixture.manifest)
    state: dict[str, Any] = {
        "data_role": f"{fixture.dataset_kind}-train",
        "pool_calls": 0,
        "component_calls": 0,
    }

    def verify_pool(pool: Path, manifest: Path) -> SimpleNamespace:
        assert pool == fixture.pool
        assert manifest == fixture.manifest
        state["pool_calls"] += 1
        return SimpleNamespace(
            data_role=state["data_role"], dataset_signature=bundle.dataset_signature
        )

    def verify_components(**kwargs: object) -> SelectorComponentArtifacts:
        assert kwargs["output_directory"] == fixture.components
        assert kwargs["source_split"] == "train"
        assert kwargs["dataset_kind"] == fixture.dataset_kind
        assert kwargs["assignment_path"] == fixture.assignment
        state["component_calls"] += 1
        return components

    monkeypatch.setattr(label_module, "verify_selector_candidate_pool_v2", verify_pool)
    monkeypatch.setattr(label_module, "verify_selector_component_artifacts", verify_components)
    return state


def _build(fixture: _Fixture) -> label_module.SelectorLabelArtifacts:
    return build_selector_label_artifacts(
        dataset_kind=fixture.dataset_kind,  # type: ignore[arg-type]
        dataset_manifest_path=fixture.manifest,
        source_parent_path=fixture.source_parent,
        candidate_pool_path=fixture.pool,
        component_directory=fixture.components,
        assignment_path=fixture.assignment,
        provenance_path=fixture.provenance,
    )


def _rows(payload: bytes) -> dict[str, dict[str, object]]:
    values = [json.loads(line) for line in payload.decode().splitlines()]
    return {str(row["document_id"]): row for row in values}


def test_niah_truth_table_masks_every_unjudged_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, components = _make_fixture(tmp_path, dataset_kind="niah")
    state = _install_upstream_verifiers(monkeypatch, fixture, components)

    artifacts = _build(fixture)
    rows = _rows(artifacts.files[LABELS_FILE])

    assert (rows["needle"]["protect_label"], rows["needle"]["harm_label"]) == (1, 0)
    assert (
        rows["cf::q-1::needle"]["protect_label"],
        rows["cf::q-1::needle"]["harm_label"],
    ) == (0, 1)
    assert (
        rows["required-2"]["protect_label"],
        rows["required-2"]["harm_label"],
    ) == (1, None)
    assert (
        rows["cf::other-query::other-needle"]["protect_label"],
        rows["cf::other-query::other-needle"]["harm_label"],
    ) == (None, None)
    assert (rows["noise-04"]["protect_label"], rows["noise-04"]["harm_label"]) == (
        None,
        None,
    )
    assert rows["required-2"]["harm_mask"] is False
    assert rows["noise-04"]["protect_mask"] is False
    assert artifacts.report["counts"] == {
        "candidate_rows": 20,
        "queries": 1,
        "components": 1,
        "roles": {"train-modelval": 20},
        "label_pairs": {
            "protect=0,harm=1": 1,
            "protect=1,harm=0": 1,
            "protect=1,harm=mask": 1,
            "protect=mask,harm=mask": 17,
        },
        "protect_supervised": 3,
        "protect_masked": 17,
        "harm_supervised": 2,
        "harm_masked": 18,
        "both_supervised": 2,
        "both_masked": 17,
    }
    assert state["pool_calls"] == state["component_calls"] == 1


def test_twowiki_supporting_is_protect_positive_and_unjudged_is_fully_masked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, components = _make_fixture(tmp_path, dataset_kind="2wiki")
    _install_upstream_verifiers(monkeypatch, fixture, components)

    artifacts = _build(fixture)
    rows = _rows(artifacts.files[LABELS_FILE])

    assert (rows["supporting"]["protect_label"], rows["supporting"]["harm_label"]) == (
        1,
        None,
    )
    assert (rows["unjudged-01"]["protect_label"], rows["unjudged-01"]["harm_label"]) == (
        None,
        None,
    )
    assert artifacts.report["counts"] == {
        "candidate_rows": 20,
        "queries": 1,
        "components": 1,
        "roles": {"train-fit": 20},
        "label_pairs": {
            "protect=1,harm=mask": 1,
            "protect=mask,harm=mask": 19,
        },
        "protect_supervised": 1,
        "protect_masked": 19,
        "harm_supervised": 0,
        "harm_masked": 20,
        "both_supervised": 0,
        "both_masked": 19,
    }
    assert artifacts.report["niah_verified_clean_counterfactual_pairs"] == 0


def test_text_pair_projection_has_no_identifier_rank_source_or_provenance_fields() -> None:
    pair = project_text_pair(question="Question only", candidate_text="Candidate only")

    assert [field.name for field in dataclasses.fields(TextPair)] == [
        "question",
        "candidate_text",
    ]
    assert pair.as_tokenizer_pair() == ("Question only", "Candidate only")
    assert text_pair_sha256(pair) == text_pair_sha256(
        TextPair(question="Question only", candidate_text="Candidate only")
    )
    assert "source" not in repr(pair).lower()
    with pytest.raises(ValueError, match="question"):
        project_text_pair(question=" ", candidate_text="Candidate")


@pytest.mark.parametrize(
    ("target", "value", "message"),
    [
        ("provenance.text_hash_before", "0" * 64, "clean text hash mismatch"),
        ("provenance.char_span", [0, 5], "does not select gold_alias_used"),
        ("provenance.replacement_value", "Eve", "not an exact one-span replacement"),
        ("provenance.needle_document_id", "noise-04", "not official required"),
        ("provenance.gold_alias_used", "Mallory", "not an official reference answer"),
        ("assignment.harmful_document_id", "noise-04", "harmful document mismatch"),
    ],
)
def test_niah_pair_must_pass_assignment_gold_span_and_dual_hash_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
    value: object,
    message: str,
) -> None:
    fixture, components = _make_fixture(tmp_path, dataset_kind="niah")
    _install_upstream_verifiers(monkeypatch, fixture, components)
    kind, field = target.split(".")
    path = fixture.provenance if kind == "provenance" else fixture.assignment
    assert path is not None
    row = json.loads(path.read_text(encoding="utf-8"))
    row[field] = value
    _jsonl(path, [row])

    with pytest.raises(ValueError, match=message):
        _build(fixture)


def test_missing_or_duplicate_provenance_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, components = _make_fixture(tmp_path, dataset_kind="niah")
    _install_upstream_verifiers(monkeypatch, fixture, components)
    assert fixture.provenance is not None
    row = json.loads(fixture.provenance.read_text(encoding="utf-8"))
    row["query_id"] = "different-query"
    _jsonl(fixture.provenance, [row])
    with pytest.raises(ValueError, match="no provenance"):
        _build(fixture)

    row["query_id"] = "q-1"
    _jsonl(fixture.provenance, [row, row])
    with pytest.raises(ValueError, match="duplicate NIAH provenance"):
        _build(fixture)


def test_extra_provenance_must_still_belong_to_pinned_source_train_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, components = _make_fixture(tmp_path, dataset_kind="niah")
    _install_upstream_verifiers(monkeypatch, fixture, components)
    assert fixture.provenance is not None
    used = json.loads(fixture.provenance.read_text(encoding="utf-8"))
    outside = {**used, "query_id": "outside-source-train"}
    _jsonl(fixture.provenance, [used, outside])

    with pytest.raises(ValueError, match="outside the pinned source-train dataset"):
        _build(fixture)


def test_niah_model_visible_harmful_chunk_must_contain_replacement_without_gold_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, components = _make_fixture(tmp_path, dataset_kind="niah")
    _install_upstream_verifiers(monkeypatch, fixture, components)
    candidate_path = fixture.pool / CANDIDATE_FILE
    candidate_set = json.loads(candidate_path.read_text(encoding="utf-8"))
    harmful = next(
        item for item in candidate_set["candidates"] if item["document_id"] == "cf::q-1::needle"
    )
    harmful["text"] = "Page\n\nAn unrelated truncated chunk."
    _jsonl(candidate_path, [candidate_set])

    with pytest.raises(ValueError, match="omits the replacement"):
        _build(fixture)


def test_wrong_pool_role_and_unverified_r002_components_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, components = _make_fixture(tmp_path, dataset_kind="2wiki")
    state = _install_upstream_verifiers(monkeypatch, fixture, components)
    state["data_role"] = "2wiki-dev"
    with pytest.raises(ValueError, match="expected '2wiki-train'"):
        _build(fixture)

    state["data_role"] = "2wiki-train"
    bad = SelectorComponentArtifacts(
        files=components.files,
        report=components.report,
        manifest={**components.manifest, "candidate_pool_manifest_verified": False},
    )
    _install_upstream_verifiers(monkeypatch, fixture, bad)
    with pytest.raises(ValueError, match="not bound to a verified"):
        _build(fixture)


def test_manifest_pins_all_label_split_inputs_and_excludes_them_from_model_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, components = _make_fixture(tmp_path, dataset_kind="niah")
    _install_upstream_verifiers(monkeypatch, fixture, components)

    artifacts = _build(fixture)
    inputs = artifacts.manifest["inputs"]
    assert isinstance(inputs, dict)
    assert set(inputs) == {
        "dataset_manifest",
        "dataset_documents",
        "dataset_queries",
        "dataset_gold_cases",
        "source_parent_split_only",
        "candidate_pool",
        "selector_candidate_pool_manifest_v2",
        "r002_components",
        "niah_assignment_label_sidecar",
        "niah_provenance_label_sidecar",
    }
    assert set(inputs["r002_components"]) == set(COMPONENT_OUTPUT_FILES)
    rules = artifacts.manifest["rules"]
    assert isinstance(rules, dict)
    assert rules["model_input_fields"] == ["question", "candidate_text"]
    forbidden = rules["forbidden_model_input_fields"]
    assert "retrieval_rank" in forbidden
    assert "source_parent_id" in forbidden
    assert "provenance" in forbidden
    labels_text = artifacts.files[LABELS_FILE].decode()
    assert "source-secret" not in labels_text
    assert "provenance-secret" not in labels_text
    assert "retrieval_rank" not in labels_text


def test_build_is_deterministic_and_freeze_is_write_once_and_verifiable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, components = _make_fixture(tmp_path, dataset_kind="niah")
    _install_upstream_verifiers(monkeypatch, fixture, components)
    first = _build(fixture)
    second = _build(fixture)
    assert first.files == second.files

    output = tmp_path / "labels"
    manifest_path = freeze_selector_label_artifacts(output, first)
    assert manifest_path == output / MANIFEST_FILE
    assert set(path.name for path in output.iterdir()) == {LABELS_FILE, REPORT_FILE, MANIFEST_FILE}
    with pytest.raises(ValueError, match="refusing to overwrite"):
        freeze_selector_label_artifacts(output, first)

    verified = verify_selector_label_artifacts(
        output_directory=output,
        dataset_kind="niah",
        dataset_manifest_path=fixture.manifest,
        source_parent_path=fixture.source_parent,
        candidate_pool_path=fixture.pool,
        component_directory=fixture.components,
        assignment_path=fixture.assignment,
        provenance_path=fixture.provenance,
    )
    assert verified.files == first.files

    (output / LABELS_FILE).write_bytes(first.files[LABELS_FILE] + b"{}\n")
    with pytest.raises(ValueError, match="differs from recomputation"):
        verify_selector_label_artifacts(
            output_directory=output,
            dataset_kind="niah",
            dataset_manifest_path=fixture.manifest,
            source_parent_path=fixture.source_parent,
            candidate_pool_path=fixture.pool,
            component_directory=fixture.components,
            assignment_path=fixture.assignment,
            provenance_path=fixture.provenance,
        )


def test_dataset_specific_sidecars_are_mandatory_and_mutually_exclusive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    niah, niah_components = _make_fixture(tmp_path / "niah", dataset_kind="niah")
    _install_upstream_verifiers(monkeypatch, niah, niah_components)
    with pytest.raises(ValueError, match="both assignment and provenance"):
        build_selector_label_artifacts(
            dataset_kind="niah",
            dataset_manifest_path=niah.manifest,
            source_parent_path=niah.source_parent,
            candidate_pool_path=niah.pool,
            component_directory=niah.components,
            assignment_path=niah.assignment,
        )

    twowiki, twowiki_components = _make_fixture(tmp_path / "twowiki", dataset_kind="2wiki")
    _install_upstream_verifiers(monkeypatch, twowiki, twowiki_components)
    assert niah.assignment is not None
    with pytest.raises(ValueError, match="must not receive"):
        build_selector_label_artifacts(
            dataset_kind="2wiki",
            dataset_manifest_path=twowiki.manifest,
            source_parent_path=twowiki.source_parent,
            candidate_pool_path=twowiki.pool,
            component_directory=twowiki.components,
            assignment_path=niah.assignment,
        )
