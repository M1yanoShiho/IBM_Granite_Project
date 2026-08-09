import json
from pathlib import Path

from evidence_rag.cli.selector_beam_m0 import FROZEN_POOL_NAMES, main
from evidence_rag.infrastructure.datasets import JsonlDatasetAdapter
from evidence_rag.materializer.selector_beam_split import sha256_file


def _write_2wiki(root: Path, split: str, query_id: str) -> Path:
    root.mkdir()
    (root / "documents.jsonl").write_text(
        '{"document_id":"d","text":"Title\\n\\nbody","source_uri":"x"}\n',
        encoding="utf-8",
    )
    (root / "queries.jsonl").write_text(
        json.dumps({"query_id": query_id, "text": "q"}) + "\n", encoding="utf-8"
    )
    (root / "gold_cases.jsonl").write_text(
        json.dumps({"query_id": query_id, "relevant_document_ids": ["d"]}) + "\n",
        encoding="utf-8",
    )
    (root / "source_parent.jsonl").write_text(
        '{"document_id":"d","source_parent_id":"title"}\n', encoding="utf-8"
    )
    (root / "provenance.jsonl").write_text("{}\n", encoding="utf-8")
    manifest = {
        "dataset_id": "2wiki/multihop",
        "dataset_version": split,
        "split": split,
        "documents_file": "documents.jsonl",
        "queries_file": "queries.jsonl",
        "gold_cases_file": "gold_cases.jsonl",
        "schema_version": "1.0",
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_m0_report_accepts_clean_frozen_inputs(tmp_path: Path) -> None:
    niah = tmp_path / "niah.json"
    train_assignments = tmp_path / "niah_train_assignments.jsonl"
    dev_assignments = tmp_path / "niah_dev_assignments.jsonl"
    train_assignments.write_text('{"query_id":"train"}\n', encoding="utf-8")
    dev_assignments.write_text('{"query_id":"dev"}\n', encoding="utf-8")
    manifests = [
        _write_2wiki(tmp_path / name, name, f"q-{name}") for name in ("train", "dev", "heldout")
    ]
    niah_inputs = {
        name: {
            filename: sha256_file(manifest.parent / filename)
            for filename in (
                "manifest.json",
                "documents.jsonl",
                "queries.jsonl",
                "gold_cases.jsonl",
                "provenance.jsonl",
            )
        }
        for name, manifest in zip(("train", "dev", "sealed"), manifests, strict=True)
    }
    niah.write_text(
        json.dumps(
            {
                "counts": {"train": 1, "dev": 1, "sealed": 1},
                "overlap_after_filtering": {
                    pair: {axis: 0 for axis in ("query_id", "source_parent_id", "synthetic_family")}
                    for pair in ("train_dev", "train_sealed", "dev_sealed")
                },
                "outputs": {
                    "niah_train_assignments.jsonl": sha256_file(train_assignments),
                    "niah_dev_assignments.jsonl": sha256_file(dev_assignments),
                },
                "inputs": niah_inputs,
            }
        ),
        encoding="utf-8",
    )
    model_file = tmp_path / "model.safetensors"
    model_file.write_bytes(b"model")
    model_config = tmp_path / "beam.toml"
    model_config.write_text(
        f'[model]\nmodel_id="m"\nrevision="r"\nmodel_sha256="{sha256_file(model_file)}"\n',
        encoding="utf-8",
    )
    candidate_file = tmp_path / "candidate_sets.jsonl"
    candidate_file.write_text("frozen pool\n", encoding="utf-8")
    source_parent = tmp_path / "source_parent.jsonl"
    source_parent.write_text("parents\n", encoding="utf-8")
    signature_by_pool = {
        "niah-train": JsonlDatasetAdapter.load(manifests[0]).dataset_signature,
        "niah-dev": JsonlDatasetAdapter.load(manifests[1]).dataset_signature,
        "sealed600": JsonlDatasetAdapter.load(manifests[2]).dataset_signature,
        "2wiki-train": JsonlDatasetAdapter.load(manifests[0]).dataset_signature,
        "2wiki-dev": JsonlDatasetAdapter.load(manifests[1]).dataset_signature,
        "2wiki-heldout": JsonlDatasetAdapter.load(manifests[2]).dataset_signature,
    }
    pools: dict[str, Path] = {}
    for name, signature in signature_by_pool.items():
        pool = tmp_path / f"pool-{name}.json"
        pool.write_text(
            json.dumps(
                {
                    "dataset": {"signature": signature},
                    "candidate_pool": {
                        "top_n": 20,
                        "exact_top_n_rate": 1.0,
                        "query_count": 1,
                        "sha256": sha256_file(candidate_file),
                        "path": str(candidate_file),
                    },
                    "audit": {"unresolved_parent_count": 0, "required_recall_at_top_n": 1.0},
                    "retriever": {"name": "hybrid"},
                    "source_parent": {
                        "path": str(source_parent),
                        "sha256": sha256_file(source_parent),
                    },
                }
            ),
            encoding="utf-8",
        )
        pools[name] = pool
    output = tmp_path / "output"
    pool_arguments = tuple(
        argument
        for name in sorted(FROZEN_POOL_NAMES)
        for argument in ("--pool-manifest", f"{name}={pools[name]}")
    )
    assert (
        main(
            (
                "--niah-audit",
                str(niah),
                "--niah-train-assignments",
                str(train_assignments),
                "--niah-dev-assignments",
                str(dev_assignments),
                "--niah-train-manifest",
                str(manifests[0]),
                "--niah-dev-manifest",
                str(manifests[1]),
                "--niah-sealed-manifest",
                str(manifests[2]),
                "--twowiki-train",
                str(manifests[0]),
                "--twowiki-dev",
                str(manifests[1]),
                "--twowiki-heldout",
                str(manifests[2]),
                *pool_arguments,
                "--model-config",
                str(model_config),
                "--model-file",
                str(model_file),
                "--output",
                str(output),
            )
        )
        == 0
    )
    report = json.loads((output / "M0_REPORT.json").read_text(encoding="utf-8"))
    assert report["status"] == "PASS"
