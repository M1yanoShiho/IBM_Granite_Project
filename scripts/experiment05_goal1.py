#!/usr/bin/env python3
"""Experiment 05 Goal 1 data inventory, exposure audit, and ID freezing.

This command never runs a Retriever, Selector, Generator, TRUE, or MiniCheck.
Its stdout is deliberately restricted to counts and hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from evidence_rag.evaluation.experiment05_data import (  # noqa: E402
    Goal1DataError,
    audit_bundle_pair,
    materialize_asqa_record,
    materialize_kilt_record,
    read_asqa_records,
    read_kilt_records,
    read_triviaqa_question_lookup,
    scan_exposure_paths,
    select_asqa_ids,
    select_kilt_ids,
)

DATASETS = ("kilt-nq", "kilt-tqa", "alce-asqa")


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _write_json(path: Path, value: object, *, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if mode is not None:
        path.chmod(mode)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _ordered_ids_sha256(query_ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(query_ids).encode("utf-8")).hexdigest()


def _ids_payload(dataset: str, split: str, query_ids: Sequence[str]) -> dict[str, object]:
    ids = list(query_ids)
    if len(ids) != len(set(ids)):
        raise Goal1DataError(f"{dataset} {split} contains duplicate canonical IDs")
    return {
        "schema_version": "experiment05.canonical_ids.v1",
        "dataset": dataset,
        "split": split,
        "count": len(ids),
        "ordered_ids_sha256": _ordered_ids_sha256(ids),
        "canonical_ids": ids,
    }


def _read_kilt_ids(path: Path) -> tuple[str, ...]:
    query_ids: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping) or not isinstance(value.get("id"), str):
                raise Goal1DataError(f"invalid KILT ID at line {line_number} in {path}")
            query_id = str(value["id"])
            if not query_id:
                raise Goal1DataError(f"blank KILT ID at line {line_number} in {path}")
            query_ids.append(query_id)
    if len(query_ids) != len(set(query_ids)):
        raise Goal1DataError(f"KILT split contains duplicate canonical IDs: {path}")
    return tuple(query_ids)


def inventory_stage(source_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Freeze source hashes and ID-only inventories without emitting example content."""

    sources = {
        "nq_train": source_dir / "nq-train-kilt.jsonl",
        "nq_formal": source_dir / "nq-dev-kilt.jsonl",
        "tqa_train": source_dir / "triviaqa-train_id-kilt.jsonl",
        "tqa_formal": source_dir / "triviaqa-dev_id-kilt.jsonl",
        "tqa_questions_train": source_dir / "triviaqa-unfiltered-nocontext-train.parquet",
        "tqa_questions_formal": source_dir
        / "triviaqa-unfiltered-nocontext-validation.parquet",
        "asqa": source_dir / "ALCE-data/asqa_eval_gtr_top100.json",
    }
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise Goal1DataError(f"missing Goal 1 sources: {missing}")

    trivia_questions = read_triviaqa_question_lookup(
        (sources["tqa_questions_train"], sources["tqa_questions_formal"])
    )
    id_sets = {
        ("kilt-nq", "train"): _read_kilt_ids(sources["nq_train"]),
        ("kilt-nq", "formal_pool"): _read_kilt_ids(sources["nq_formal"]),
        ("kilt-tqa", "train"): _read_kilt_ids(sources["tqa_train"]),
        ("kilt-tqa", "formal_pool"): _read_kilt_ids(sources["tqa_formal"]),
        ("alce-asqa", "all"): tuple(
            str(row["sample_id"]) for row in read_asqa_records(sources["asqa"])
        ),
    }

    datasets: dict[str, dict[str, object]] = defaultdict(dict)
    for (dataset, split), query_ids in id_sets.items():
        payload = _ids_payload(dataset, split, query_ids)
        ids_path = output_dir / f"{dataset}.{split}.ids.json"
        _write_json(ids_path, payload)
        row: dict[str, object] = {
            "count": payload["count"],
            "ordered_ids_sha256": payload["ordered_ids_sha256"],
            "ids_file": ids_path.name,
            "ids_file_sha256": _sha256(ids_path),
        }
        if dataset == "kilt-tqa":
            matched = sum(query_id in trivia_questions for query_id in query_ids)
            row["question_lookup_matched_count"] = matched
            row["question_lookup_missing_count"] = len(query_ids) - matched
        datasets[dataset][split] = row

    manifest: dict[str, Any] = {
        "schema_version": "experiment05.data_inventory.v1",
        "status": "FROZEN_SOURCE_IDS",
        "sources": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in sorted(sources.items())
        },
        "datasets": {dataset: datasets[dataset] for dataset in DATASETS},
    }
    _write_json(output_dir / "inventory_manifest.json", manifest)
    return manifest


def _read_ids(path: Path) -> tuple[str, ...]:
    value = json.loads(path.read_text(encoding="utf-8"))
    raw_ids = value.get("canonical_ids") if isinstance(value, Mapping) else None
    if not isinstance(raw_ids, list) or any(not isinstance(item, str) for item in raw_ids):
        raise Goal1DataError(f"invalid canonical ID file: {path}")
    ids = tuple(raw_ids)
    if len(ids) != len(set(ids)):
        raise Goal1DataError(f"duplicate canonical IDs in {path}")
    return ids


def exposure_stage(
    dataset: str,
    canonical_id_paths: Sequence[Path],
    source_paths: Sequence[Path],
    output_path: Path,
) -> dict[str, Any]:
    if dataset not in DATASETS:
        raise Goal1DataError(f"unsupported dataset: {dataset}")
    canonical_ids = {
        query_id for path in canonical_id_paths for query_id in _read_ids(path)
    }
    registry = scan_exposure_paths(source_paths, canonical_ids=canonical_ids)
    payload: dict[str, Any] = {
        "schema_version": "experiment05.exposure_registry.v1",
        "dataset": dataset,
        "canonical_id_count": len(canonical_ids),
        "exposure_count": len(registry.exposure_ids),
        "exposure_ids_sha256": _ordered_ids_sha256(registry.exposure_ids),
        "exposure_ids": list(registry.exposure_ids),
        "sources": list(registry.sources),
    }
    _write_json(output_path, payload)
    return payload


def reconstruct_asqa_exposure_stage(
    source_path: Path,
    output_path: Path,
    *,
    limit: int = 400,
    seed: int = 13,
    top_k: int = 5,
) -> dict[str, Any]:
    """Replay the recorded 400-query ASQA calibration draw when run rows are absent."""

    import g3_baseline_comparison as historical_g3

    raw = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise Goal1DataError("historical ASQA source must be a JSON list")
    cases = historical_g3.build_cases(raw, limit, random.Random(seed), top_k=top_k)
    if len(cases) != limit:
        raise Goal1DataError(
            f"historical ASQA sampler requested {limit} cases but reconstructed {len(cases)}"
        )
    exposure_ids = tuple(case.query_id for case in cases)
    if len(exposure_ids) != len(set(exposure_ids)):
        raise Goal1DataError("historical ASQA sampler reconstructed duplicate query IDs")
    sampler_path = Path(historical_g3.__file__).resolve()
    payload: dict[str, Any] = {
        "schema_version": "experiment05.exposure_registry.v1",
        "dataset": "alce-asqa",
        "canonical_id_count": len(raw),
        "exposure_count": len(exposure_ids),
        "exposure_ids_sha256": _ordered_ids_sha256(exposure_ids),
        "exposure_ids": list(exposure_ids),
        "sources": [
            {
                "path": str(source_path),
                "sha256": _sha256(source_path),
                "matched_count": len(exposure_ids),
                "matched_ids_sha256": _ordered_ids_sha256(sorted(exposure_ids)),
            }
        ],
        "reconstruction": {
            "method": "exact_historical_g3_build_cases_replay",
            "limit": limit,
            "seed": seed,
            "top_k": top_k,
            "sampler_script": str(sampler_path),
            "sampler_script_sha256": _sha256(sampler_path),
        },
    }
    _write_json(output_path, payload)
    return payload


def _read_exposure_ids(dataset: str, paths: Sequence[Path]) -> set[str]:
    ids: set[str] = set()
    for path in paths:
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("dataset") != dataset or not isinstance(value.get("exposure_ids"), list):
            raise Goal1DataError(f"invalid {dataset} exposure registry: {path}")
        ids.update(str(item) for item in value["exposure_ids"])
    return ids


def selection_stage(
    inventory_dir: Path,
    exposure_paths: Mapping[str, Sequence[Path]],
    output_dir: Path,
    *,
    development_count: int = 120,
    formal_count: int = 400,
) -> dict[str, Any]:
    selections: dict[str, Any] = {}
    for dataset in DATASETS:
        registry_paths = tuple(exposure_paths.get(dataset, ()))
        if not registry_paths:
            raise Goal1DataError(f"no historical exposure registry supplied for {dataset}")
        exposure_ids = _read_exposure_ids(dataset, registry_paths)
        if dataset == "alce-asqa":
            selection = select_asqa_ids(
                dataset=dataset,
                all_ids=_read_ids(inventory_dir / f"{dataset}.all.ids.json"),
                exposure_ids=exposure_ids,
                development_count=development_count,
                formal_count=formal_count,
            )
        else:
            selection = select_kilt_ids(
                dataset=dataset,
                train_ids=_read_ids(inventory_dir / f"{dataset}.train.ids.json"),
                formal_pool_ids=_read_ids(
                    inventory_dir / f"{dataset}.formal_pool.ids.json"
                ),
                exposure_ids=exposure_ids,
                development_count=development_count,
                formal_count=formal_count,
            )

        development_payload = _ids_payload(dataset, "development", selection.development_ids)
        formal_payload = _ids_payload(dataset, "formal", selection.formal_ids)
        development_path = output_dir / f"{dataset}.development.ids.json"
        formal_path = output_dir / f"{dataset}.formal.ids.json"
        _write_json(development_path, development_payload)
        _write_json(formal_path, formal_payload, mode=0o600)
        selections[dataset] = {
            "exposure_count": len(exposure_ids),
            "exposure_registry_sha256": [_sha256(path) for path in registry_paths],
            "development_count": len(selection.development_ids),
            "development_ordered_ids_sha256": selection.development_ordered_ids_sha256,
            "development_ids_file": development_path.name,
            "development_ids_file_sha256": _sha256(development_path),
            "formal_count": len(selection.formal_ids),
            "formal_ordered_ids_sha256": selection.formal_ordered_ids_sha256,
            "formal_ids_file": formal_path.name,
            "formal_ids_file_sha256": _sha256(formal_path),
            "development_formal_disjoint": set(selection.development_ids).isdisjoint(
                selection.formal_ids
            ),
            "formal_exposure_disjoint": set(selection.formal_ids).isdisjoint(exposure_ids),
        }

    manifest = {
        "schema_version": "experiment05.selection_manifest.v1",
        "status": "FROZEN",
        "selection_script_sha256": _sha256(Path(__file__).resolve()),
        "development_count_per_dataset": development_count,
        "formal_count_per_dataset": formal_count,
        "datasets": selections,
    }
    _write_json(output_dir / "selection_manifest.json", manifest, mode=0o600)
    return manifest


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]], *, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    path.chmod(mode)


def materialize_stage(
    source_dir: Path,
    selection_dir: Path,
    output_dir: Path,
    *,
    index_identities: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    """Build the frozen gold-free runtime and physically separate scorer sidecars."""

    for corpus in ("kilt", "dpr"):
        identity = index_identities.get(corpus)
        if not isinstance(identity, Mapping) or set(identity) != {
            "corpus_snapshot_id",
            "bm25_index_id",
            "dense_index_id",
        }:
            raise Goal1DataError(f"incomplete {corpus} index identity")

    question_lookup = read_triviaqa_question_lookup(
        (
            source_dir / "triviaqa-unfiltered-nocontext-train.parquet",
            source_dir / "triviaqa-unfiltered-nocontext-validation.parquet",
        )
    )
    selected_ids = {
        (dataset, split): _read_ids(
            selection_dir / f"{dataset}.{split}.ids.json"
        )
        for dataset in DATASETS
        for split in ("development", "formal")
    }
    raw_records = {
        ("kilt-nq", "development"): read_kilt_records(
            source_dir / "nq-train-kilt.jsonl",
            allowed_ids=set(selected_ids[("kilt-nq", "development")]),
        ),
        ("kilt-nq", "formal"): read_kilt_records(
            source_dir / "nq-dev-kilt.jsonl",
            allowed_ids=set(selected_ids[("kilt-nq", "formal")]),
        ),
        ("kilt-tqa", "development"): read_kilt_records(
            source_dir / "triviaqa-train_id-kilt.jsonl",
            question_lookup=question_lookup,
            allowed_ids=set(selected_ids[("kilt-tqa", "development")]),
        ),
        ("kilt-tqa", "formal"): read_kilt_records(
            source_dir / "triviaqa-dev_id-kilt.jsonl",
            question_lookup=question_lookup,
            allowed_ids=set(selected_ids[("kilt-tqa", "formal")]),
        ),
    }
    asqa_records = read_asqa_records(source_dir / "ALCE-data/asqa_eval_gtr_top100.json")
    raw_records[("alce-asqa", "development")] = asqa_records
    raw_records[("alce-asqa", "formal")] = asqa_records

    scorer_root = output_dir / "scorer_only"
    runtime_root = output_dir / "runtime"
    scorer_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    runtime_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    scorer_root.chmod(0o700)
    runtime_root.chmod(0o700)
    datasets: dict[str, Any] = {}
    for dataset in DATASETS:
        datasets[dataset] = {}
        for split in ("development", "formal"):
            query_ids = selected_ids[(dataset, split)]
            by_id = {str(record.get("id", record.get("sample_id", ""))): record for record in raw_records[(dataset, split)]}
            missing = [query_id for query_id in query_ids if query_id not in by_id]
            if missing:
                raise Goal1DataError(
                    f"{dataset} {split} is missing {len(missing)} frozen selected IDs"
                )
            runtime_rows: list[dict[str, Any]] = []
            sidecar_rows: list[dict[str, Any]] = []
            corpus = "dpr" if dataset == "alce-asqa" else "kilt"
            identity = index_identities[corpus]
            for query_id in query_ids:
                raw = by_id[query_id]
                if dataset == "alce-asqa":
                    runtime, sidecar = materialize_asqa_record(raw, **identity)
                else:
                    runtime, sidecar = materialize_kilt_record(
                        raw,
                        dataset=dataset,
                        **identity,
                    )
                runtime_rows.append(runtime)
                sidecar_rows.append(sidecar)
            runtime_path = runtime_root / split / f"{dataset}.jsonl"
            sidecar_path = scorer_root / split / f"{dataset}.jsonl"
            file_mode = 0o600 if split == "formal" else 0o640
            _write_jsonl(runtime_path, runtime_rows, mode=file_mode)
            _write_jsonl(sidecar_path, sidecar_rows, mode=0o600)
            sidecar_path.parent.chmod(0o700)
            runtime_path.parent.chmod(0o700 if split == "formal" else 0o750)
            isolation = audit_bundle_pair(
                runtime_path,
                sidecar_path,
                dataset=dataset,
                expected_query_ids=query_ids,
            )
            datasets[dataset][split] = {
                "count": len(query_ids),
                "ordered_ids_sha256": _ordered_ids_sha256(query_ids),
                "runtime_file": str(runtime_path.relative_to(output_dir)),
                "runtime_sha256": _sha256(runtime_path),
                "sidecar_file": str(sidecar_path.relative_to(output_dir)),
                "sidecar_sha256": _sha256(sidecar_path),
                "isolation": isolation,
            }
    manifest = {
        "schema_version": "experiment05.bundle_manifest.v1",
        "status": "FROZEN_RUNTIME_SIDECAR_ISOLATION_PASS",
        "materializer_sha256": _sha256(Path(__file__).resolve()),
        "index_identities": {
            name: dict(identity) for name, identity in sorted(index_identities.items())
        },
        "datasets": datasets,
    }
    _write_json(output_dir / "bundle_manifest.json", manifest, mode=0o600)
    return manifest


def _exposure_mapping(values: Sequence[str]) -> dict[str, tuple[Path, ...]]:
    grouped: dict[str, list[Path]] = defaultdict(list)
    for value in values:
        dataset, separator, raw_path = value.partition("=")
        if not separator or dataset not in DATASETS:
            raise Goal1DataError("--exposure must be DATASET=PATH")
        grouped[dataset].append(Path(raw_path))
    return {dataset: tuple(paths) for dataset, paths in grouped.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)

    inventory = subparsers.add_parser("inventory")
    inventory.add_argument("--source-dir", required=True, type=Path)
    inventory.add_argument("--output-dir", required=True, type=Path)

    exposure = subparsers.add_parser("scan-exposure")
    exposure.add_argument("--dataset", required=True, choices=DATASETS)
    exposure.add_argument("--canonical-ids", required=True, action="append", type=Path)
    exposure.add_argument("--path", required=True, action="append", type=Path)
    exposure.add_argument("--output", required=True, type=Path)

    replay = subparsers.add_parser("reconstruct-asqa-exposure")
    replay.add_argument("--source", required=True, type=Path)
    replay.add_argument("--output", required=True, type=Path)
    replay.add_argument("--limit", type=int, default=400)
    replay.add_argument("--seed", type=int, default=13)
    replay.add_argument("--top-k", type=int, default=5)

    selection = subparsers.add_parser("select")
    selection.add_argument("--inventory-dir", required=True, type=Path)
    selection.add_argument("--exposure", required=True, action="append")
    selection.add_argument("--output-dir", required=True, type=Path)

    materialize = subparsers.add_parser("materialize")
    materialize.add_argument("--source-dir", required=True, type=Path)
    materialize.add_argument("--selection-dir", required=True, type=Path)
    materialize.add_argument("--output-dir", required=True, type=Path)
    for corpus in ("kilt", "dpr"):
        materialize.add_argument(f"--{corpus}-corpus-id", required=True)
        materialize.add_argument(f"--{corpus}-bm25-index-id", required=True)
        materialize.add_argument(f"--{corpus}-dense-index-id", required=True)

    args = parser.parse_args()
    if args.stage == "inventory":
        result = inventory_stage(args.source_dir, args.output_dir)
        summary = {
            dataset: {split: row["count"] for split, row in result["datasets"][dataset].items()}
            for dataset in DATASETS
        }
    elif args.stage == "scan-exposure":
        result = exposure_stage(args.dataset, args.canonical_ids, args.path, args.output)
        summary = {
            "dataset": args.dataset,
            "canonical_id_count": result["canonical_id_count"],
            "exposure_count": result["exposure_count"],
            "exposure_ids_sha256": result["exposure_ids_sha256"],
        }
    elif args.stage == "reconstruct-asqa-exposure":
        result = reconstruct_asqa_exposure_stage(
            args.source,
            args.output,
            limit=args.limit,
            seed=args.seed,
            top_k=args.top_k,
        )
        summary = {
            "dataset": "alce-asqa",
            "exposure_count": result["exposure_count"],
            "exposure_ids_sha256": result["exposure_ids_sha256"],
            "method": result["reconstruction"]["method"],
        }
    elif args.stage == "select":
        result = selection_stage(
            args.inventory_dir, _exposure_mapping(args.exposure), args.output_dir
        )
        summary = {
            dataset: {
                "exposure_count": result["datasets"][dataset]["exposure_count"],
                "development_count": result["datasets"][dataset]["development_count"],
                "development_ordered_ids_sha256": result["datasets"][dataset][
                    "development_ordered_ids_sha256"
                ],
                "formal_count": result["datasets"][dataset]["formal_count"],
                "formal_ordered_ids_sha256": result["datasets"][dataset][
                    "formal_ordered_ids_sha256"
                ],
            }
            for dataset in DATASETS
        }
    else:
        result = materialize_stage(
            args.source_dir,
            args.selection_dir,
            args.output_dir,
            index_identities={
                "kilt": {
                    "corpus_snapshot_id": args.kilt_corpus_id,
                    "bm25_index_id": args.kilt_bm25_index_id,
                    "dense_index_id": args.kilt_dense_index_id,
                },
                "dpr": {
                    "corpus_snapshot_id": args.dpr_corpus_id,
                    "bm25_index_id": args.dpr_bm25_index_id,
                    "dense_index_id": args.dpr_dense_index_id,
                },
            },
        )
        summary = {
            dataset: {
                split: {
                    "count": result["datasets"][dataset][split]["count"],
                    "ordered_ids_sha256": result["datasets"][dataset][split][
                        "ordered_ids_sha256"
                    ],
                    "runtime_sha256": result["datasets"][dataset][split][
                        "runtime_sha256"
                    ],
                    "sidecar_sha256": result["datasets"][dataset][split][
                        "sidecar_sha256"
                    ],
                }
                for split in ("development", "formal")
            }
            for dataset in DATASETS
        }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
