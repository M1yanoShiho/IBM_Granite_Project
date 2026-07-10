"""Audit official selector datasets and write deterministic metadata manifests."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Sequence

from src.retrieval.selector_data import (
    ContractNLIAudit,
    FinanceBenchAudit,
    RAMDocsAudit,
    build_financebench_nested_folds,
    load_contractnli,
    load_financebench,
    load_ramdocs,
    sha256_file,
)


_FINANCE_QUESTIONS = Path("financebench/data/financebench_open_source.jsonl")
_FINANCE_DOCUMENTS = Path(
    "financebench/data/financebench_document_information.jsonl"
)
_FINANCE_PDFS = Path("financebench/pdfs")
_CONTRACT_DIR = Path("contract-nli/resources/contract-nli/contract-nli")
_RAMDOCS_FILE = Path("RAMDocs/RAMDocs_test.jsonl")


def _raw_files(data_root: Path, paths: Sequence[Path]) -> list[dict[str, str]]:
    return [
        {"path": path.relative_to(data_root).as_posix(), "sha256": sha256_file(path)}
        for path in sorted(paths, key=lambda item: item.relative_to(data_root).as_posix())
    ]


def _finance_audit(audit: FinanceBenchAudit) -> dict[str, int]:
    return asdict(audit)


def _contract_audit(audit: ContractNLIAudit) -> dict[str, object]:
    return {
        "splits": {
            split_name: {
                "annotation_choice_counts": summary.annotation_choice_counts,
                "document_count": summary.document_count,
                "evidence_span_count": summary.evidence_span_count,
                "hypothesis_count": summary.hypothesis_count,
            }
            for split_name in ("train", "dev", "test")
            for summary in (audit.split(split_name),)
        }
    }


def _ramdocs_audit(audit: RAMDocsAudit) -> dict[str, object]:
    return {
        "document_count": audit.document_count,
        "document_type_counts": audit.document_type_counts,
        "example_count": audit.example_count,
        "examples_without_wrong_answers": audit.examples_without_wrong_answers,
        "max_documents_per_query": audit.max_documents_per_query,
        "min_documents_per_query": audit.min_documents_per_query,
    }


def build_manifests(
    data_root: str | Path,
    *,
    financebench_commit: str,
    contractnli_commit: str,
    ramdocs_commit: str,
    seed: int = 42,
) -> tuple[dict[str, object], dict[str, object]]:
    """Audit raw inputs and return deterministic dataset and split manifests."""

    root = Path(data_root)
    finance_records, finance_audit = load_financebench(
        root / _FINANCE_QUESTIONS,
        root / _FINANCE_DOCUMENTS,
        root / _FINANCE_PDFS,
    )
    contract_records, contract_audit = load_contractnli(root / _CONTRACT_DIR)
    ramdocs_records, ramdocs_audit = load_ramdocs(root / _RAMDOCS_FILE)

    finance_pdf_paths = [
        path
        for path in (root / _FINANCE_PDFS).rglob("*")
        if path.is_file() and path.suffix.lower() == ".pdf"
    ]
    contract_paths = [root / _CONTRACT_DIR / f"{split}.json" for split in ("train", "dev", "test")]
    dataset_manifest: dict[str, object] = {
        "schema_version": "1.0",
        "datasets": {
            "contractnli": {
                "official_url": "https://stanfordnlp.github.io/contract-nli/",
                "repository_commit": contractnli_commit,
                "license": {
                    "identifier": "CC-BY-4.0",
                    "source": "contract-nli/LICENSE",
                },
                "raw_files": _raw_files(root, contract_paths),
                "audit": _contract_audit(contract_audit),
            },
            "financebench": {
                "official_url": "https://github.com/patronus-ai/financebench",
                "repository_commit": financebench_commit,
                "license": {
                    "identifier": "CC-BY-NC-4.0",
                    "source": "https://huggingface.co/datasets/PatronusAI/financebench",
                },
                "raw_files": _raw_files(
                    root,
                    [
                        root / _FINANCE_QUESTIONS,
                        root / _FINANCE_DOCUMENTS,
                        *finance_pdf_paths,
                    ],
                ),
                "audit": _finance_audit(finance_audit),
            },
            "ramdocs": {
                "official_url": "https://github.com/HanNight/RAMDocs",
                "repository_commit": ramdocs_commit,
                "license": {"identifier": "MIT", "source": "RAMDocs/LICENSE"},
                "raw_files": _raw_files(root, [root / _RAMDOCS_FILE]),
                "audit": _ramdocs_audit(ramdocs_audit),
            },
        },
    }
    split_manifest: dict[str, object] = {
        "schema_version": "1.0",
        "seed": seed,
        "datasets": {
            "contractnli": {
                "strategy": "official",
                "splits": {
                    split: [record.document_id for record in contract_records[split]]
                    for split in ("train", "dev", "test")
                },
                "zero_overlap_audit": {
                    "document_ids_disjoint": True,
                    "overlap_count": 0,
                },
            },
            "financebench": {
                "strategy": "nested_5_fold_company_grouped",
                "n_folds": 5,
                "outer_folds": build_financebench_nested_folds(
                    finance_records, seed=seed, n_folds=5
                ),
            },
            "niah": {
                "status": "pending_raw_data",
                "targets": {"train": 2000, "dev": 300, "test": 300},
                "pilot_train_target": 500,
            },
            "ramdocs": {
                "query_ids": [record.query_id for record in ramdocs_records],
                "protocols": {
                    "official": {
                        "candidate_pool": "official",
                        "context_size": 3,
                        "status": "ready",
                    },
                    "adapted": {
                        "candidate_pool_size": 20,
                        "context_size": 10,
                        "status": "pending_mining",
                    },
                },
            },
        },
    }
    return dataset_manifest, split_manifest


def _write_pretty_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m eval.prepare_selector_data",
        description="Audit official ML-selector datasets and write metadata-only manifests.",
    )
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--financebench-commit", required=True)
    parser.add_argument("--contractnli-commit", required=True)
    parser.add_argument("--ramdocs-commit", required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    dataset_manifest, split_manifest = build_manifests(
        args.data_root,
        financebench_commit=args.financebench_commit,
        contractnli_commit=args.contractnli_commit,
        ramdocs_commit=args.ramdocs_commit,
        seed=args.seed,
    )
    _write_pretty_json(args.out_dir / "dataset_manifest.json", dataset_manifest)
    _write_pretty_json(args.out_dir / "split_manifest.json", split_manifest)


if __name__ == "__main__":
    main()
