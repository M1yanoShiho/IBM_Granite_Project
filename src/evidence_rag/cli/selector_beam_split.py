"""Build the leakage-safe NIAH train/dev assignments used by Beam Selector."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Sequence
from pathlib import Path

from evidence_rag.materializer.selector_beam_split import (
    build_splits,
    load_assignments,
    sha256_file,
    write_assignments,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Freeze Beam Selector NIAH assignments")
    parser.add_argument("--train-source", required=True, type=Path)
    parser.add_argument("--dev-source", required=True, type=Path)
    parser.add_argument("--sealed-source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--git-root", type=Path, default=Path.cwd())
    return parser


def _git_commit(root: Path) -> str:
    return subprocess.run(
        ("git", "-C", str(root), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _source_hashes(directory: Path) -> dict[str, str]:
    return {
        name: sha256_file(directory / name)
        for name in (
            "manifest.json",
            "documents.jsonl",
            "queries.jsonl",
            "gold_cases.jsonl",
            "provenance.jsonl",
        )
    }


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    train = load_assignments(arguments.train_source)
    dev = load_assignments(arguments.dev_source)
    sealed = load_assignments(arguments.sealed_source)
    train_kept, dev_kept, report = build_splits(train=train, dev=dev, sealed=sealed)

    arguments.output.mkdir(parents=True, exist_ok=True)
    train_path = arguments.output / "niah_train_assignments.jsonl"
    dev_path = arguments.output / "niah_dev_assignments.jsonl"
    write_assignments(train_path, train_kept)
    write_assignments(dev_path, dev_kept)
    report.update(
        {
            "git_commit": _git_commit(arguments.git_root),
            "inputs": {
                "train": _source_hashes(arguments.train_source),
                "dev": _source_hashes(arguments.dev_source),
                "sealed": _source_hashes(arguments.sealed_source),
            },
            "outputs": {
                train_path.name: sha256_file(train_path),
                dev_path.name: sha256_file(dev_path),
            },
        }
    )
    report_path = arguments.output / "niah_split_audit.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"audit": str(report_path), "counts": report["counts"]}, sort_keys=True
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
