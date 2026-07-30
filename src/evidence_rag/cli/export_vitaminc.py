"""CLI: export VitaminC official test (and the decontamination log) as relation pairs.

Official test is emitted verbatim and never filtered — decontamination only ever removes rows
from train and dev. The removal log is archived because the leakage protocol requires it.

`datasets` is imported lazily inside `load_splits`, matching how the repo treats other heavy
optional dependencies, so importing this module never requires the extra to be installed.
"""

import argparse
import importlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from evidence_rag.relations.models import RelationPair
from evidence_rag.relations.vitaminc import decontaminate, to_pairs

DATASET = "tals/vitaminc"


def load_splits() -> Mapping[str, Sequence[Mapping[str, str]]]:
    """Load the three official splits. Seam for testing; the only place `datasets` is needed."""
    try:
        datasets = importlib.import_module("datasets")
    except ImportError as error:  # pragma: no cover - depends on optional runtime deps
        raise RuntimeError(
            "exporting VitaminC requires the optional 'datasets' package "
            "(pip install 'evidence-rag[relations]')"
        ) from error
    bundle = datasets.load_dataset(DATASET)
    return {split: bundle[split] for split in ("train", "validation", "test")}


def _write_pairs(path: Path, pairs: Sequence[RelationPair]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(
                {
                    "premise": pair.premise,
                    "hypothesis": pair.hypothesis,
                    "label": pair.label.value,
                    "group": pair.group,
                },
                sort_keys=True,
            )
            + "\n"
            for pair in pairs
        ),
        encoding="utf-8",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export VitaminC relation pairs")
    parser.add_argument("--out-test", required=True, type=Path)
    parser.add_argument("--out-train", type=Path)
    parser.add_argument("--out-dev", type=Path)
    parser.add_argument("--removed-log", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    raw = load_splits()
    splits = decontaminate(
        train=to_pairs(raw["train"]),
        dev=to_pairs(raw["validation"]),
        test=to_pairs(raw["test"]),
    )

    _write_pairs(arguments.out_test, splits.test)
    if arguments.out_train is not None:
        _write_pairs(arguments.out_train, splits.train)
    if arguments.out_dev is not None:
        _write_pairs(arguments.out_dev, splits.dev)

    report = {
        "n_train": len(splits.train),
        "n_dev": len(splits.dev),
        "n_test": len(splits.test),
        "removed_train_groups": list(splits.removed_train_groups),
        "removed_dev_groups": list(splits.removed_dev_groups),
    }
    arguments.removed_log.parent.mkdir(parents=True, exist_ok=True)
    arguments.removed_log.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
