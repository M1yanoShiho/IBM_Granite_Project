from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "scripts", ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import experiment05_goal1_index as index_cli  # noqa: E402


def test_normalize_stage_returns_only_audit_summary(tmp_path: Path) -> None:
    source = tmp_path / "kilt.jsonl"
    source.write_text(
        json.dumps(
            {
                "wikipedia_id": "1",
                "wikipedia_title": "Title",
                "text": ["Title", "private corpus text"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = index_cli.normalize_stage(
        argparse.Namespace(
            dataset="kilt-wikipedia-20190801",
            source_kind="kilt",
            source=source,
            output_dir=tmp_path / "normalized",
            max_words=100,
            shard_size=100,
        )
    )

    assert summary["passage_count"] == 1
    assert len(summary["ordered_passage_ids_sha256"]) == 64
    assert "private corpus text" not in json.dumps(summary)
