from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from full_flow_g110_finalize import _review_map  # noqa: E402


def test_review_map_rejects_duplicate_rows(tmp_path: Path) -> None:
    path = tmp_path / "review.jsonl"
    row = {
        "audit_row_id": "a1",
        "label": "TRUE_ROUTING_OR_ATTACHMENT",
        "confidence": "high",
        "reason": "fixture",
    }
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate"):
        _review_map(path, "A")


def test_review_map_rejects_unknown_labels(tmp_path: Path) -> None:
    path = tmp_path / "review.jsonl"
    path.write_text(
        json.dumps({"audit_row_id": "a1", "label": "OTHER"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid"):
        _review_map(path, "A")
