"""Deterministic document -> source parent mapping (spec §3.4).

dpr-w100 splits each Wikipedia article into 100-word passages, so ONE article spans many
`document_id`s, while `base_loader._document_text` prepends the article title as the first
paragraph. Counting distinct `document_id`s therefore treats several passages of one article as
independent votes — the transcription inflation that `independent_support` exists to prevent.

This module recovers the parent deterministically. The rule is unit tested exhaustively because
Gate 0B requires it: SAME_SOURCE is a rule, never a prediction.
"""

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path


def normalize_parent(title: str) -> str:
    return " ".join(title.split()).casefold()


def parse_parent(document_text: str) -> str | None:
    """First paragraph = the title `_document_text` prepended. None when absent or blank."""
    head, separator, _ = document_text.partition("\n\n")
    if not separator:
        return None
    return normalize_parent(head) or None


@dataclass(frozen=True)
class ParentIndex:
    parent_by_document: Mapping[str, str]

    def parent_of(self, document_id: str) -> str:
        """Unresolved documents are their own parent, so a missing sidecar entry can never
        merge two genuinely distinct sources — it only fails to merge two related ones."""
        return self.parent_by_document.get(document_id, document_id)

    def n_unresolved(self, document_ids: Iterable[str]) -> int:
        return sum(1 for item in document_ids if item not in self.parent_by_document)


def write_parent_index(path: Path, parent_by_document: Mapping[str, str]) -> None:
    lines = [
        json.dumps({"document_id": key, "source_parent_id": value}, sort_keys=True)
        for key, value in sorted(parent_by_document.items())
    ]
    Path(path).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def read_parent_index(path: Path) -> ParentIndex:
    mapping: dict[str, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        document_id = row["document_id"]
        if document_id in mapping:
            raise ValueError(f"duplicate document_id in parent index: {document_id}")
        mapping[document_id] = row["source_parent_id"]
    return ParentIndex(parent_by_document=mapping)
