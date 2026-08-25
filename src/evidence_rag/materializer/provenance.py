"""Provenance sidecar for injected counterfactuals (spec §6).

One MutationRecord per injected query; the eval harness reads counterfactual_document_id
to compute harmful-in-context. Convention-named jsonl (spec §8 decision 1); no manifest
schema change in v1.
"""

from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class MutationRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    query_id: str
    needle_document_id: str
    counterfactual_document_id: str
    gold_value: str
    gold_alias_used: str
    replacement_value: str
    string_class: str
    seed: int
    char_span: tuple[int, int]
    text_hash_before: str
    text_hash_after: str
    answer_bank_hash: str


def write_provenance(path: Path, records: Iterable[MutationRecord]) -> None:
    lines = [record.model_dump_json() for record in records]
    Path(path).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def read_provenance(path: Path) -> tuple[MutationRecord, ...]:
    text = Path(path).read_text(encoding="utf-8")
    return tuple(
        MutationRecord.model_validate_json(line) for line in text.splitlines() if line.strip()
    )
