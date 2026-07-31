"""CPU rescoring of an E1 dump (M0 §3 G-PQ).

The dump carries every system-dependent quantity (the raw extracted answers) and every
system-independent one (pool order, document ids, gold-alias flags), so re-clustering under a
different equivalence and running paired tests never needs a GPU again. S6 lost its per-query
data precisely because this path did not exist.
"""

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evidence_rag.contracts.models import EvidenceCandidate
from evidence_rag.evaluation.cluster_eval import (
    ClusterEvalCase,
    ClusterEvalReport,
    aggregate,
    evaluate_case,
)

METRICS = ("missed_conflict", "false_conflict", "fixed_false_conflict", "needle_gold_recovery")

Equivalence = Callable[[str, str], bool] | None


@dataclass(frozen=True)
class DumpRow:
    query_id: str
    needle_document_id: str
    counterfactual_document_id: str
    gold_value: str
    gold_aliases: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    document_ids: tuple[str, ...]
    answers: tuple[str, ...]
    contains_gold_alias: tuple[bool, ...]

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "DumpRow":
        window = payload["window"]
        return cls(
            query_id=payload["query_id"],
            needle_document_id=payload["needle_document_id"],
            counterfactual_document_id=payload["counterfactual_document_id"],
            gold_value=payload["gold_value"],
            gold_aliases=tuple(payload["gold_aliases"]),
            evidence_ids=tuple(item["evidence_id"] for item in window),
            document_ids=tuple(item["document_id"] for item in window),
            answers=tuple(item["answer"] for item in window),
            contains_gold_alias=tuple(bool(item["contains_gold_alias"]) for item in window),
        )

    def window(self) -> tuple[EvidenceCandidate, ...]:
        """Rebuild minimal candidates for scoring.

        The dump records a gold-alias BOOLEAN rather than the passage, to stay compact. `text` is
        therefore synthesised so that `contains_alias` reproduces the recorded flag exactly: a
        gold alias when the flag was set, and a token containing no alias otherwise.
        """
        placeholder = self.gold_aliases[0] if self.gold_aliases else "_"
        return tuple(
            EvidenceCandidate(
                evidence_id=evidence_id,
                document_id=document_id,
                chunk_id=f"{document_id}::c0",
                text=placeholder if flag else "_",
                source_uri=f"dump://{document_id}",
                retrieval_score=1.0 / (index + 1),
                retrieval_rank=index + 1,
            )
            for index, (evidence_id, document_id, flag) in enumerate(
                zip(self.evidence_ids, self.document_ids, self.contains_gold_alias, strict=True)
            )
        )


def read_dump(path: Path) -> tuple[DumpRow, ...]:
    text = Path(path).read_text(encoding="utf-8")
    return tuple(
        DumpRow.from_json(json.loads(line)) for line in text.splitlines() if line.strip()
    )


def _case(row: DumpRow, equivalence: Equivalence) -> ClusterEvalCase:
    return evaluate_case(
        row.window(),
        row.answers,
        query_id=row.query_id,
        needle_document_id=row.needle_document_id,
        counterfactual_document_id=row.counterfactual_document_id,
        gold_value=row.gold_value,
        gold_aliases=row.gold_aliases,
        equivalence=equivalence,
    )


def rescore(rows: Sequence[DumpRow], *, equivalence: Equivalence) -> ClusterEvalReport:
    return aggregate([_case(row, equivalence) for row in rows])


def per_query_metric(
    rows: Sequence[DumpRow],
    *,
    metric: str,
    equivalence: Equivalence,
) -> dict[str, float | None]:
    """Per-query mapping for `paired_metric.compare_paired`.

    None means the query was not scored for this metric; `compare_paired` drops a query unless
    both arms scored it, so the distinction must survive.
    """
    if metric not in METRICS:
        raise ValueError(f"unknown metric: {metric}")
    result: dict[str, float | None] = {}
    for row in rows:
        value = getattr(_case(row, equivalence), metric)
        result[row.query_id] = None if value is None else float(value)
    return result
