"""Official Provence context-pruning adapter for Experiment 04."""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from typing import Any, Protocol

from evidence_rag.contracts.models import (
    CandidateSet,
    EvidenceCandidate,
    Query,
    SelectedEvidenceSet,
    SelectionItem,
    SelectionResult,
)


class PassagePruner(Protocol):
    def prune(self, *, question: str, title: str, text: str) -> str: ...


class ProvencePassagePruner:
    """Pinned official remote-code interface with the registered conservative settings."""

    def __init__(
        self,
        *,
        model_id: str,
        revision: str,
        threshold: float = 0.1,
        always_select_title: bool = True,
        reorder: bool = False,
        local_files_only: bool = False,
    ) -> None:
        if not revision:
            raise ValueError("Provence revision must be pinned")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Provence threshold must be in [0, 1]")
        if reorder:
            raise ValueError("Experiment 04 freezes Provence reorder=False")
        self.model_id = model_id
        self.revision = revision
        self.threshold = threshold
        self.always_select_title = always_select_title
        self.reorder = reorder
        self.local_files_only = local_files_only
        self.model = self._load()

    def _load(self) -> Any:
        try:
            transformers = importlib.import_module("transformers")
        except ImportError as exc:  # pragma: no cover - optional runtime dependency
            raise RuntimeError("Provence requires transformers") from exc
        return transformers.AutoModel.from_pretrained(
            self.model_id,
            revision=self.revision,
            trust_remote_code=True,
            token=os.getenv("HUGGINGFACE_API_KEY") or None,
            cache_dir=os.getenv("MODEL_CACHE_DIR") or None,
            local_files_only=self.local_files_only,
        )

    def prune(self, *, question: str, title: str, text: str) -> str:
        output = self.model.process(
            question,
            text,
            title=title or None,
            threshold=self.threshold,
            always_select_title=self.always_select_title,
            reorder=self.reorder,
            enable_warnings=False,
        )
        if not isinstance(output, dict) or not isinstance(output.get("pruned_context"), str):
            raise ValueError("Provence returned an invalid process result")
        return " ".join(output["pruned_context"].split())


@dataclass(frozen=True)
class PrunedSelection:
    selection: SelectionResult
    selected: SelectedEvidenceSet
    dropped_empty: int


class ProvenceSelector:
    """Preserve retrieval order while replacing each passage by Provence output."""

    def __init__(self, pruner: PassagePruner) -> None:
        self.pruner = pruner
        self.last_pruned: PrunedSelection | None = None

    def select_with_context(
        self,
        query: Query,
        candidates: CandidateSet,
        max_selected: int,
    ) -> PrunedSelection:
        if query.query_id != candidates.query_id:
            raise ValueError("query and candidates query IDs differ")
        if max_selected <= 0:
            raise ValueError("max_selected must be positive")
        retained: list[tuple[EvidenceCandidate, float]] = []
        dropped_empty = 0
        for candidate in sorted(
            candidates.candidates,
            key=lambda item: (item.retrieval_rank, item.evidence_id),
        )[:max_selected]:
            pruned_text = self.pruner.prune(
                question=query.text,
                title="",
                text=candidate.text,
            )
            if not pruned_text:
                dropped_empty += 1
                continue
            retained.append(
                (
                    EvidenceCandidate(
                        evidence_id=candidate.evidence_id,
                        document_id=candidate.document_id,
                        chunk_id=candidate.chunk_id,
                        text=pruned_text,
                        source_uri=candidate.source_uri,
                        retrieval_score=candidate.retrieval_score,
                        retrieval_rank=candidate.retrieval_rank,
                        metadata=candidate.metadata,
                    ),
                    candidate.retrieval_score,
                )
            )
        selection = SelectionResult(
            query_id=query.query_id,
            items=tuple(
                SelectionItem(
                    evidence_id=candidate.evidence_id,
                    selection_score=score,
                    selection_rank=rank,
                )
                for rank, (candidate, score) in enumerate(retained, start=1)
            ),
        )
        result = PrunedSelection(
            selection=selection,
            selected=SelectedEvidenceSet(
                query_id=query.query_id,
                evidence=tuple(candidate for candidate, _score in retained),
            ),
            dropped_empty=dropped_empty,
        )
        self.last_pruned = result
        return result

    def select(
        self,
        query: Query,
        candidates: CandidateSet,
        max_selected: int,
    ) -> SelectionResult:
        return self.select_with_context(query, candidates, max_selected).selection
