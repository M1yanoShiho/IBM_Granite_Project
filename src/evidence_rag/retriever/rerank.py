"""Cross-encoder reranking for the frozen Experiment 04 baseline."""

from __future__ import annotations

import importlib
import os
from collections.abc import Sequence
from typing import Any, Protocol

from evidence_rag.contracts.models import CandidateSet, EvidenceCandidate, Query
from evidence_rag.contracts.protocols import Retriever


class PairReranker(Protocol):
    def score(self, query: str, passages: Sequence[str]) -> Sequence[float]: ...


class GraniteCrossEncoderReranker:
    """Pinned Hugging Face sequence-classification view of Granite reranker r2."""

    def __init__(
        self,
        *,
        model_id: str,
        revision: str,
        device: str = "auto",
        max_length: int = 8192,
        local_files_only: bool = False,
    ) -> None:
        if not revision:
            raise ValueError("Granite reranker revision must be pinned")
        if max_length <= 0:
            raise ValueError("Granite reranker max_length must be positive")
        self.model_id = model_id
        self.revision = revision
        self.device = device
        self.max_length = max_length
        self.local_files_only = local_files_only
        self.tokenizer, self.model = self._load()

    def _load(self) -> tuple[Any, Any]:
        try:
            transformers = importlib.import_module("transformers")
        except ImportError as exc:  # pragma: no cover - optional runtime dependency
            raise RuntimeError("Granite reranking requires transformers") from exc
        token = os.getenv("HUGGINGFACE_API_KEY") or None
        cache_dir = os.getenv("MODEL_CACHE_DIR") or None
        shared = {
            "revision": self.revision,
            "token": token,
            "cache_dir": cache_dir,
            "local_files_only": self.local_files_only,
        }
        tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_id, **shared)
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            self.model_id,
            dtype="auto",
            device_map="auto" if self.device == "auto" else None,
            **shared,
        )
        if self.device != "auto":
            model = model.to(self.device)
        model.eval()
        return tokenizer, model

    def score(self, query: str, passages: Sequence[str]) -> tuple[float, ...]:
        if not passages:
            return ()
        try:
            torch = importlib.import_module("torch")
        except ImportError as exc:  # pragma: no cover - optional runtime dependency
            raise RuntimeError("Granite reranking requires torch") from exc
        encoded = self.tokenizer(
            [[query, passage] for passage in passages],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        model_device = getattr(self.model, "device", None)
        if model_device is not None:
            encoded = {
                key: value.to(model_device) if hasattr(value, "to") else value
                for key, value in encoded.items()
            }
        with torch.no_grad():
            raw = self.model(**encoded).logits.view(-1).float()
        values = raw.detach().cpu().tolist()
        return tuple(float(value) for value in values)


class RerankingRetriever:
    """Retrieve a fixed deep pool, cross-encode it, and expose only final TopK."""

    def __init__(self, base: Retriever, reranker: PairReranker, *, pool_size: int = 40) -> None:
        if pool_size <= 0:
            raise ValueError("reranker pool_size must be positive")
        self.base = base
        self.reranker = reranker
        self.pool_size = pool_size

    def retrieve(self, query: Query, top_k: int) -> CandidateSet:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if top_k > self.pool_size:
            raise ValueError("final top_k may not exceed the reranker pool_size")
        pool = self.base.retrieve(query, self.pool_size)
        if pool.query_id != query.query_id:
            raise ValueError("base retriever returned the wrong query ID")
        scores = tuple(
            float(value)
            for value in self.reranker.score(
                query.text,
                tuple(candidate.text for candidate in pool.candidates),
            )
        )
        if len(scores) != len(pool.candidates):
            raise ValueError("reranker returned a different number of scores than passages")
        ranked = sorted(
            zip(scores, pool.candidates, strict=True),
            key=lambda item: (-item[0], item[1].retrieval_rank, item[1].evidence_id),
        )
        candidates = tuple(
            EvidenceCandidate(
                evidence_id=candidate.evidence_id,
                document_id=candidate.document_id,
                chunk_id=candidate.chunk_id,
                text=candidate.text,
                source_uri=candidate.source_uri,
                retrieval_score=score,
                retrieval_rank=rank,
                metadata=candidate.metadata,
            )
            for rank, (score, candidate) in enumerate(ranked[:top_k], start=1)
        )
        return CandidateSet(query_id=query.query_id, candidates=candidates)
