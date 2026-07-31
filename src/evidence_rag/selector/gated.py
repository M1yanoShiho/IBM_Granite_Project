"""Gated corroboration selector (spec §7) and coverage variant (A2 spec).

Tolerant rerank stage (convex blend, unchanged from CorroborationSelector) followed
by a strict contrastive gate. The gate reads NO scores — only in-pool integer votes:
drop c iff (1) c extracted a valid answer, (2) a competing answer cluster exists,
(3) the winner's independent support beats c's by >= margin, (4) c's support is
<= support_cap. Failure direction is silence: fragmented voting shrinks margins and
the gate stops firing (spec §7.1).

GatedCoverageSelector reuses the same gate, then packs the survivors by set-level
coverage instead of truncating by blended score (A2 spec §5-§7).
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from evidence_rag.contracts.models import (
    CandidateSet,
    EvidenceCandidate,
    Query,
    SelectionItem,
    SelectionResult,
)
from evidence_rag.selector.answer_equivalence import lenient_equivalent
from evidence_rag.selector.answer_norm import canonicalize_answer, is_valid_answer
from evidence_rag.selector.clusters import AnswerCluster, build_clusters, build_clusters_lenient
from evidence_rag.selector.corroboration import corroboration_scores, minmax
from evidence_rag.selector.coverage import coverage_select
from evidence_rag.selector.extraction import AnswerExtractionEngine, TextGenerator


@dataclass(frozen=True)
class GateDecision:
    evidence_id: str
    action: Literal["keep", "drop"]
    answer: str | None
    own_support: int
    winner_answer: str | None
    winner_support: int
    margin: int
    has_valid_answer: bool
    has_competitor: bool
    margin_met: bool
    isolation_met: bool


def gate_decision(
    evidence_id: str,
    answer: str,
    clusters: Sequence[AnswerCluster],
    cluster_by_member: Mapping[str, AnswerCluster],
    *,
    margin: int,
    support_cap: int,
) -> GateDecision:
    """The frozen four-condition judgement (spec §7), as a pure function.

    Module-level so that offline analysis scores the SAME code the production gate runs.
    Re-implementing these conditions for analysis would let the two drift apart silently, which
    is precisely the class of defect this project keeps finding.
    """
    has_valid_answer = is_valid_answer(answer)
    own = cluster_by_member.get(evidence_id)
    competitors = tuple(
        cluster for cluster in clusters if own is not None and cluster.answer != own.answer
    )
    winner = max(competitors, key=lambda cluster: cluster.independent_support, default=None)
    own_support = own.independent_support if own is not None else 0
    winner_support = winner.independent_support if winner is not None else 0
    has_competitor = winner is not None
    difference = winner_support - own_support if has_competitor else 0
    margin_met = has_competitor and difference >= margin
    isolation_met = own_support <= support_cap
    drop = has_valid_answer and has_competitor and margin_met and isolation_met
    return GateDecision(
        evidence_id=evidence_id,
        action="drop" if drop else "keep",
        answer=own.answer if own is not None else None,
        own_support=own_support,
        winner_answer=winner.answer if winner is not None else None,
        winner_support=winner_support,
        margin=difference,
        has_valid_answer=has_valid_answer,
        has_competitor=has_competitor,
        margin_met=margin_met,
        isolation_met=isolation_met,
    )


@dataclass(frozen=True)
class GateResult:
    survivors: tuple[EvidenceCandidate, ...]
    blended_by_id: dict[str, float]
    answer_by_id: dict[str, str | None]


def _to_selection(
    query: Query,
    output: Sequence[EvidenceCandidate],
    blended_by_id: Mapping[str, float],
) -> SelectionResult:
    return SelectionResult(
        query_id=query.query_id,
        items=tuple(
            SelectionItem(
                evidence_id=candidate.evidence_id,
                selection_score=blended_by_id.get(
                    candidate.evidence_id, candidate.retrieval_score
                ),
                selection_rank=rank,
            )
            for rank, candidate in enumerate(output, start=1)
        ),
    )


class GatedCorroborationSelector:
    """Corroboration rerank plus contrastive drop gate (spec §5-§7)."""

    def __init__(
        self,
        answer_extractor: TextGenerator,
        *,
        alpha: float = 0.6,
        margin: int = 2,
        support_cap: int = 1,
        top_n: int = 20,
        equivalence: Literal["exact", "lenient"] = "exact",
        parent_by_document: Mapping[str, str] | None = None,
        use_parametric: bool = True,
        passage_chars: int = 600,
        on_gate_decision: Callable[[GateDecision], None] | None = None,
    ) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if margin < 1:
            raise ValueError("margin must be at least 1")
        if support_cap < 0:
            raise ValueError("support_cap must be non-negative")
        if top_n <= 0:
            raise ValueError("top_n must be positive")
        if equivalence not in ("exact", "lenient"):
            raise ValueError("equivalence must be 'exact' or 'lenient'")
        self.alpha = alpha
        self.margin = margin
        self.support_cap = support_cap
        self.top_n = top_n
        self.equivalence = equivalence
        self.parent_by_document = parent_by_document
        self.on_gate_decision = on_gate_decision
        self._engine = AnswerExtractionEngine(
            answer_extractor,
            passage_chars=passage_chars,
            use_parametric=use_parametric,
        )

    def _decide(
        self,
        evidence_id: str,
        answer: str,
        clusters: tuple[AnswerCluster, ...],
        cluster_by_member: dict[str, AnswerCluster],
    ) -> GateDecision:
        return gate_decision(
            evidence_id,
            answer,
            clusters,
            cluster_by_member,
            margin=self.margin,
            support_cap=self.support_cap,
        )

    def _gate(self, query: Query, candidates: CandidateSet) -> GateResult:
        ranked = tuple(sorted(candidates.candidates, key=lambda item: item.retrieval_rank))
        window = ranked[: self.top_n]
        tail = ranked[self.top_n :]
        extracted = self._engine.extract(query, window)
        corroboration = minmax(corroboration_scores(extracted.answers, extracted.parametric))
        relevance = minmax(tuple(candidate.retrieval_score for candidate in window))
        blended = [
            self.alpha * relevance[index] + (1.0 - self.alpha) * corroboration[index]
            for index in range(len(window))
        ]
        blended_by_id: dict[str, float] = {
            window[index].evidence_id: blended[index] for index in range(len(window))
        }
        clusters = (
            build_clusters_lenient(
                window,
                extracted.answers,
                lenient_equivalent,
                parent_by_document=self.parent_by_document,
            )
            if self.equivalence == "lenient"
            else build_clusters(
                window, extracted.answers, parent_by_document=self.parent_by_document
            )
        )
        cluster_by_member = {
            member_id: cluster for cluster in clusters for member_id in cluster.member_ids
        }
        dropped: set[str] = set()
        answer_by_id: dict[str, str | None] = {}
        for index, candidate in enumerate(window):
            raw = extracted.answers[index]
            answer_by_id[candidate.evidence_id] = (
                canonicalize_answer(raw) if is_valid_answer(raw) else None
            )
            decision = self._decide(candidate.evidence_id, raw, clusters, cluster_by_member)
            if decision.action == "drop":
                dropped.add(candidate.evidence_id)
            if self.on_gate_decision is not None:
                self.on_gate_decision(decision)

        reranked_window = tuple(
            window[index]
            for index in sorted(
                range(len(window)),
                key=lambda i: (-blended[i], window[i].retrieval_rank, window[i].evidence_id),
            )
            if window[index].evidence_id not in dropped
        )
        survivors = reranked_window + tail
        for candidate in tail:
            blended_by_id.setdefault(candidate.evidence_id, candidate.retrieval_score)
            answer_by_id.setdefault(candidate.evidence_id, None)
        return GateResult(
            survivors=survivors,
            blended_by_id=blended_by_id,
            answer_by_id=answer_by_id,
        )

    def select(
        self,
        query: Query,
        candidates: CandidateSet,
        max_selected: int,
    ) -> SelectionResult:
        if query.query_id != candidates.query_id:
            raise ValueError("query and candidates query IDs differ")
        if max_selected <= 0:
            raise ValueError("max_selected must be positive")
        if not candidates.candidates:
            return SelectionResult(query_id=query.query_id, items=())
        result = self._gate(query, candidates)
        output = result.survivors[:max_selected]
        return _to_selection(query, output, result.blended_by_id)


class GatedCoverageSelector(GatedCorroborationSelector):
    """Gate (drop) followed by set-level coverage packing of survivors (A2 spec)."""

    def select(
        self,
        query: Query,
        candidates: CandidateSet,
        max_selected: int,
    ) -> SelectionResult:
        if query.query_id != candidates.query_id:
            raise ValueError("query and candidates query IDs differ")
        if max_selected <= 0:
            raise ValueError("max_selected must be positive")
        if not candidates.candidates:
            return SelectionResult(query_id=query.query_id, items=())
        result = self._gate(query, candidates)
        output = coverage_select(
            result.survivors,
            blended_by_id=result.blended_by_id,
            answer_by_id=result.answer_by_id,
            query_text=query.text,
            max_selected=max_selected,
        )
        return _to_selection(query, output, result.blended_by_id)
