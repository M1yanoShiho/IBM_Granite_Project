"""Gated corroboration selector (spec §7).

Tolerant rerank stage (convex blend, unchanged from CorroborationSelector) followed
by a strict contrastive gate. The gate reads NO scores — only in-pool integer votes:
drop c iff (1) c extracted a valid answer, (2) a competing answer cluster exists,
(3) the winner's independent support beats c's by >= margin, (4) c's support is
<= support_cap. Failure direction is silence: fragmented voting shrinks margins and
the gate stops firing (spec §7.1).
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from evidence_rag.contracts.models import (
    CandidateSet,
    Query,
    SelectionItem,
    SelectionResult,
)
from evidence_rag.selector.answer_norm import is_valid_answer
from evidence_rag.selector.clusters import AnswerCluster, build_clusters
from evidence_rag.selector.corroboration import corroboration_scores, minmax
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
        self.alpha = alpha
        self.margin = margin
        self.support_cap = support_cap
        self.top_n = top_n
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
        has_valid_answer = is_valid_answer(answer)
        own = cluster_by_member.get(evidence_id)
        competitors = tuple(
            cluster for cluster in clusters if own is not None and cluster.answer != own.answer
        )
        winner = max(competitors, key=lambda cluster: cluster.independent_support, default=None)
        own_support = own.independent_support if own is not None else 0
        winner_support = winner.independent_support if winner is not None else 0
        has_competitor = winner is not None
        margin = winner_support - own_support if has_competitor else 0
        margin_met = has_competitor and margin >= self.margin
        isolation_met = own_support <= self.support_cap
        drop = has_valid_answer and has_competitor and margin_met and isolation_met
        return GateDecision(
            evidence_id=evidence_id,
            action="drop" if drop else "keep",
            answer=own.answer if own is not None else None,
            own_support=own_support,
            winner_answer=winner.answer if winner is not None else None,
            winner_support=winner_support,
            margin=margin,
            has_valid_answer=has_valid_answer,
            has_competitor=has_competitor,
            margin_met=margin_met,
            isolation_met=isolation_met,
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

        ranked_candidates = tuple(
            sorted(candidates.candidates, key=lambda item: item.retrieval_rank)
        )
        window = ranked_candidates[: self.top_n]
        tail = ranked_candidates[self.top_n :]
        extracted = self._engine.extract(query, window)
        corroboration = minmax(corroboration_scores(extracted.answers, extracted.parametric))
        relevance = minmax(tuple(candidate.retrieval_score for candidate in window))
        blended = tuple(
            self.alpha * relevance[index] + (1.0 - self.alpha) * corroboration[index]
            for index in range(len(window))
        )

        clusters = build_clusters(window, extracted.answers)
        cluster_by_member = {
            member_id: cluster for cluster in clusters for member_id in cluster.member_ids
        }
        dropped: set[str] = set()
        for index, candidate in enumerate(window):
            decision = self._decide(
                candidate.evidence_id,
                extracted.answers[index],
                clusters,
                cluster_by_member,
            )
            if decision.action == "drop":
                dropped.add(candidate.evidence_id)
            if self.on_gate_decision is not None:
                self.on_gate_decision(decision)

        surviving_window = tuple(
            window[index]
            for index in sorted(
                range(len(window)),
                key=lambda i: (-blended[i], window[i].retrieval_rank, window[i].evidence_id),
            )
            if window[index].evidence_id not in dropped
        )
        score_by_id = {
            candidate.evidence_id: blended[index]
            for index, candidate in enumerate(window)
        }
        output = (surviving_window + tail)[:max_selected]
        return SelectionResult(
            query_id=query.query_id,
            items=tuple(
                SelectionItem(
                    evidence_id=candidate.evidence_id,
                    selection_score=score_by_id.get(
                        candidate.evidence_id,
                        candidate.retrieval_score,
                    ),
                    selection_rank=rank,
                )
                for rank, candidate in enumerate(output, start=1)
            ),
        )
