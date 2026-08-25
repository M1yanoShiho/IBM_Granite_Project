"""Query-local relation graph and `independent_support` (design §2.3, M0 §2.2).

A passage x claim bipartite graph over one query's rerank window. Votes are counted by distinct
`source_parent_id`, never by passage and never by `document_id`: dpr-w100 gives every 100-word
passage its own document_id, so counting document_ids lets several passages of one Wikipedia
article cast several votes, which violates the definition of `independent_support` itself
(M0 §3.4).

This module supplies gate conditions 1 and 2 and the integer votes conditions 3 and 4 read. It
deliberately stops there: margin and support_cap stay in the single frozen four-condition
decision the two arms share, so Graph 2.0 changes only how edges are built (D1=A). Nothing here
imports from `evidence_rag.selector` — the Relation Builder has to run standalone for Gate 0B,
where no selector exists.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from evidence_rag.relations.models import ClaimNode, RelationEdge, RelationLabel
from evidence_rag.relations.predictor import RelationPredictor

ConflictMode = Literal["distinct_cluster", "refutes_edge"]


@dataclass(frozen=True)
class PassageNode:
    """One passage vertex.

    `source_parent_id` is required with no default so that a caller cannot silently fall back to
    counting document_ids: under `support_unit=parent` a missing sidecar must be an error, never
    a quiet reversion to the old tally (M0 §3.4). Resolution itself belongs to the caller
    (`materializer.source_parent.ParentIndex`); this package stays free of I/O.
    """

    passage_id: str
    text: str
    source_parent_id: str


@dataclass(frozen=True)
class ClaimCluster:
    """A mutually-entailing group of claims plus the votes it holds.

    Not part of the frozen schema — this is the graph's own output type, the Graph 2.0 analogue
    of the selector's `AnswerCluster`. `member_ids` are the passages holding a SUPPORTS edge into
    the cluster, deduplicated and in window order; `independent_support` counts their distinct
    parents, which is why it can be smaller than `len(member_ids)`.
    """

    cluster_id: str
    answer: str
    claim_ids: tuple[str, ...]
    member_ids: tuple[str, ...]
    independent_support: int


@dataclass(frozen=True)
class ParentResolution:
    """How the support unit actually resolved for this query — a G-PQ line (M0 §3.7).

    Nothing here can tell a correct `source_parent_id` from a `document_id` a caller mistakenly
    passed in: both are strings. What it can do is refuse to let the reversion be silent. Under
    `support_unit=parent` a pool of dpr-w100 passages must show FEWER parents than passages;
    `self_parented == n_passages` is the signature of the pre-§3.4 tally, where every 100-word
    passage of one article casts its own vote. High self-parenting is legitimate for a pool of
    genuinely distinct sources, so this is reported, never raised on.
    """

    n_passages: int
    n_parents: int
    self_parented: int


def predict_edges(
    passages: Sequence[PassageNode],
    claims: Sequence[ClaimNode],
    predictor: RelationPredictor,
) -> tuple[RelationEdge, ...]:
    """One batched passage x claim call, passage-major (design §2.4).

    The premise is the passage text and the hypothesis is the claim's full sentence — never the
    bare answer string, which would be degenerate input for a cross-encoder (design §2.4 rule 2).
    """
    grid = tuple((passage, claim) for passage in passages for claim in claims)
    predictions = predictor.predict([(passage.text, claim.hypothesis) for passage, claim in grid])
    return tuple(
        RelationEdge.from_prediction(passage.passage_id, claim.claim_id, prediction)
        for (passage, claim), prediction in zip(grid, predictions, strict=True)
    )


def _parent_of(parent_by_passage: Mapping[str, str], passage_id: str) -> str:
    """A voter with no PassageNode has no parent to count. Falling back to the passage id would
    hand it an independent vote, so this is an error rather than a guess (M0 §3.4)."""
    if passage_id not in parent_by_passage:
        raise ValueError(f"edge from unknown passage: {passage_id}")
    return parent_by_passage[passage_id]


class QueryLocalGraph:
    """Passage x claim bipartite graph for one query."""

    @classmethod
    def build(
        cls,
        *,
        passages: Sequence[PassageNode],
        claim_clusters: Sequence[Sequence[ClaimNode]],
        predictor: RelationPredictor,
        conflict_mode: ConflictMode = "distinct_cluster",
    ) -> "QueryLocalGraph":
        claims = tuple(claim for cluster in claim_clusters for claim in cluster)
        return cls(
            passages=passages,
            claim_clusters=claim_clusters,
            edges=predict_edges(passages, claims, predictor),
            conflict_mode=conflict_mode,
        )

    def __init__(
        self,
        *,
        passages: Sequence[PassageNode],
        claim_clusters: Sequence[Sequence[ClaimNode]],
        edges: Sequence[RelationEdge],
        conflict_mode: ConflictMode = "distinct_cluster",
    ) -> None:
        if conflict_mode not in ("distinct_cluster", "refutes_edge"):
            raise ValueError("conflict_mode must be 'distinct_cluster' or 'refutes_edge'")
        if conflict_mode == "refutes_edge":
            # M0 §9.7. The implementation below is intact and correct; what it needs no longer
            # exists. Failing here rather than in `competing_clusters` is deliberate: this arm
            # would otherwise run to completion and report an empty competitor set for every
            # candidate, which reads as "the ablation suppresses all dropping" rather than "the
            # arm is unrunnable" — a wrong finding, not a wrong number. Kept rather than deleted
            # so the arm returns intact if A1 is ever overturned.
            raise ValueError(
                "conflict_mode='refutes_edge' is not executable under protocol g2-proto-2: "
                "amendment A1 made the relation model binary (SUPPORTS / NOT_SUPPORTED), so no "
                "REFUTES edge is ever produced and this arm's competing-cluster condition can "
                "never be met. EXPERIMENT_TRACKER R036 is marked N/A for the same reason. Use "
                "conflict_mode='distinct_cluster' (the primary mode, which never read REFUTES), "
                "or reinstate three-class prediction under a new amendment first."
            )
        self.passages = tuple(passages)
        self.edges = tuple(edges)
        self.conflict_mode = conflict_mode
        parent_by_passage = {
            passage.passage_id: passage.source_parent_id for passage in self.passages
        }

        # One pass over the edges. Only SUPPORTS edges vote: a NOT_SUPPORTED edge (or a REFUTES /
        # UNKNOWN one read back from a pre-A1 cache) is recorded — abstention has to stay
        # measurable (M0 §3.6, G-AB) — but casts nothing, so a passage that declines to commit
        # can never be dropped (design §2.3). This is why A1's narrowing of the output space does
        # not touch `independent_support`: it only ever counted the SUPPORTS side.
        supporters: dict[str, list[str]] = {}
        self._supported_claims: dict[str, set[str]] = {}
        self._refuted_claims: dict[str, set[str]] = {}
        for edge in self.edges:
            if edge.label is RelationLabel.SUPPORTS:
                supporters.setdefault(edge.claim_id, []).append(edge.passage_id)
                self._supported_claims.setdefault(edge.passage_id, set()).add(edge.claim_id)
            elif edge.label is RelationLabel.REFUTES:
                self._refuted_claims.setdefault(edge.passage_id, set()).add(edge.claim_id)

        clusters: list[ClaimCluster] = []
        for claims in claim_clusters:
            if not claims:
                raise ValueError("empty claim cluster")
            claim_ids = tuple(claim.claim_id for claim in claims)
            member_ids = tuple(
                dict.fromkeys(
                    passage_id
                    for claim_id in claim_ids
                    for passage_id in supporters.get(claim_id, ())
                )
            )
            parents = {_parent_of(parent_by_passage, passage_id) for passage_id in member_ids}
            clusters.append(
                ClaimCluster(
                    cluster_id=claim_ids[0],
                    answer=claims[0].answer,
                    claim_ids=claim_ids,
                    member_ids=member_ids,
                    independent_support=len(parents),
                )
            )
        self.clusters = tuple(clusters)

        self._supported: dict[str, list[ClaimCluster]] = {}
        for cluster in self.clusters:
            for passage_id in cluster.member_ids:
                self._supported.setdefault(passage_id, []).append(cluster)

    @property
    def parent_resolution(self) -> ParentResolution:
        """See `ParentResolution`. Computed over every passage in the window, not only voters —
        a reverted support unit is a property of the pool, not of who happened to vote."""
        parents = [passage.source_parent_id for passage in self.passages]
        return ParentResolution(
            n_passages=len(self.passages),
            n_parents=len(set(parents)),
            self_parented=sum(
                1 for passage in self.passages
                if passage.source_parent_id == passage.passage_id
            ),
        )

    def supported_clusters(self, passage_id: str) -> tuple[ClaimCluster, ...]:
        """The clusters `passage_id` holds a SUPPORTS edge into, in cluster order."""
        return tuple(self._supported.get(passage_id, ()))

    def takes_position(self, passage_id: str) -> bool:
        """Gate condition 1 (M0 §2.2): c has at least one SUPPORTS edge.

        Invariant 1, newly frozen: supporting several mutually-non-entailing clusters is taking
        NO position, so condition 1 is false and c cannot be dropped. Distinct clusters are
        non-entailing by construction (clustering merged by mutual entailment), which is why
        "exactly one" is the whole check. This is the conservative direction — the design's
        failure mode is silence.
        """
        return len(self.supported_clusters(passage_id)) == 1

    def own_cluster(self, passage_id: str) -> ClaimCluster | None:
        """Invariant 2 (M0 §2.2): c's own cluster is whichever of the clusters IT SUPPORTS has
        the largest `independent_support` — never the pool leader c did not support.

        Invariant 1 has already stripped droppability from the ambiguous cases, so for the gate
        this only ever picks out the single supported cluster; the rule is still implemented as
        written because own support is reported for every candidate, dropped or not. Ties keep
        the earlier cluster (`max` is stable over cluster order), so the choice is deterministic
        rather than dictated by dict iteration order.
        """
        return max(
            self.supported_clusters(passage_id),
            key=lambda cluster: cluster.independent_support,
            default=None,
        )

    def competing_clusters(self, passage_id: str) -> tuple[ClaimCluster, ...]:
        """Gate condition 2 (M0 §2.2), in cluster order.

        `distinct_cluster` (primary): a competing cluster is any OTHER claim cluster. Clustering
        merged by mutual entailment, so distinct clusters are non-entailing by construction and
        no extra entailment check belongs here — that is what keeps this structurally parallel to
        Graph 1.0, with the edge construction as the only variable.

        `refutes_edge` (ablation): UNREACHABLE under protocol g2-proto-2 — the constructor
        refuses that mode (M0 §9.7). The branch is retained verbatim against A1 being overturned:
        the competing cluster must additionally hold a REFUTES edge toward the candidate's claim,
        emitted by one of its own members. "The candidate's claim" is read here as a claim the
        candidate SUPPORTS — the narrower of the readings the frozen text admits, chosen so this
        arm stays strictly stricter than the primary one and the failure direction stays pointed
        at silence.
        """
        own = self.own_cluster(passage_id)
        if own is None:
            return ()
        others = tuple(
            cluster for cluster in self.clusters if cluster.cluster_id != own.cluster_id
        )
        if self.conflict_mode == "distinct_cluster":
            return others
        own_claims = self._supported_claims.get(passage_id, set())
        return tuple(
            cluster
            for cluster in others
            if any(
                self._refuted_claims.get(member_id, set()) & own_claims
                for member_id in cluster.member_ids
            )
        )
