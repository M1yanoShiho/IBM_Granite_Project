from collections.abc import Iterable, Mapping, Sequence
from typing import cast

import pytest

from evidence_rag.relations.graph import ConflictMode, PassageNode, QueryLocalGraph
from evidence_rag.relations.models import (
    ClaimNode,
    RelationEdge,
    RelationLabel,
    RelationPrediction,
)
from evidence_rag.relations.predictor import text_hash


class _FakePredictor:
    """A RelationPredictor that reads labels off a table. No model, no torch."""

    def __init__(self, labels: Mapping[tuple[str, str], RelationLabel]) -> None:
        self.labels = labels
        self.calls: list[tuple[tuple[str, str], ...]] = []

    def predict(self, pairs: Sequence[tuple[str, str]]) -> tuple[RelationPrediction, ...]:
        self.calls.append(tuple(pairs))
        return tuple(
            RelationPrediction(
                label=self.labels.get(pair, RelationLabel.UNKNOWN),
                confidence=1.0,
                model_version="fake-1",
                premise_hash=text_hash(pair[0]),
                hypothesis_hash=text_hash(pair[1]),
            )
            for pair in pairs
        )


def _passage(passage_id: str, parent: str) -> PassageNode:
    return PassageNode(
        passage_id=passage_id, text=f"passage {passage_id}", source_parent_id=parent
    )


def _claim(claim_id: str, answer: str) -> ClaimNode:
    return ClaimNode(
        claim_id=claim_id,
        answer=answer,
        hypothesis=f'The answer to the question "capital" is {answer}.',
    )


def _edge(passage_id: str, claim_id: str, label: RelationLabel) -> RelationEdge:
    return RelationEdge(
        passage_id=passage_id,
        claim_id=claim_id,
        label=label,
        confidence=1.0,
        model_version="fake-1",
        premise_hash="p" * 16,
        hypothesis_hash="h" * 16,
    )


def _graph(
    passages: Sequence[PassageNode],
    claim_clusters: Sequence[Sequence[ClaimNode]],
    *,
    supports: Iterable[tuple[str, str]] = (),
    refutes: Iterable[tuple[str, str]] = (),
    unknown: Iterable[tuple[str, str]] = (),
    conflict_mode: ConflictMode = "distinct_cluster",
) -> QueryLocalGraph:
    edges = (
        tuple(_edge(passage, claim, RelationLabel.SUPPORTS) for passage, claim in supports)
        + tuple(_edge(passage, claim, RelationLabel.REFUTES) for passage, claim in refutes)
        + tuple(_edge(passage, claim, RelationLabel.UNKNOWN) for passage, claim in unknown)
    )
    return QueryLocalGraph(
        passages=passages,
        claim_clusters=claim_clusters,
        edges=edges,
        conflict_mode=conflict_mode,
    )


def test_independent_support_counts_distinct_parents_not_passages() -> None:
    """dpr-w100 gives every 100-word passage its own document_id, so counting passages lets one
    Wikipedia article cast several votes — which violates the definition of independent_support
    itself (M0 §3.4)."""
    passages = (
        _passage("p1", "parent-a"),
        _passage("p2", "parent-a"),
        _passage("p3", "parent-b"),
    )
    claim = _claim("c1", "Paris")
    graph = _graph(
        passages,
        ((claim,),),
        supports=(("p1", "c1"), ("p2", "c1"), ("p3", "c1")),
    )
    assert graph.clusters[0].independent_support == 2


def test_a_supports_edge_is_what_makes_a_candidate_take_a_position() -> None:
    """Gate condition 1 remapped (M0 §2.2): `is_valid_answer(raw)` becomes "c has at least one
    SUPPORTS edge". A passage no longer has to extract the gold string itself — supporting a
    claim someone else raised is enough."""
    passages = (_passage("p1", "parent-a"),)
    claim = _claim("c1", "Paris")
    graph = _graph(passages, ((claim,),), supports=(("p1", "c1"),))
    assert graph.takes_position("p1") is True
    own = graph.own_cluster("p1")
    assert own is not None
    assert own.cluster_id == "c1"


def test_supporting_two_non_entailing_clusters_is_taking_no_position() -> None:
    """Newly frozen in M0 §2.2 invariant 1: c supporting several mutually-non-entailing clusters
    has taken no position, so condition 1 is FALSE and c is NOT droppable. Distinct clusters are
    non-entailing by construction, since clustering already merged by mutual entailment."""
    passages = (_passage("p1", "parent-a"), _passage("p2", "parent-b"))
    paris, lyon = _claim("c1", "Paris"), _claim("c2", "Lyon")
    graph = _graph(
        passages,
        ((paris,), (lyon,)),
        supports=(("p1", "c1"), ("p1", "c2"), ("p2", "c2")),
    )
    assert graph.supported_clusters("p1") == graph.clusters
    assert graph.takes_position("p1") is False


def test_own_cluster_is_the_best_supported_one_not_the_pool_leader() -> None:
    """Newly frozen in M0 §2.2 invariant 2: c's own cluster is whichever of the clusters IT
    SUPPORTS has the largest independent_support. A bigger cluster that c does not support is
    not c's; that is the difference between "own support" and "winner support" in condition 3."""
    passages = (
        _passage("p1", "parent-a"),
        _passage("p2", "parent-b"),
        _passage("p3", "parent-c"),
    )
    lyon, paris = _claim("c1", "Lyon"), _claim("c2", "Paris")
    graph = _graph(
        passages,
        ((lyon,), (paris,)),
        supports=(("p1", "c1"), ("p1", "c2"), ("p2", "c2"), ("p3", "c2")),
    )
    lyon_cluster, paris_cluster = graph.clusters
    assert (lyon_cluster.independent_support, paris_cluster.independent_support) == (1, 3)

    own = graph.own_cluster("p1")
    assert own is not None
    assert own.cluster_id == "c2"

    isolated = graph.own_cluster("p2")
    assert isolated is not None
    assert isolated.cluster_id == "c2"


def test_own_cluster_of_a_lone_supporter_is_not_the_bigger_cluster_it_ignores() -> None:
    """The discriminating case for invariant 2, which the test above does not reach: a passage
    supporting ONLY the smaller cluster, while a larger cluster it never supported sits in the
    same pool. Above, every passage either supports the pool leader or supports both, so an
    implementation that maxed over ALL clusters instead of the supported ones would still agree
    with it. That mutation is currently killed only incidentally, by the `distinct_cluster` test;
    invariant 2 is newly frozen in M0 §2.2 and needs a guard that names it.
    """
    passages = (
        _passage("p1", "parent-a"),
        _passage("p2", "parent-b"),
        _passage("p3", "parent-c"),
        _passage("p4", "parent-d"),
    )
    lyon, paris = _claim("c1", "Lyon"), _claim("c2", "Paris")
    graph = _graph(
        passages,
        ((lyon,), (paris,)),
        supports=(("p1", "c1"), ("p2", "c2"), ("p3", "c2"), ("p4", "c2")),
    )
    lyon_cluster, paris_cluster = graph.clusters
    assert (lyon_cluster.independent_support, paris_cluster.independent_support) == (1, 3)

    own = graph.own_cluster("p1")
    assert own is not None
    assert own.cluster_id == "c1", "own cluster is the one p1 supports, not the pool leader"
    assert graph.takes_position("p1") is True, "p1 supports exactly one cluster, so invariant 1 \
does not fire and invariant 2 is doing the work here"


def test_a_distractor_with_only_unknown_edges_is_never_droppable() -> None:
    """Condition 1 gives abstention first-class status (design §2.3): a passage that says
    nothing about any claim holds no position, joins no cluster and casts no vote. UNKNOWN is a
    predicted class, so this is the model declining to commit, not a low score."""
    passages = (_passage("p1", "parent-a"), _passage("noise", "parent-z"))
    paris, lyon = _claim("c1", "Paris"), _claim("c2", "Lyon")
    graph = _graph(
        passages,
        ((paris,), (lyon,)),
        supports=(("p1", "c1"),),
        unknown=(("noise", "c1"), ("noise", "c2")),
    )
    assert graph.takes_position("noise") is False
    assert graph.own_cluster("noise") is None
    assert graph.supported_clusters("noise") == ()
    assert all("noise" not in cluster.member_ids for cluster in graph.clusters)
    assert graph.clusters[0].independent_support == 1
    assert len(graph.edges) == 3, "abstention must stay measurable (M0 §3.6, G-AB)"


def test_distinct_cluster_mode_needs_no_refutes_edge_and_no_entailment_check() -> None:
    """Primary conflict_mode (M0 §2.2): a competing cluster is any OTHER claim cluster. Clustering
    already merged by mutual entailment, so distinct clusters are non-entailing by construction —
    no extra check, and REFUTES edges are recorded but not consumed."""
    passages = (_passage("p1", "parent-a"), _passage("p2", "parent-b"))
    paris, lyon = _claim("c1", "Paris"), _claim("c2", "Lyon")
    graph = _graph(
        passages,
        ((paris,), (lyon,)),
        supports=(("p1", "c1"), ("p2", "c2")),
    )
    competitors = graph.competing_clusters("p1")
    assert tuple(cluster.cluster_id for cluster in competitors) == ("c2",)
    assert tuple(cluster.cluster_id for cluster in graph.competing_clusters("p2")) == ("c1",)


def test_refutes_edge_mode_demands_a_refutes_edge_from_the_competing_cluster() -> None:
    """Ablation arm (M0 §2.2): a competing cluster additionally needs a member holding a REFUTES
    edge toward the candidate's claim. Stricter than the primary mode, so the same pool that
    fires under `distinct_cluster` must go quiet here without such an edge."""
    passages = (_passage("p1", "parent-a"), _passage("p2", "parent-b"))
    paris, lyon = _claim("c1", "Paris"), _claim("c2", "Lyon")
    supports = (("p1", "c1"), ("p2", "c2"))

    silent = _graph(passages, ((paris,), (lyon,)), supports=supports, conflict_mode="refutes_edge")
    assert silent.competing_clusters("p1") == ()

    contested = _graph(
        passages,
        ((paris,), (lyon,)),
        supports=supports,
        refutes=(("p2", "c1"),),
        conflict_mode="refutes_edge",
    )
    assert tuple(cluster.cluster_id for cluster in contested.competing_clusters("p1")) == ("c2",)


def test_refutes_edge_mode_ignores_a_refutation_from_a_non_member() -> None:
    """The REFUTES edge has to come from a MEMBER of the competing cluster. A bystander that
    supports nothing can object to c's claim without lending its objection to any cluster."""
    passages = (
        _passage("p1", "parent-a"),
        _passage("p2", "parent-b"),
        _passage("bystander", "parent-z"),
    )
    paris, lyon = _claim("c1", "Paris"), _claim("c2", "Lyon")
    graph = _graph(
        passages,
        ((paris,), (lyon,)),
        supports=(("p1", "c1"), ("p2", "c2")),
        refutes=(("bystander", "c1"),),
        conflict_mode="refutes_edge",
    )
    assert graph.competing_clusters("p1") == ()


def test_build_predicts_every_passage_claim_pair_in_one_batch() -> None:
    """Data flow of design §2.4: passage x claim is the bipartite edge set, so the predictor sees
    |P| x |C| pairs of (passage text, claim hypothesis) — one batched call, since cost is
    reported per query and a per-pair loop would misstate it."""
    passages = (_passage("p1", "parent-a"), _passage("p2", "parent-a"))
    paris, lyon = _claim("c1", "Paris"), _claim("c2", "Lyon")
    predictor = _FakePredictor(
        {
            (passages[0].text, paris.hypothesis): RelationLabel.SUPPORTS,
            (passages[1].text, paris.hypothesis): RelationLabel.SUPPORTS,
            (passages[1].text, lyon.hypothesis): RelationLabel.REFUTES,
        }
    )
    graph = QueryLocalGraph.build(
        passages=passages,
        claim_clusters=((paris,), (lyon,)),
        predictor=predictor,
    )
    assert len(predictor.calls) == 1
    assert predictor.calls[0] == (
        (passages[0].text, paris.hypothesis),
        (passages[0].text, lyon.hypothesis),
        (passages[1].text, paris.hypothesis),
        (passages[1].text, lyon.hypothesis),
    )
    assert len(graph.edges) == 4
    assert graph.clusters[0].independent_support == 1, "both passages share one parent"
    assert graph.clusters[1].independent_support == 0
    assert graph.edges[0].passage_id == "p1"
    assert graph.edges[0].claim_id == "c1"
    assert graph.edges[0].model_version == "fake-1"
    assert graph.edges[0].premise_hash == text_hash(passages[0].text)


def test_an_unfrozen_conflict_mode_is_rejected_not_silently_treated_as_one() -> None:
    """M0 §2.2 freezes exactly two values. A third must fail loudly: a mode that quietly behaves
    like one of them would run a whole arm under the wrong conflict rule with no metric able to
    notice."""
    with pytest.raises(ValueError, match="conflict_mode"):
        _graph((), (), conflict_mode=cast(ConflictMode, "entailment_check"))


def test_a_vote_from_an_unknown_passage_is_an_error_not_a_guessed_parent() -> None:
    """A voter with no PassageNode has no `source_parent_id`. Treating it as its own parent would
    hand it an independent vote — the exact inflation `independent_support` exists to stop."""
    with pytest.raises(ValueError, match="unknown passage"):
        _graph(
            (_passage("p1", "parent-a"),),
            ((_claim("c1", "Paris"),),),
            supports=(("ghost", "c1"),),
        )


def test_an_empty_claim_cluster_is_rejected() -> None:
    """A cluster with no claims has no representative and can hold no edge; admitting one would
    add a phantom competitor to condition 2."""
    with pytest.raises(ValueError, match="empty claim cluster"):
        _graph((_passage("p1", "parent-a"),), ((),))


def test_a_parent_collapse_is_measurable_instead_of_silent() -> None:
    """A caller that passes `document_id` into `source_parent_id` silently reverts to the
    pre-§3.4 tally, where every 100-word passage of one article votes separately. The graph
    cannot tell a wrong string from a right one, so it reports the SHAPE instead: every passage
    being its own parent is the signature of the reversion, and M0 §3.7 (G-PQ) is where it has
    to land so a run cannot look clean while counting the old way.
    """
    clusters = ((_claim("c1", "Paris"),),)
    supports = (("p1", "c1"), ("p2", "c1"), ("p3", "c1"))

    reverted = _graph(
        (_passage("p1", "p1"), _passage("p2", "p2"), _passage("p3", "p3")),
        clusters,
        supports=supports,
    )
    assert reverted.clusters[0].independent_support == 3, "the old, inflated tally"
    assert reverted.parent_resolution.self_parented == 3
    assert reverted.parent_resolution.n_parents == 3

    resolved = _graph(
        (_passage("p1", "parent-a"), _passage("p2", "parent-a"), _passage("p3", "parent-b")),
        clusters,
        supports=supports,
    )
    assert resolved.clusters[0].independent_support == 2, "two passages share one article"
    assert resolved.parent_resolution.self_parented == 0
    assert resolved.parent_resolution.n_passages == 3
