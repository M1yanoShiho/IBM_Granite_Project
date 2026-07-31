from evidence_rag.contracts.models import EvidenceCandidate
from evidence_rag.selector.answer_equivalence import lenient_equivalent
from evidence_rag.selector.clusters import build_clusters, build_clusters_lenient


def evidence(evidence_id: str, document_id: str) -> EvidenceCandidate:
    return EvidenceCandidate(
        evidence_id=evidence_id,
        document_id=document_id,
        chunk_id=f"chunk-{evidence_id}",
        text=f"text {evidence_id}",
        source_uri=f"fixture://{evidence_id}",
        retrieval_score=1.0,
        retrieval_rank=int(evidence_id[-1]),
    )


def test_clusters_group_by_canonical_answer() -> None:
    window = (evidence("e1", "d1"), evidence("e2", "d2"), evidence("e3", "d3"))
    clusters = build_clusters(window, ("$1.2B", "1,200 million", "3 billion"))
    by_answer = {cluster.answer: cluster for cluster in clusters}
    assert set(by_answer) == {"1200000000", "3000000000"}
    assert by_answer["1200000000"].member_ids == ("e1", "e2")
    assert by_answer["1200000000"].independent_support == 2


def test_same_document_counts_once() -> None:
    window = (evidence("e1", "dX"), evidence("e2", "dX"), evidence("e3", "dX"))
    clusters = build_clusters(window, ("Acme", "Acme", "Acme"))
    assert clusters[0].independent_support == 1
    assert clusters[0].member_ids == ("e1", "e2", "e3")


def test_invalid_answers_are_excluded() -> None:
    window = (evidence("e1", "d1"), evidence("e2", "d2"))
    clusters = build_clusters(window, ("NONE", "it"))
    assert clusters == ()


def test_length_mismatch_raises() -> None:
    window = (evidence("e1", "d1"),)
    try:
        build_clusters(window, ("Acme", "Globex"))
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_lenient_merges_containment_fragments() -> None:
    # "Apostle Paul" and "Paul" are the same answer; exact clustering splits them, lenient merges.
    window = (evidence("e1", "d1"), evidence("e2", "d2"), evidence("e3", "d3"))
    clusters = build_clusters_lenient(window, ("Apostle Paul", "Paul", "Nixon"), lenient_equivalent)
    by_answer = {c.answer: c for c in clusters}
    assert set(by_answer) == {"apostle paul", "nixon"}
    assert by_answer["apostle paul"].member_ids == ("e1", "e2")
    assert by_answer["apostle paul"].independent_support == 2


def test_lenient_keeps_distinct_values_separate() -> None:
    # gold vs canonically-distinct counterfactual must NOT merge — the planted conflict is preserved.
    window = (evidence("e1", "d1"), evidence("e2", "d2"))
    clusters = build_clusters_lenient(window, ("Kennedy", "Nixon"), lenient_equivalent)
    assert {c.answer for c in clusters} == {"kennedy", "nixon"}


def test_lenient_no_transitive_chaining() -> None:
    # A="alpha", C="beta" are distinct; B="alpha beta" is equivalent to BOTH by containment.
    # Representative-anchored greedy must NOT let B chain A and C into one cluster.
    window = (evidence("e1", "d1"), evidence("e2", "d2"), evidence("e3", "d3"))
    clusters = build_clusters_lenient(window, ("alpha", "beta", "alpha beta"), lenient_equivalent)
    by_answer = {c.answer: c for c in clusters}
    assert set(by_answer) == {"alpha", "beta"}
    assert by_answer["alpha"].member_ids == ("e1", "e3")  # B joined A (first match), not C
    assert by_answer["beta"].member_ids == ("e2",)


def test_lenient_representative_is_first_in_order() -> None:
    window = (evidence("e1", "d1"), evidence("e2", "d2"))
    clusters = build_clusters_lenient(window, ("Paul", "Apostle Paul"), lenient_equivalent)
    assert clusters[0].answer == "paul"  # representative = first member's canonical answer
    assert clusters[0].member_ids == ("e1", "e2")


def test_lenient_same_document_counts_once() -> None:
    window = (evidence("e1", "dX"), evidence("e2", "dX"))
    clusters = build_clusters_lenient(window, ("Paul", "Apostle Paul"), lenient_equivalent)
    assert clusters[0].independent_support == 1


def test_lenient_excludes_invalid_answers() -> None:
    window = (evidence("e1", "d1"), evidence("e2", "d2"))
    assert build_clusters_lenient(window, ("NONE", "it"), lenient_equivalent) == ()


def test_same_parent_passages_are_one_vote() -> None:
    """dpr-w100 splits one article into many document_ids; they are not independent sources."""
    window = (evidence("e1", "d1"), evidence("e2", "d2"))
    clusters = build_clusters(
        window, ("Kennedy", "Kennedy"), parent_by_document={"d1": "page a", "d2": "page a"}
    )
    assert len(clusters) == 1
    assert clusters[0].independent_support == 1


def test_distinct_parents_still_count_separately() -> None:
    window = (evidence("e1", "d1"), evidence("e2", "d2"))
    clusters = build_clusters(
        window, ("Kennedy", "Kennedy"), parent_by_document={"d1": "page a", "d2": "page b"}
    )
    assert clusters[0].independent_support == 2


def test_default_is_document_unit_so_existing_behaviour_is_unchanged() -> None:
    window = (evidence("e1", "d1"), evidence("e2", "d2"))
    clusters = build_clusters(window, ("Kennedy", "Kennedy"))
    assert clusters[0].independent_support == 2


def test_unmapped_document_is_its_own_parent() -> None:
    """A missing sidecar entry must never merge two distinct sources."""
    window = (evidence("e1", "d1"), evidence("e2", "d2"))
    clusters = build_clusters(window, ("Kennedy", "Kennedy"), parent_by_document={"d1": "page a"})
    assert clusters[0].independent_support == 2


def test_lenient_clustering_honours_the_parent_unit() -> None:
    window = (evidence("e1", "d1"), evidence("e2", "d2"))
    clusters = build_clusters_lenient(
        window,
        ("Apostle Paul", "paul"),
        lenient_equivalent,
        parent_by_document={"d1": "page a", "d2": "page a"},
    )
    assert len(clusters) == 1
    assert clusters[0].independent_support == 1
