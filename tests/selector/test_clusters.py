from evidence_rag.contracts.models import EvidenceCandidate
from evidence_rag.selector.clusters import build_clusters


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
