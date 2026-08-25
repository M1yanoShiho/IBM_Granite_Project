from types import SimpleNamespace

from evidence_rag.evaluation.experiment05_retrieval import build_rapid_retrieval_trace
from evidence_rag.evaluation.experiment05_runtime import ALL_ARMS, prepare_query
from evidence_rag.selector.threshold_only import NliThresholdOnlySelector


class _Encoder:
    def encode(self, texts, **_kwargs):  # type: ignore[no-untyped-def]
        return [[float(len(text)), 1.0] for text in texts]


class _Reranker:
    def score(self, _query, passages):  # type: ignore[no-untyped-def]
        return [float(len(item)) for item in passages]


class _Provence:
    def prune(self, *, question, title, text):  # type: ignore[no-untyped-def]
        return f"{title} {text}".strip()


class _SelectorModel:
    def __call__(self, **kwargs):  # type: ignore[no-untyped-def]
        size = len(kwargs["question"])
        return SimpleNamespace(protect_scores=[0.5] * size, harm_scores=[0.5] * size)


def test_prepares_exact_ten_arms_with_shared_ablation_inputs() -> None:
    candidates = [
        {
            "id": f"e{rank}",
            "contents": f"Title {rank}\nPassage text {rank}",
            "title": f"Title {rank}",
            "source_kind": "fixture",
            "source_id": str(rank),
            "start_unit": rank,
            "end_unit": rank,
            "bm25_rank": rank,
            "bm25_score": float(50 - rank),
        }
        for rank in range(1, 41)
    ]
    trace = build_rapid_retrieval_trace(
        dataset="kilt-nq",
        query_id="q1",
        question="What is the answer?",
        bm25_candidates=candidates,
        encoder=_Encoder(),
    )
    prepared = prepare_query(
        {
            "schema_version": "experiment05.runtime.v1",
            "dataset": "kilt-nq",
            "query_id": "q1",
            "question": "What is the answer?",
            "corpus_snapshot_id": "corpus",
            "bm25_index_id": "bm25",
            "dense_index_id": "rapid",
        },
        trace,
        reranker=_Reranker(),
        provence=_Provence(),
        selector=NliThresholdOnlySelector(model=_SelectorModel()),
    )

    assert set(prepared.arms) == set(ALL_ARMS)
    assert prepared.arms["ours_seed13"] == prepared.arms["ours_seed42"]
    assert prepared.arms["ours_seed13"] == prepared.arms["ablation_direct_generator"]
    assert prepared.arms["hybrid_rag"] == prepared.arms["ablation_no_selector"]
    assert prepared.selector_traces["hybrid"].dropped_evidence_ids == ()
