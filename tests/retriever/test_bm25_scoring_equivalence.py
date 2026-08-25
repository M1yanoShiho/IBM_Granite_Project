"""Pins the R5 constant-factor rewrite to bit-for-bit identical output.

`BM25Retriever.retrieve` used to rebuild `Counter(tokens)` for every chunk on every
query, re-analyse the query once per chunk, and recompute each term's IDF and each
chunk's length normalisation inside the innermost loop. All four are functions of the
corpus and the query alone, so they were hoisted (see `_set_chunks`). That is a pure
constant-factor change: it must not move a single score, or the recorded SciFact/NQ/2Wiki
numbers stop being reproducible.

`reference_scores` below is the pre-rewrite algorithm, transcribed verbatim. The tests
assert equality with `==`, not `pytest.approx` — the rewrite keeps the arithmetic grouped
as it was, so any drift at all means the grouping changed and the claim is false.
"""

import math
import random
from collections import Counter

from evidence_rag.contracts.models import Document, Query
from evidence_rag.infrastructure.corpus import CorpusBuilder, WordChunker
from evidence_rag.retriever.bm25 import BM25Retriever
from evidence_rag.retriever.strong_bm25 import StrongBM25Retriever


def reference_scores(retriever: BM25Retriever, query: Query) -> list[tuple[str, float]]:
    """The original scoring loop: a full scan, rebuilding everything per chunk.

    Deliberately self-contained — it re-analyses every chunk from ``retriever.chunks`` and
    recomputes document frequencies and lengths itself, touching none of the retriever's
    cached structures. It was not always so: it used to read ``retriever.tokens`` and
    ``retriever.term_frequencies``, which the inverted index replaced, and a reference
    implementation that shares structure with the thing it checks can only detect a subset
    of the ways that thing can be wrong.
    """

    analyzed = [retriever.analyzer(chunk.text) for chunk in retriever.chunks]
    total = len(analyzed)
    document_frequency = Counter(term for tokens in analyzed for term in set(tokens))
    average_length = sum(len(tokens) for tokens in analyzed) / total if total else 0.0

    out: list[tuple[str, float]] = []
    for chunk, tokens in zip(retriever.chunks, analyzed, strict=True):
        counts = Counter(tokens)
        score = 0.0
        for term in retriever.analyzer(query.text):
            frequency = counts[term]
            if frequency == 0:
                continue
            df = document_frequency[term]
            inverse_document_frequency = math.log(1.0 + (total - df + 0.5) / (df + 0.5))
            length_ratio = len(tokens) / average_length if average_length else 0.0
            denominator = frequency + retriever.k1 * (1.0 - retriever.b + retriever.b * length_ratio)
            score += inverse_document_frequency * (frequency * (retriever.k1 + 1.0) / denominator)
        if score > 0.0:
            out.append((chunk.evidence_id, score))
    out.sort(key=lambda item: (-item[1], item[0]))
    return out


VOCABULARY = (
    "revenue cost margin quarter growth policy travel expense audit filing segment "
    "guidance impairment inventory receivable liability equity dividend buyback tax"
).split()


def synthetic_documents(count: int, *, seed: int) -> tuple[Document, ...]:
    rng = random.Random(seed)
    documents = []
    for index in range(count):
        # Deliberately skewed so document frequencies differ widely across terms, which
        # is what makes the IDF hoist observable if it were wrong.
        length = rng.randint(30, 120)
        words = [rng.choice(VOCABULARY[: rng.randint(3, len(VOCABULARY))]) for _ in range(length)]
        documents.append(
            Document(
                document_id=f"doc-{index}",
                text=" ".join(words),
                source_uri=f"fixture://doc-{index}",
            )
        )
    return tuple(documents)


def build(documents: tuple[Document, ...]) -> BM25Retriever:
    corpus = CorpusBuilder(WordChunker(chunk_size=20, overlap=4)).build(documents, "fixture-signature")
    return BM25Retriever.from_corpus(corpus, k1=1.5, b=0.75)


def test_scores_are_bit_for_bit_identical_to_the_pre_rewrite_loop() -> None:
    retriever = build(synthetic_documents(60, seed=11))
    rng = random.Random(99)
    for index in range(40):
        text = " ".join(rng.choice(VOCABULARY) for _ in range(rng.randint(1, 6)))
        query = Query(query_id=f"q-{index}", text=text)
        expected = reference_scores(retriever, query)
        actual = [
            (candidate.evidence_id, candidate.retrieval_score)
            for candidate in retriever.retrieve(query, top_k=len(retriever.chunks)).candidates
        ]
        assert actual == expected, f"scores diverged on {text!r}"


def test_a_repeated_query_term_still_counts_once_per_occurrence() -> None:
    # The hoist deduplicates IDF lookups but must keep iterating query terms with their
    # multiplicity: "cost cost" scores double "cost". Collapsing to a set would be a
    # silent behaviour change that no aggregate metric would obviously catch.
    retriever = build(synthetic_documents(20, seed=5))
    once = retriever.retrieve(Query(query_id="q", text="cost"), top_k=1).candidates[0]
    twice = retriever.retrieve(Query(query_id="q", text="cost cost"), top_k=1).candidates[0]
    assert twice.evidence_id == once.evidence_id
    assert twice.retrieval_score == once.retrieval_score * 2


def test_postings_are_exactly_the_transpose_of_the_per_chunk_counts() -> None:
    # The inverted index must hold the same information as the forward index it replaced,
    # in the other orientation: same terms, same chunks, same frequencies, no extras and
    # nothing dropped. A posting list missing an entry loses a chunk silently — it simply
    # never gets scored, and no metric says why.
    retriever = build(synthetic_documents(15, seed=7))
    expected: dict[str, list[tuple[int, int]]] = {}
    for index, chunk in enumerate(retriever.chunks):
        for term, frequency in Counter(retriever.analyzer(chunk.text)).items():
            expected.setdefault(term, []).append((index, frequency))

    assert set(retriever.postings) == set(expected)
    for term, entries in expected.items():
        # Corpus order, so a posting list can be walked without sorting it first.
        assert list(retriever.postings[term]) == sorted(entries)
    # And it agrees with the document frequencies computed independently of it.
    for term, entries in retriever.postings.items():
        assert retriever.document_frequency[term] == len(entries)


def test_a_chunk_is_scored_only_when_it_holds_a_query_term() -> None:
    # The whole point: the scan is over the query's postings, not over the corpus. If a
    # chunk sharing no term with the query were still reachable, the linearity R5 measured
    # would still be there and the change would have bought nothing.
    documents = (
        Document(document_id="hit", text="revenue revenue margin", source_uri="fixture://hit"),
        Document(document_id="miss", text="inventory dividend", source_uri="fixture://miss"),
    )
    retriever = build(documents)
    candidates = retriever.retrieve(Query(query_id="q", text="revenue"), top_k=10).candidates
    assert [candidate.document_id for candidate in candidates] == ["hit"]


def test_scoring_an_absent_term_does_not_grow_the_index() -> None:
    # Postings are read with .get, never indexed into, so an unseen term cannot insert an
    # empty list. A regression here would leak on every unseen term — unbounded growth
    # over a long-running index.
    retriever = build(synthetic_documents(10, seed=3))
    before = len(retriever.postings)
    retriever.retrieve(Query(query_id="q", text="wholly absent vocabulary"), top_k=5)
    assert len(retriever.postings) == before


def test_strong_bm25_inherits_the_rewrite_unchanged() -> None:
    # StrongBM25 wraps the same engine with a stopword analyzer; the reference loop reads
    # the wrapped retriever's own analyzer, so this checks the hoist is analyzer-agnostic.
    documents = synthetic_documents(40, seed=13)
    corpus = CorpusBuilder(WordChunker(chunk_size=20, overlap=4)).build(documents, "fixture-signature")
    strong = StrongBM25Retriever.from_corpus(corpus)
    query = Query(query_id="q", text="the revenue and the cost of a quarter")
    expected = reference_scores(strong._bm25, query)
    actual = [
        (candidate.evidence_id, candidate.retrieval_score)
        for candidate in strong.retrieve(query, top_k=len(strong.chunks)).candidates
    ]
    assert actual == expected


def test_empty_corpus_still_returns_nothing_rather_than_dividing_by_zero() -> None:
    corpus = CorpusBuilder(WordChunker(chunk_size=20, overlap=4)).build((), "fixture-signature")
    retriever = BM25Retriever.from_corpus(corpus)
    assert retriever.retrieve(Query(query_id="q", text="revenue"), top_k=5).candidates == ()
