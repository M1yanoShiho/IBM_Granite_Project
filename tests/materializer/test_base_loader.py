from pathlib import Path

from evidence_rag.infrastructure.datasets import JsonlDatasetAdapter
from evidence_rag.materializer.base_loader import materialize_niah_base


class FakeDoc:
    def __init__(self, doc_id: str, text: str, title: str = "") -> None:
        self.doc_id = doc_id
        self.text = text
        self.title = title


class FakeQuery:
    def __init__(self, query_id: str, text: str, answers: tuple[str, ...]) -> None:
        self.query_id = query_id
        self.text = text
        self.answers = answers


class FakeQrel:
    def __init__(self, query_id: str, doc_id: str, relevance: int) -> None:
        self.query_id = query_id
        self.doc_id = doc_id
        self.relevance = relevance


class FakeDataset:
    def __init__(self, docs: list, queries: list, qrels: list) -> None:  # type: ignore[type-arg]
        self._docs = docs
        self._queries = queries
        self._qrels = qrels

    def docs_iter(self):  # type: ignore[no-untyped-def]
        return iter(self._docs)

    def queries_iter(self):  # type: ignore[no-untyped-def]
        return iter(self._queries)

    def qrels_iter(self):  # type: ignore[no-untyped-def]
        return iter(self._qrels)


def _dataset() -> FakeDataset:
    docs = [FakeDoc(f"d{i}", f"passage {i}", title=f"T{i}") for i in range(7)]
    queries = [
        FakeQuery("q1", "question one", ("18%",)),
        FakeQuery("q2", "question two", ("Alice", "Alice")),
    ]
    qrels = [
        FakeQrel("q1", "d0", 1),
        FakeQrel("q2", "d1", 1),
        FakeQrel("q1", "d2", 0),
    ]
    return FakeDataset(docs, queries, qrels)


def test_materialize_maps_fields_and_keeps_gold(tmp_path: Path) -> None:
    result = materialize_niah_base(_dataset(), tmp_path, corpus_size=4, seed=42)
    bundle = JsonlDatasetAdapter.load(result.manifest_path)
    ids = {d.document_id for d in bundle.documents}
    assert {"d0", "d1"} <= ids
    assert len(bundle.documents) == 4
    gold = {g.query_id: g for g in bundle.gold_cases}
    assert gold["q1"].relevant_document_ids == ("d0",)
    assert gold["q1"].reference_answers == ("18%",)
    assert gold["q2"].reference_answers == ("Alice",)
    d0 = next(d for d in bundle.documents if d.document_id == "d0")
    assert "T0" in d0.text and "passage 0" in d0.text


def test_subsample_is_deterministic(tmp_path: Path) -> None:
    a = materialize_niah_base(_dataset(), tmp_path / "a", corpus_size=4, seed=42)
    b = materialize_niah_base(_dataset(), tmp_path / "b", corpus_size=4, seed=42)
    ids_a = {d.document_id for d in JsonlDatasetAdapter.load(a.manifest_path).documents}
    ids_b = {d.document_id for d in JsonlDatasetAdapter.load(b.manifest_path).documents}
    assert ids_a == ids_b


def test_query_without_positive_qrel_is_dropped(tmp_path: Path) -> None:
    docs = [FakeDoc("d0", "p0"), FakeDoc("d1", "p1")]
    queries = [FakeQuery("q1", "q", ("x",)), FakeQuery("q2", "q", ("y",))]
    qrels = [FakeQrel("q1", "d0", 1)]
    result = materialize_niah_base(
        FakeDataset(docs, queries, qrels), tmp_path, corpus_size=2, seed=1
    )
    bundle = JsonlDatasetAdapter.load(result.manifest_path)
    assert {q.query_id for q in bundle.queries} == {"q1"}
