"""
unit test for benchmark loader
"""

from __future__ import annotations
from collections import namedtuple
from unittest.mock import MagicMock, patch
from eval.benchmarks.loader import load_benchmark

# Mock factories
MockDoc = namedtuple("MockDoc", ["doc_id", "text"])
MockQuery = namedtuple("MockQuery", ["query_id", "text"])
MockQrel = namedtuple("MockQrel", ["query_id", "doc_id", "relevance"])
MockAnswer = namedtuple("MockAnswer", ["query_id", "text", "answers"])

def _make_mock_dataset(docs=None, queries=None, qrels=None):
    mock = MagicMock()
    mock.docs_iter.return_value = docs or []
    mock.queries_iter.return_value = queries or []
    mock.qrels_iter.return_value = qrels or []
    return mock


# NQ gold answers
class TestNQWithAnswers:

    @patch("eval.benchmarks.loader.ir_datasets")
    def test_nq_loads_from_dpr_single_source(self, mock_ir):
        dpr_nq = _make_mock_dataset(
            docs=[MockDoc("d1", "Paris is the capital of France.")],
            queries=[MockAnswer("q1", "capital of france", ("Paris",))],
            qrels=[MockQrel("q1", "d1", 1)],
        )

        def side_effect(dataset_id):
            if dataset_id == "dpr-w100/natural-questions/dev":
                return dpr_nq
            raise KeyError(dataset_id)

        mock_ir.load.side_effect = side_effect

        result = load_benchmark("nq")

        assert result.answers == {"q1": ["Paris"]}
        assert result.corpus == {"d1": "Paris is the capital of France."}
        assert result.qrels == {"q1": {"d1": 1}}
        mock_ir.load.assert_called_once_with("dpr-w100/natural-questions/dev")

    @patch("eval.benchmarks.loader.ir_datasets")
    def test_nq_keeps_all_gold_answers(self, mock_ir):
        # NQ questions have several acceptable aliases; all are kept so the
        # answer-correctness metric can score against the best match.
        dpr_nq = _make_mock_dataset(
            docs=[MockDoc("d1", "...")],
            queries=[MockAnswer("q1", "q?", ("NYC", "New York City"))],
            qrels=[MockQrel("q1", "d1", 1)],
        )

        def side_effect(dataset_id):
            if dataset_id == "dpr-w100/natural-questions/dev":
                return dpr_nq
            raise KeyError(dataset_id)

        mock_ir.load.side_effect = side_effect

        result = load_benchmark("nq")
        assert result.answers == {"q1": ["NYC", "New York City"]}

    @patch("eval.benchmarks.loader.ir_datasets")
    def test_non_nq_dataset_has_no_answers(self, mock_ir):
        beir = _make_mock_dataset(
            docs=[MockDoc("d1", "Doc.")],
            queries=[MockQuery("q0", "question?")],
            qrels=[MockQrel("q0", "d1", 1)],
        )

        def side_effect(dataset_id):
            if dataset_id == "beir/scifact/test":
                return beir
            raise KeyError(dataset_id)

        mock_ir.load.side_effect = side_effect

        result = load_benchmark("scifact")

        assert result.answers is None


# Subsampling (so a 21M-passage set like NQ can be run on a small subset first)
class TestSubsampling:

    def _dataset(self):
        return _make_mock_dataset(
            docs=[MockDoc(f"d{i}", f"text {i}") for i in range(1, 6)],
            queries=[
                MockAnswer("q1", "Q1", ("a1",)),
                MockAnswer("q2", "Q2", ("a2",)),
                MockAnswer("q3", "Q3", ("a3",)),
            ],
            qrels=[
                MockQrel("q1", "d1", 1),
                MockQrel("q2", "d2", 1),
                MockQrel("q3", "d3", 1),
            ],
        )

    @patch("eval.benchmarks.loader.ir_datasets")
    def test_no_limits_loads_everything(self, mock_ir):
        mock_ir.load.return_value = self._dataset()
        result = load_benchmark("nq")
        assert set(result.queries) == {"q1", "q2", "q3"}
        assert len(result.corpus) == 5

    @patch("eval.benchmarks.loader.ir_datasets")
    def test_max_queries_limits_queries_and_their_qrels(self, mock_ir):
        mock_ir.load.return_value = self._dataset()
        result = load_benchmark("nq", max_queries=2)
        assert set(result.queries) == {"q1", "q2"}
        assert set(result.qrels) == {"q1", "q2"}
        assert set(result.answers) == {"q1", "q2"}

    @patch("eval.benchmarks.loader.ir_datasets")
    def test_max_docs_caps_corpus_but_always_keeps_gold_docs(self, mock_ir):
        mock_ir.load.return_value = self._dataset()
        # keep 2 queries (gold docs d1, d2) + at most 1 distractor doc
        result = load_benchmark("nq", max_queries=2, max_docs=1)
        # gold docs for the kept queries must be present even past the cap
        assert {"d1", "d2"} <= set(result.corpus)
        # exactly one non-gold distractor kept
        assert len(set(result.corpus) - {"d1", "d2"}) == 1
