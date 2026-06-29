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
