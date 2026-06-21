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
MockNqQrel = namedtuple(
    "MockNqQrel",
    ["query_id", "doc_id", "relevance", "short_answers", "yes_no_answer"],
)


def _make_mock_dataset(docs=None, queries=None, qrels=None):
    mock = MagicMock()
    mock.docs_iter.return_value = docs or []
    mock.queries_iter.return_value = queries or []
    mock.qrels_iter.return_value = qrels or []
    return mock


# NQ gold answers
class TestNQWithAnswers:

    @patch("eval.benchmarks.loader.ir_datasets")
    def test_nq_returns_answers_joined_by_raw_query_id(self, mock_ir):
        beir_nq = _make_mock_dataset(
            docs=[MockDoc("d1", "Paris is the capital of France.")],
            queries=[MockQuery("nq_orig_42", "what is the capital of france")],
            qrels=[MockQrel("nq_orig_42", "d1", 1)],
        )
        raw_nq = _make_mock_dataset(
            queries=[
                MockQuery("nq_orig_42", "what is the capital of france"),
            ],
            qrels=[
                MockNqQrel("nq_orig_42", "raw_doc_1", 1, ["Paris"], "NONE"),
            ],
        )

        def side_effect(dataset_id):
            if dataset_id == "beir/nq":
                return beir_nq
            if dataset_id == "natural-questions/dev":
                return raw_nq
            raise KeyError(dataset_id)

        mock_ir.load.side_effect = side_effect

        result = load_benchmark("nq")

        assert result.answers is not None
        assert result.answers == {"nq_orig_42": "Paris"}

    @patch("eval.benchmarks.loader.ir_datasets")
    def test_nq_answers_is_none_when_dpr_fails(self, mock_ir):
        beir_nq = _make_mock_dataset(
            docs=[MockDoc("d1", "Doc.")],
            queries=[MockQuery("q0", "question?")],
            qrels=[],
        )

        def side_effect(dataset_id):
            if dataset_id == "beir/nq/test":
                raise KeyError(dataset_id)
            if dataset_id == "beir/nq":
                return beir_nq
            raise RuntimeError("NQ download failed")

        mock_ir.load.side_effect = side_effect

        result = load_benchmark("nq")

        assert result.answers is None

    @patch("eval.benchmarks.loader.ir_datasets")
    def test_nq_returns_answers_for_beir_test_ids_from_raw_dev_order(self, mock_ir):
        beir_nq = _make_mock_dataset(
            docs=[MockDoc("d1", "Doc.")],
            queries=[MockQuery("test0", "what is the capital of france")],
            qrels=[],
        )
        raw_nq = _make_mock_dataset(
            queries=[
                MockQuery("nq_x", "what is the capital of france"),
            ],
            qrels=[
                MockNqQrel("nq_x", "raw_doc_1", 1, ["Paris"], "NONE"),
            ],
        )

        def side_effect(dataset_id):
            if "beir" in dataset_id:
                return beir_nq
            if dataset_id == "natural-questions/dev":
                return raw_nq
            raise KeyError(dataset_id)

        mock_ir.load.side_effect = side_effect

        result = load_benchmark("nq")
        assert result.answers == {"test0": "Paris"}

    @patch("eval.benchmarks.loader.ir_datasets")
    def test_nq_unmatched_queries_excluded_from_answers(self, mock_ir):
        beir_nq = _make_mock_dataset(
            docs=[MockDoc("d1", "Doc one."), MockDoc("d2", "Doc two.")],
            queries=[
                MockQuery("q_matched", "what is the capital"),
                MockQuery("q_unmatched", "some completely different question"),
            ],
            qrels=[],
        )
        raw_nq = _make_mock_dataset(
            queries=[
                MockQuery("q_matched", "what is the capital"),
            ],
            qrels=[
                MockNqQrel("q_matched", "raw_doc_1", 1, ["Paris"], "NONE"),
            ],
        )

        def side_effect(dataset_id):
            if "beir" in dataset_id:
                return beir_nq
            if dataset_id == "natural-questions/dev":
                return raw_nq
            raise KeyError(dataset_id)

        mock_ir.load.side_effect = side_effect

        result = load_benchmark("nq")

        assert result.answers == {"q_matched": "Paris"}
        assert "q_unmatched" not in result.answers
