"""Tests for the NIAH -> run_rag bridge (pure/injectable parts)."""
import pytest

from eval.run_niah_rag import _parse_args, niah_to_benchmark_data, run_niah_rag
from src.niah.types import NiahTask


def _task_with_answers():
    return NiahTask(
        corpus={"d1": "x", "cf1": "wrong"},
        queries={"q1": "who?"},
        qrels={"q1": {"d1": 1}},
        answers={"q1": ["Linda Davis"]},
    )


def test_niah_to_benchmark_data_maps_fields():
    task = _task_with_answers()
    bd = niah_to_benchmark_data(task)
    assert bd.corpus == task.corpus
    assert bd.queries == task.queries
    assert bd.qrels == task.qrels
    assert bd.answers == task.answers


def test_niah_to_benchmark_data_raises_without_answers():
    task = NiahTask(corpus={"d1": "x"}, queries={"q1": "q"}, qrels={"q1": {"d1": 1}})
    with pytest.raises(ValueError):
        niah_to_benchmark_data(task)


def test_parse_args_defaults():
    args = _parse_args(["--task", "results/niah_nq300_frozen.json"])
    assert args.task.name == "niah_nq300_frozen.json"
    assert args.retrievers == ["granite_dense", "q2d_granite", "q2d_corroborate"]
    assert args.top_k == 4
    assert args.max_docs is None


def test_run_niah_rag_drives_run_per_retriever_with_shared_data_and_llm(tmp_path):
    task = _task_with_answers()
    llm = object()                         # sentinel: the ONE shared client
    calls = []

    def fake_run(config, data=None, llm=None, **kwargs):
        calls.append((config.retriever, config.append, data, llm))

    run_niah_rag(
        task, ["granite_dense", "q2d_granite", "q2d_corroborate"], llm,
        top_k=4, out=tmp_path / "agg.csv", per_query_out=tmp_path / "cmp",
        run_fn=fake_run,
    )

    assert [c[0] for c in calls] == ["granite_dense", "q2d_granite", "q2d_corroborate"]
    assert all(c[1] is True for c in calls)                 # append -> one accumulated table
    assert all(c[2].answers == task.answers for c in calls)  # same NIAH data each arm
    assert all(c[3] is llm for c in calls)                   # ONE shared llm (OOM-safe)
