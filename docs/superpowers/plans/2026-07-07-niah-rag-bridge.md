# NIAH → RAG bridge — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure end-to-end RAG answer quality (cover-EM) over the NIAH counterfactual-distractor haystack across the retrieval stack (dense → q2d → q2d_corroborate), by bridging a `NiahTask` into `eval.run_rag`.

**Architecture:** Expose gold answers on `NiahTask` (populated in `load_niah_task` from the `BenchmarkData` it already loads), then a thin `eval/run_niah_rag.py` that turns a `NiahTask` into a `BenchmarkData` and drives `run_rag.run()` once per retriever with a single shared `LLMClient`. No changes to `run_rag`. Spec: `docs/superpowers/specs/2026-07-07-niah-rag-bridge-design.md`.

**Tech Stack:** Python stdlib (argparse/pathlib), pytest. Reuses `load_niah_task`, `BenchmarkData`, `run_rag.run`, the `run_benchmark` retriever builders, `eval.significance`. No new dependencies.

---

## File structure

- Modify: `src/niah/types.py` — add `NiahTask.answers`.
- Modify: `eval/build_niah_task.py` — `load_niah_task` populates `answers`.
- Modify: `tests/test_build_niah_task.py` — one new test.
- Create: `eval/run_niah_rag.py` — the bridge (pure units + thin `main`).
- Create: `tests/test_run_niah_rag.py` — bridge tests.
- Create: `scripts/run_niah_rag.slurm` — the HPC job.

Full suite is currently **473 passed + 1 xfailed**. Run the new file's tests with `python -m pytest tests/test_run_niah_rag.py -v`.

**Interfaces (verified against current code):** `NiahTask(corpus, queries, qrels, examples=[])` (`src/niah/types.py:45`); `BenchmarkData(corpus, queries, qrels, answers=None)` (`eval/benchmarks/loader.py:27`); `RAGEvalConfig(dataset, retriever, top_k, ..., results_path, append, per_query_out, predictions_out, pipeline)` and `run(config, data=None, retriever=None, llm=None, judge=None)` (`eval/run_rag.py`); `load_niah_task(path, *, loader=None, max_docs=None)` (`eval/build_niah_task.py:257`).

---

### Task 1: Expose `NiahTask.answers`

**Files:**
- Modify: `src/niah/types.py`
- Modify: `eval/build_niah_task.py` (`load_niah_task`)
- Modify: `tests/test_build_niah_task.py`

- [ ] **Step 1: Write the failing test**

In `tests/test_build_niah_task.py`, append this test (it mirrors the existing `test_load_niah_task_reconstructs_from_recipe`, adding answers to the fake loader):

```python
def test_load_niah_task_carries_gold_answers(tmp_path) -> None:
    out = tmp_path / "task.json"
    write_task_json(_sample_task(), out, recipe=_RECIPE)

    def fake_loader(name, split="test", max_queries=None, max_docs=None):
        return BenchmarkData(
            corpus={"d1": "Linda Davis won the 1994 award.", "bg1": "unrelated hay"},
            queries={"q1": "who won the 1994 award?"},
            qrels={"q1": {"d1": 1}},
            answers={"q1": ["Linda Davis"]},
        )

    task = load_niah_task(out, loader=fake_loader)
    assert task.answers == {"q1": ["Linda Davis"]}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_build_niah_task.py::test_load_niah_task_carries_gold_answers -v`
Expected: FAIL — `AttributeError: 'NiahTask' object has no attribute 'answers'`.

- [ ] **Step 3: Write minimal implementation**

In `src/niah/types.py`, add an `answers` field to `NiahTask` (after `examples`):

```python
@dataclass
class NiahTask:
    """A built task: a runnable benchmark + distractor provenance."""

    corpus: Dict[str, str]
    queries: Dict[str, str]
    qrels: Dict[str, Dict[str, int]]
    examples: List[NiahExample] = field(default_factory=list)
    # Gold free-text answers ``{query_id: [answer, ...]}`` from the source QA set,
    # carried through for end-to-end RAG scoring (eval/run_niah_rag.py). Empty for
    # retrieval-only sources. See 2026-07-07-niah-rag-bridge-design.md.
    answers: Dict[str, List[str]] = field(default_factory=dict)
```

In `eval/build_niah_task.py`, in `load_niah_task`, add `answers=data.answers or {}` to the `NiahTask(...)` construction (the `return NiahTask(...)` near the end of the function):

```python
    return NiahTask(
        corpus=inject(data.corpus, distractors),
        queries=data.queries,
        qrels=data.qrels,
        examples=examples,
        answers=data.answers or {},
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_build_niah_task.py -v`
Expected: all pass (the new test + the existing `load_niah_task` tests, which use a loader without answers → `answers` defaults to `{}`, unaffected).

- [ ] **Step 5: Commit**

```bash
git add src/niah/types.py eval/build_niah_task.py tests/test_build_niah_task.py
git commit -m "feat(niah): carry gold answers on NiahTask (load_niah_task) for end-to-end RAG scoring"
```

---

### Task 2: `niah_to_benchmark_data` + CLI args

**Files:**
- Create: `eval/run_niah_rag.py`
- Create: `tests/test_run_niah_rag.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_run_niah_rag.py`:

```python
"""Tests for the NIAH -> run_rag bridge (pure/injectable parts)."""
import pytest

from eval.run_niah_rag import _parse_args, niah_to_benchmark_data
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_run_niah_rag.py -v`
Expected: FAIL (ModuleNotFoundError: `eval.run_niah_rag`).

- [ ] **Step 3: Write minimal implementation**

Create `eval/run_niah_rag.py`:

```python
"""Bridge a NIAH task into eval.run_rag: measure RAG answer quality (cover-EM) over the
counterfactual-distractor haystack, across retrievers (dense -> q2d -> q2d_corroborate).

A ``NiahTask`` is an ordinary (corpus, queries, qrels) benchmark plus gold answers, so
it becomes a ``BenchmarkData`` that ``run_rag.run`` scores unchanged. One shared
``LLMClient`` serves the q2d transform, the corroboration reranking, and generation (the
OOM-safe pattern from run_niah). Design: docs/superpowers/specs/2026-07-07-niah-rag-bridge-design.md.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, List, Optional

from eval.benchmarks.loader import BenchmarkData
from src.niah.types import NiahTask


def niah_to_benchmark_data(task: NiahTask) -> BenchmarkData:
    """Wrap a NIAH task (corpus WITH counterfactual distractors + queries + qrels + gold
    answers) as a ``BenchmarkData`` for RAG scoring. Raises if the task has no answers —
    cover-EM needs gold to score against."""
    if not task.answers:
        raise ValueError(
            "NIAH task carries no gold answers -- cover-EM needs them. Load a recipe "
            "whose source dataset provides answers (e.g. nq)."
        )
    return BenchmarkData(
        corpus=task.corpus, queries=task.queries, qrels=task.qrels, answers=task.answers
    )


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m eval.run_niah_rag",
        description="RAG answer quality (cover-EM) over the NIAH counterfactual haystack, "
        "across retrievers. Bridges a NIAH task into eval.run_rag.",
    )
    p.add_argument("--task", type=Path, required=True,
                   help="NIAH recipe JSON (loaded via load_niah_task).")
    p.add_argument("--retrievers", nargs="+",
                   default=["granite_dense", "q2d_granite", "q2d_corroborate"],
                   help="Retriever arms to compare (default: %(default)s).")
    p.add_argument("--top-k", type=int, default=4, dest="top_k",
                   help="RAG context size -- chunks passed to the generator (default: %(default)s).")
    p.add_argument("--max-docs", type=int, default=None, dest="max_docs",
                   help="Override the recipe corpus cap (scale sweep).")
    p.add_argument("--out", type=Path, default=Path("results/rag_niah_agg.csv"))
    p.add_argument("--per-query-out", type=Path, default=Path("results/rag_niah_cmp"),
                   dest="per_query_out",
                   help="Per-metric per-query CSV prefix (one column per retriever).")
    p.add_argument("--predictions-out", type=Path, default=None, dest="predictions_out")
    return p.parse_args(argv)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_run_niah_rag.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add eval/run_niah_rag.py tests/test_run_niah_rag.py
git commit -m "feat(eval): run_niah_rag - niah_to_benchmark_data + CLI (NIAH->RAG bridge)"
```

---

### Task 3: `run_niah_rag` loop + `main`

**Files:**
- Modify: `eval/run_niah_rag.py`
- Modify: `tests/test_run_niah_rag.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_run_niah_rag.py` (extend the import to `from eval.run_niah_rag import _parse_args, niah_to_benchmark_data, run_niah_rag`):

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_run_niah_rag.py::test_run_niah_rag_drives_run_per_retriever_with_shared_data_and_llm -v`
Expected: ImportError (`run_niah_rag`).

- [ ] **Step 3: Write minimal implementation**

In `eval/run_niah_rag.py`, append `run_niah_rag` and `main` (the `run_rag` imports are lazy so the pure units above stay light to import):

```python
def run_niah_rag(
    task: NiahTask,
    retrievers: List[str],
    llm,
    top_k: int,
    out: Path,
    per_query_out: Path,
    predictions_out: Optional[Path] = None,
    run_fn: Optional[Callable] = None,
) -> None:
    """Drive ``run_rag.run`` once per retriever over the SAME NIAH ``BenchmarkData`` and
    the SAME ``llm`` (shared -> no duplicate model loads). ``append`` accumulates all arms
    into one per-metric per-query CSV (column per retriever) for paired significance.
    ``run_fn`` is injected in tests; production uses ``eval.run_rag.run``."""
    from eval.run_rag import RAGEvalConfig

    if run_fn is None:
        from eval.run_rag import run as run_fn

    data = niah_to_benchmark_data(task)
    for name in retrievers:
        config = RAGEvalConfig(
            dataset="niah",
            retriever=name,
            top_k=top_k,
            results_path=out,
            append=True,
            per_query_out=per_query_out,
            predictions_out=predictions_out,
        )
        run_fn(config, data=data, llm=llm)


def main(argv: Optional[List[str]] = None) -> None:
    from eval.build_niah_task import load_niah_task
    from src.llm_client import LLMClient

    args = _parse_args(argv)
    task = load_niah_task(args.task, max_docs=args.max_docs)
    llm = LLMClient()  # one client: q2d transform + corroboration reranking + generation
    run_niah_rag(
        task, args.retrievers, llm, args.top_k, args.out,
        args.per_query_out, args.predictions_out,
    )
    cover = f"{args.per_query_out}_answer_cover.csv"
    print(f"wrote {cover} -> significance:")
    print(f"  python -m eval.significance --per-query-csv {cover} --reference granite_dense")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the file's tests**

Run: `python -m pytest tests/test_run_niah_rag.py -v`
Expected: 4 passed.

- [ ] **Step 5: Run the FULL suite**

Run: `python -m pytest -q`
Expected: all green, no NEW failures (baseline 473 passed + 1 xfailed → now 478 passed + 1 xfailed).

- [ ] **Step 6: Commit**

```bash
git add eval/run_niah_rag.py tests/test_run_niah_rag.py
git commit -m "feat(eval): run_niah_rag loop + main - shared-LLM RAG over the NIAH haystack per retriever"
```

---

### Task 4: HPC job script

**Files:**
- Create: `scripts/run_niah_rag.slurm`

- [ ] **Step 1: Write the slurm script**

Create `scripts/run_niah_rag.slurm`:

```bash
#!/bin/bash
# End-to-end system test: does better retrieval survive to a correct, gold-matched answer
# despite counterfactual distractors? RAG cover-EM / F1 over the NIAH haystack, across the
# retrieval stack (granite_dense -> q2d_granite -> q2d_corroborate). Same generator/prompt/
# top_k; only the retriever varies. One shared 8B client (transform + rerank + generate).
#
# Reuses the run_rag prefetch (8B generator + Granite dense embedder + dpr-w100 NQ). Submit
# from the project root AFTER `git pull`:
#   mkdir -p logs results && sbatch scripts/run_niah_rag.slurm
#   # a smaller/faster first read (q2d_corroborate is ~20 extractions/query):
#   sbatch scripts/run_niah_rag.slurm nq300 results/niah_nq300_frozen.json
#SBATCH --job-name=granite-niah-rag
#SBATCH --account=coms039904
#SBATCH --partition=gpu
#SBATCH --qos=normal
# 8B bf16 + q2d transform + corroboration reranking. The sole rtx_3090 (bp1-gpu030) is
# frequently drained -> target the A100 40GB MIG slice (bp1-gpu035).
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
TAG="${1:-nq300}"
TASK="${2:-results/niah_nq300_frozen.json}"

module load languages/python/3.12.3
export HF_HOME=/user/work/$USER/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export IR_DATASETS_HOME=/user/work/$USER/ir_datasets PYTHONUNBUFFERED=1
export GRANITE_MODEL_ID=ibm-granite/granite-4.1-8b   # instruct (no -base) for RAG
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /user/work/$USER/venv/bin/activate
cd /user/work/$USER/IBM_Granite_Project

python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| torch', torch.__version__)"

python -m eval.run_niah_rag \
    --task "$TASK" \
    --retrievers granite_dense q2d_granite q2d_corroborate \
    --top-k 4 \
    --out "results/rag_niah_${TAG}_agg.csv" \
    --per-query-out "results/rag_niah_${TAG}_cmp" \
    --predictions-out "results/rag_niah_${TAG}_pred"

echo "===== aggregate (cover-EM / F1 per retriever) ====="
cat "results/rag_niah_${TAG}_agg.csv"
echo "===== significance: cover-EM vs granite_dense ====="
python -m eval.significance --per-query-csv "results/rag_niah_${TAG}_cmp_answer_cover.csv" --reference granite_dense
echo "===== significance: F1 vs granite_dense ====="
python -m eval.significance --per-query-csv "results/rag_niah_${TAG}_cmp_answer_f1.csv" --reference granite_dense
echo "===== DONE (tag=${TAG}) ====="
```

- [ ] **Step 2: Verify the full suite is still green** (the slurm adds no code paths, but confirm nothing else drifted)

Run: `python -m pytest -q`
Expected: 478 passed, 1 xfailed.

- [ ] **Step 3: Commit**

```bash
git add scripts/run_niah_rag.slurm
git commit -m "ops: run_niah_rag.slurm - 3-arm RAG cover-EM over the NIAH haystack (A100 40GB slice)"
```

---

## After implementation (user-run, on HPC)

```bash
git pull
mkdir -p logs results && sbatch scripts/run_niah_rag.slurm
# after it lands, the significance vs granite_dense (cover-EM) is the end-to-end verdict:
#   does q2d / corroboration retrieval -> better cited answers under counterfactual distractors?
```

Paste the aggregate + significance into `docs/results-summary.md` as the system-leg / end-to-end result (a new finding: retrieval gains → answer gains, or an honest null if they wash out at the generation stage).
