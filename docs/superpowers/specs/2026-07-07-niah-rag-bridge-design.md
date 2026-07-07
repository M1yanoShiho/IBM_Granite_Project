# NIAH → RAG bridge — design spec

- Date: 2026-07-07
- Owner: P6 (Weikai Mao)
- Status: design approved (brainstorm 2026-07-07) → writing-plans.
- Context: closes the loop from retrieval wins (finding 13 Q2D, finding 15 Corroboration)
  to the deliverable — a *citing* system. `run_rag` currently only consumes the QA
  benchmarks (NQ/TriviaQA); it cannot score answer quality over the NIAH
  counterfactual-distractor haystack. This adds a thin bridge so we can measure
  end-to-end RAG **cover-EM** on the NIAH task across the retrieval stack
  (dense → q2d → q2d_corroborate). Related: `2026-07-05-corroboration-reranking-design.md`.

## 1. Problem

We have proven **better retrieval** (Q2D +0.070; Corroboration +0.037 needle-found@10),
but not that it **translates into better generated, gold-matched answers** despite
counterfactual distractors (which carry *wrong* answers). The stronger end-to-end system
test: retrieve from the NIAH distractor haystack → generate → score cover-EM vs the NQ
gold answer, comparing the three retrieval arms. `run_rag` can score any injected
`BenchmarkData`, but has no path to build one from a `NiahTask`.

## 2. Feasibility (established during brainstorm)

- `NiahTask` does **not** currently store gold answers. But `load_niah_task` already
  calls `load_benchmark(...)`, whose returned `BenchmarkData` **carries `.answers`** —
  they are simply discarded when the `NiahTask` is built.
- Rebuilding a task from its recipe does **not** re-run the LLM: distractor texts are
  stored in the recipe JSON and spliced back deterministically. So the bridge is cheap.
- `run_rag.run(config, data=..., retriever=..., llm=...)` accepts an injected
  `BenchmarkData` and scores cover-EM over it. No change to `run_rag` is needed.

## 3. Design

### 3.1 Core change — expose answers on the task (backward-compatible)

`src/niah/types.py`: add `answers: Dict[str, List[str]] = field(default_factory=dict)`
to `NiahTask`. In `load_niah_task` (`eval/build_niah_task.py`), pass
`answers=data.answers or {}` into the `NiahTask(...)` construction. Existing consumers
(`run_niah`, `tune_corroboration`, `gate_corroboration`) ignore the new field.

### 3.2 Bridge — `eval/run_niah_rag.py`

Two pure/injectable units + a CLI:

- `niah_to_benchmark_data(task: NiahTask) -> BenchmarkData` — wraps the NIAH
  corpus-with-distractors + queries + qrels + answers into a `BenchmarkData`. Raises a
  clear `ValueError` if `task.answers` is empty (the recipe's source provided no gold).
- `run_niah_rag(task, retrievers, llm, top_k, out, per_query_out, predictions_out,
  run_fn=run_rag.run) -> None` — builds the `BenchmarkData` once, then loops the
  retrievers, calling `run_fn(config, data=data, llm=llm)` per arm with a
  `RAGEvalConfig(dataset="niah", retriever=name, top_k=..., results_path=out,
  append=True, per_query_out=per_query_out, predictions_out=predictions_out)`. **One
  shared `LLMClient`** serves retrieval-transform (q2d), corroboration reranking, and
  generation — the OOM-safe pattern from `run_niah`. `run_fn` is injected so the loop is
  unit-testable without loading models.
- CLI `main(argv)`: `--task <recipe.json>` · `--retrievers granite_dense q2d_granite
  q2d_corroborate` (default) · `--max-docs` (task rebuild cap) · `--top-k` (RAG context
  size, default 4) · `--out`/`--per-query-out`/`--predictions-out`. Loads the task
  (`load_niah_task(task, max_docs=...)`), makes one `LLMClient`, calls `run_niah_rag`,
  then prints the `eval.significance --per-query-csv <cover-EM csv> --reference
  granite_dense` command.

`run_rag` builds each retriever fresh over the injected corpus (no persistent index
cache in `RAGEvalConfig`), so `dataset="niah"` is only a label — `load_benchmark` is not
re-invoked (the `data is None` guard is skipped because we inject `data`).

### 3.3 Data flow

recipe → `load_niah_task` → `NiahTask{corpus(+counterfactual distractors), queries,
qrels, answers}` → `niah_to_benchmark_data` → for each retriever: retrieve from the
distractor haystack → RAGPipeline generate → `evaluate_rag` cover-EM/F1 vs NQ gold →
column-per-retriever per-query CSV → paired significance (q2d vs dense, corroborate vs
dense/q2d).

## 4. Outputs

- `results/rag_niah_<tag>_agg.csv` — aggregate metrics per retriever arm.
- `results/rag_niah_<tag>_cmp_answer_cover.csv` (+ `_answer_f1.csv`) — per-query, one
  column per retriever (granite_dense / q2d_granite / q2d_corroborate) → feeds
  `eval.significance --reference granite_dense`.
- prediction dumps per arm (eyeball what the model answered under distractors).

## 5. Testing (TDD)

- `load_niah_task` populates `NiahTask.answers` — synthetic recipe + a fake `loader`
  returning `BenchmarkData` with answers; assert the task carries them.
- `niah_to_benchmark_data` maps corpus/queries/qrels/answers correctly; raises
  `ValueError` on empty answers.
- `run_niah_rag` loops the arms — inject a fake `run_fn` that records `(config.retriever,
  data, llm)`; assert one call per retriever, all with the same NIAH `BenchmarkData` and
  the shared `llm`, and `config.append` set so arms accumulate one table.
- CLI arg parsing (defaults: 3 retrievers, top_k 4).

Model-loading paths (`run_rag.run` itself, the retrievers) are NOT unit-tested here —
they are exercised by the existing `run_rag`/`run_benchmark` suites and by the HPC job.

## 6. Reuse / YAGNI

Reused verbatim: `load_niah_task`, `BenchmarkData`, `run_rag.run` + its scoring/CSV
writers, the `run_benchmark` retriever builders (`q2d_corroborate` is already registered,
`run_benchmark.py:569`), `eval.significance`. No changes to `run_rag`. No new metrics
(cover-EM = existing `answer_cover`). No online/demo wiring.

## 7. Slurm — `scripts/run_niah_rag.slurm`

GPU (8B generator + q2d transform + corroboration reranking), **A100 40GB MIG slice**
(`gpu:3g.40gb:1`; the sole rtx_3090 is frequently drained). Runs the 3-arm bridge on the
frozen task, then significance. Time budget generous — `q2d_corroborate` does ~20 LLM
extractions/query on top of generation.

## 8. Risks & honest scope

- **This is the end-to-end system test**: does finding the needle survive to a correct,
  gold-matched, citable answer despite counterfactual distractors. Report it as such.
- Single designated needle on a natural multi-gold corpus (the standing NIAH caveat) —
  other relevant golds remain in the haystack; cover-EM against the NQ gold still scores
  correctness.
- `q2d_corroborate` is expensive; a first read may cap `--max-queries` if wall-clock is
  tight (the bridge forwards it via the task load / config).
- cover-EM is a lexical-cover proxy; report F1 alongside. A judge-based metric is out of
  scope.

## 9. Not doing (YAGNI)

- No changes to `run_rag`, the metrics, or the retriever registry.
- No NIAH-specific RAG prompt tuning (same prompt as the QA-set RAG runs — controls the
  comparison).
- No abstention/faithfulness scoring here (separate concern).
- No re-generation of the haystack (recipe rebuild is deterministic, LLM-free).
