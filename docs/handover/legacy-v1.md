# Legacy V1 handover

## Recovery

- Tag: `legacy-before-three-module-reset-2026-07-13`
- Commit: `071e4358dc2e2accc9358b67fc5146539bf790a6`
- Temporary recovery: `git worktree add ../IBM_Granite_Project_legacy legacy-before-three-module-reset-2026-07-13`
- One-file recovery example: `git show legacy-before-three-module-reset-2026-07-13:src/rag_pipeline.py`

## What was done

The archived repository explored dense and hybrid retrieval, Query2Doc, NIAH and
artificial counterfactual construction, corroboration reranking, Astute RAG, and ML
Selector V1.

Selector V1 used fixed Top-20 candidates and selected Top-10 evidence. It tested
relevance, rank, support, source, condition, and learned reranking signals.

## What the work showed

- Some approaches improved whether required evidence entered the final context window.
- Those gains did not prove that harmful or misleading evidence was reliably removed.
- Selector V1 did not pass its full promotion gates and was not connected as the new
  production Selector.
- Artificial-needle and counterfactual construction reflected one diagnostic setup,
  not the final task definition.
- Module responsibilities and interfaces were mixed, motivating the clean reset.

## Result disposition

The compact conclusions were already recorded in the old tracked documentation and are
recoverable from the tag. The local `results/ml_selector` directory contained the full
non-final experiment output and was intentionally deleted during the reset.

Do not present V1 diagnostics as results of the new Pipeline.
