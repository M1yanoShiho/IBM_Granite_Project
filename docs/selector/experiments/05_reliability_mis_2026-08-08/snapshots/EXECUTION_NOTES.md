# Selector experiment execution notes

Date: 2026-08-09  
Host: `fl25387@10.70.71.11`  
Final experiment/report commit: `39e573935ddcb79a2fe7c8a9610c0d6d6f47e5e0`  
Final Git status: clean

## Frozen controls

- Retriever: Hybrid RRF (`strong-bm25` + `ibm-granite/granite-embedding-english-r2`, `k=60`, Top-20).
- Baseline Selector: TopK, maximum 10 items.
- Candidate Selector: Reliability-MIS, maximum 10 items.
- Answer extraction: `ibm-granite/granite-4.1-3b`, temperature 0, max 32 new tokens.
- NLI: `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli@b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7`, threshold 0.5, batch 32.
- Statistics: query-level paired comparison, 10,000 iterations, seed 20260809.
- No Retriever, Selector method, threshold, model, or success criterion was changed after formal data were opened.

## Formal commands

Candidate pools were generated once with:

```text
python -m evidence_rag.cli.experiment --config configs/experiments/selector_mis_sealed600_pool.toml retriever
python -m evidence_rag.cli.experiment --config configs/experiments/selector_mis_2wiki_pool.toml retriever
```

Both pools were frozen with `evidence_rag.cli.selector_pool_audit`, including the dataset manifest, source-parent file, Retriever config, actual embedding revision, and candidate-file SHA-256.

Formal comparisons used `evidence_rag.cli.selector_compare` with:

```text
--top-n 20 --max-selected 10 --run-seed 13 --stats-seed 20260809
--iterations 10000 --nli-batch-size 32 --resume
```

The sealed600 run additionally used its provenance file and `--min-harmful-pool-hits 200`. The 2Wiki run had no harmful labels and therefore evaluated supporting-document protection only.

The pre-frozen 60-query stability subset reused the exact sealed600 pool and settings through `selector_compare --query-ids .../stability_query_ids.txt`; `selector_stability` then compared selected IDs exactly.

The final package was produced with `selector_error_analysis` and `selector_final_report`.

## Runtime incident and recovery

One simultaneous SSH/VPN interruption stopped the two foreground comparison processes. Checkpoints were intact at sealed600 150/600 and 2Wiki 950/2000. Both jobs resumed with unchanged arguments and later ran under `nohup` with logs in `runs/selector-reliability-mis/logs/`. Completed records were loaded from checkpoint rather than recomputed.

## Verification

- sealed600: 600 comparison records; all four arm artifacts match recorded SHA-256; deterministic 20-query audit passed.
- 2Wiki: 2,000 comparison records; all four arm artifacts match recorded SHA-256; deterministic 20-query audit passed.
- Stability: 60/60 exact selected-ID agreement for both arms.
- Final regression: 309 relevant tests passed; mypy passed for 131 source files; Ruff passed.
