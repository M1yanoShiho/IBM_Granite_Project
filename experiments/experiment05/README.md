# Experiment 05: general RAG evaluation

Experiment 05 evaluates KILT-NQ, KILT-TriviaQA, and ALCE-ASQA across five reported systems,
three Generator seeds, and three single-module ablations. The frozen execution produced and
scored 12,000 outputs with zero scorer errors.

## Quick table verification

```bash
python experiments/experiment05/build_tables.py --output-dir build/experiment05
```

This reads only the audited `table1.json`, `table2.json`, and `claim_labels.json` files under
`results/experiment05/`. It rebuilds both CSVs, the LaTeX table, and the final report without
access to raw generations.

## Full execution chain

1. `experiment05_goal1.py`, `experiment05_goal1_index.py`, and
   `experiment05_goal1_scorer_validation.py`: inventory datasets, audit exposure, freeze 400
   IDs per dataset, build the complete-corpus index, and validate the scorer.
2. `experiment05_rapid_retrieval.py`: freeze complete-corpus BM25 Top-1000 retrieval traces.
3. `experiment05_goal2_prepare.py` and `experiment05_goal2_audit.py`: prepare and audit all
   model-facing arms from the runtime and retrieval bundles.
4. `experiment05_generate.py`: run direct and grounded generation without reading scorer gold.
5. `experiment05_freeze_outputs.py` and import helpers: freeze or import complete remote bundles
   while checking identity and ordering.
6. `experiment05_score_arm.py`: run the registered real-ground-truth plus model-judged scorer
   for one dataset/arm only after its generation bundle is frozen.
7. `experiment05_compile_results.py`: compile all 30 dataset-arm bundles, run 10,000 bootstrap
   resamples with seed 13, apply the registered Claim A/B rules, and emit the integrity audit.

The raw runroot, dataset caches, indexes, generations, claim traces, model snapshots, and
adapters are external because they are large or license-governed. Their frozen counts and
hashes are recorded in `results/experiment05/final_audit.json` and the immutable archive.

## Frozen conclusion

Technical status is `FINAL PASS`, but Claim A and Claim B are both `NOT SUPPORTED`. The system
shows citation-quality gains alongside factual-coverage and response-rate trade-offs. The
ordinary-dataset Selector activation rate was zero, so those near-identical Full/keep-all rows
cannot establish a causal Selector uplift; the separate misleading-evidence result is reported
under `results/selector/`.
