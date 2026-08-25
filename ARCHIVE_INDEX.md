# Complete Research History Archive Index

This index identifies the complete pre-release research snapshot retained before the dissertation
repository is curated. It is an archive navigation document, not the public quickstart.

## Immutable references

- Branch: `archive/full-research-history-2026-08-25`
- Annotated tag: `research-archive-2026-08-25`
- Source development branch at freeze time: `refactor/three-module-baseline`

Resolve the exact frozen commit with:

```bash
git rev-parse research-archive-2026-08-25^{commit}
```

Read a historical file without changing branches:

```bash
git show research-archive-2026-08-25:path/to/file
```

Restore the complete snapshot in a separate worktree:

```bash
git worktree add ../evidence-rag-research-archive research-archive-2026-08-25
```

## Final connected system at freeze time

The final runtime route is:

```text
Strong BM25 + Granite dense + RRF
→ trained seed-13 NLI risk-controlled Selector
→ Granite + seed-13 GR-C adapter + grounded verification/annotation
```

Primary navigation:

| Responsibility | Archive path |
|---|---|
| Three-module handoff | `docs/three-module-runtime-handoff.md` |
| Seed-13 runtime config | `configs/experiments/systemf_three_module_smoke_seed13.toml` |
| Seed-13 model manifest | `configs/models/three_module_seed13.json` |
| Runtime composition | `src/evidence_rag/composition.py` |
| End-to-end pipeline | `src/evidence_rag/pipeline/service.py` |
| Hybrid Retriever | `src/evidence_rag/retriever/` |
| Trained and baseline Selectors | `src/evidence_rag/selector/` |
| Grounded and direct Generators | `src/evidence_rag/generator/` |

Top-K, threshold-only, Provence, Granite rerank and Direct Granite are retained as labelled
baselines/ablations. Top-K is not the final trained Selector.

## Selector research history

| Route | Archive path | Purpose |
|---:|---|---|
| 01 | `docs/selector/experiments/01_corroboration_reranking_2026-07-05/` | Corroboration reranking baseline |
| 02 | `docs/selector/experiments/02_ml_selector_v1_2026-07-10/` | First learned Selector |
| 03 | `docs/selector/experiments/03_gated_graph1_2026-07-20/` | Gated Graph 1 route |
| 04 | `docs/selector/experiments/04_graph2_relation_layer_2026-07-30/` | Relation/Graph 2 route |
| 05 | `docs/selector/experiments/05_reliability_mis_2026-08-08/` | Misleading-evidence reliability experiment |
| 06 | `docs/selector/experiments/06_beam_selector_2026-08-09/` | Beam Selector route |
| 07 | `docs/selector/experiments/07_adaptive_conservative_r001_r005_2026-08-11/` | Adaptive conservative risk-control route |
| 08 | `docs/selector/experiments/08_r005ab_repair_current_2026-08-12/` | Final repair/current learned Selector route |

Selector training, materialisation and evaluation implementations remain under:

- `src/evidence_rag/selector/`
- `src/evidence_rag/materializer/`
- `src/evidence_rag/evaluation/selector_*.py`
- `src/evidence_rag/cli/*selector*.py`
- `scripts/selector_*.py` and associated tests

## Generator and full-flow research history

| Route | Archive path | Purpose |
|---:|---|---|
| 01 | `docs/full-flow/experiments/01_selector_generator_bridge_2026-08-12/` | Selector–Generator bridge |
| 02 | `docs/full-flow/experiments/02_generator_selector_alignment_2026-08-15/` | Generator/Selector objective alignment |
| 03 | `docs/full-flow/experiments/03_generator_grounding_repair_2026-08-18/` | GR-C training, seed screening, qualification and grounding repair |
| 04 | `docs/full-flow/experiments/04_frozen_three_module_system_evaluation_2026-08-21/` | Frozen three-module evaluation and module ablations |
| 05 | `docs/full-flow/experiments/05_general_rag_reliability_traceability_evaluation_2026-08-22/` | General RAG reliability and traceability evaluation |

The route-03 `artifacts/` and `reports/` directories retain the G300/G310/G320/G330/G400/G410
training and qualification chain, including seed 13/42/73 provenance and hashes. The actual model
weights remain external.

## Final Experiment 04 evidence

Root:
`docs/full-flow/experiments/04_frozen_three_module_system_evaluation_2026-08-21/`

Important files:

| File | Purpose |
|---|---|
| `FINAL_REPORT.md` | Final interpretation and validity boundary |
| `results/final_results.json` | Frozen machine-readable tables, bootstrap results and claim decisions |
| `results/final_audit.json` | Cross-goal hashes, counts and integrity checks |
| `results/summary_metrics.csv` | Aggregated metrics |
| `results/FINAL_TABLES.md` | Human-readable final tables |
| `results/TABLE1.tex`, `results/TABLE2.tex` | Dissertation LaTeX tables |
| `results/per_query_metrics.csv` | Goal-3 per-query archive input; archive-only in the public cleanup |
| `results/per_query_goal4_metrics.csv` | Goal-4 per-query archive input; archive-only in the public cleanup |

Scientific boundary: technical execution is `FINAL PASS`, while whole-system RAR superiority is
not supported. Direct Granite is higher on parts of the evaluation, identifying the frozen GR-C
Generator as the main system-level weakness.

## Final Experiment 05 evidence

Root:
`docs/full-flow/experiments/05_general_rag_reliability_traceability_evaluation_2026-08-22/`

Important files:

| File | Purpose |
|---|---|
| `EXPERIMENT_AUDIT.md`, `EXPERIMENT_AUDIT.json` | Independent protocol/integrity review |
| `results/final/FINAL_REPORT.md` | Final public-facing experiment report |
| `results/final/table1.csv`, `table1.json` | Main-system comparison |
| `results/final/table2.csv`, `table2.json` | Module ablations |
| `results/final/bootstrap.json` | Frozen confidence intervals |
| `results/final/claim_labels.json` | Registered claim decisions |
| `results/final/final_audit.json` | Nine final artifact hashes and output counts |
| `results/final/tables.tex` | Dissertation LaTeX tables |

Scientific boundary: Claim A and Claim B are both `NOT SUPPORTED`. Technical `FINAL PASS` means
the frozen protocol completed; it does not mean system superiority. On ordinary Experiment 05
data, the trained Selector was mostly inactive.

## Dedicated Selector contribution evidence

The strongest supported Selector result is conditional and evidence-level: the trained Selector
reduces harm in the registered misleading-evidence stress test. The dedicated blind answer gate
does not show a supported answer-level uplift. Preserve both positive and negative findings when
using the archive.

Primary route:
`docs/selector/experiments/05_reliability_mis_2026-08-08/`

## External model assets

No model weights are stored in Git. The archive records the following logical assets:

| Asset | Frozen identifier/checksum |
|---|---|
| Granite embedding Retriever | `ibm-granite/granite-embedding-english-r2` revision `47ea694b257b703fee9253d75c2b1f2985180498` |
| Trained Selector seed 13 | `model.safetensors` SHA-256 `86622bd9ab6391c9eb560133b01b0cf3744c3706638ff8b0bd38925b84bf72bf` |
| Granite Generator base | `ibm-granite/granite-4.1-3b` revision `c0650403e44e78ec0262dab1c90914c65b196c4e` |
| GR-C Generator seed-13 adapter | weights SHA-256 `492d336c7acc32785becded707220dbdf1a8acf9a895630147fc4e8cd28d0707` |
| TRUE verifier | `google/t5_xxl_true_nli_mixture` revision `aa6cfe1dd4257853bfdd772992045f41bfc14988` |

Seed-42/73 Generator provenance is retained under the route-03 G330/G400/G410 artifacts. Their
public download locations, exact released sizes and redistribution permissions must be resolved by
the public release artifact goal; historical HPC absolute paths are evidence only, not portable
download instructions.

## Development-process records

The archive intentionally retains items that will not appear on the public default branch:

- `.aris/`, `refine-logs/` and internal planning/tracker material;
- superseded plans and snapshots;
- raw/per-query results and intermediate artifacts;
- historic absolute HPC paths and machine-specific run manifests;
- `task_plan.md`, `findings.md`, `progress.md` and the public-release cleanup design documents.

These records preserve chronology and team collaboration. They must not be copied back into the
clean release unless a documented runtime or reproduction dependency requires them.

## Freeze-time verification baseline

- Focused Experiment 04/05 and new baseline tests: pass.
- Ruff across source, tests and the new Experiment 04/05 scripts: pass.
- Strict mypy: `147 source files`, no issues after non-behavioural Exp05 type corrections.
- Result integrity: Experiment 04 deterministic rebuild matches the frozen tree; Experiment 05
  nine artifact hashes and claim labels match; 38 Experiment 04/05 JSON files parse.
- Full pytest: `1877 passed`, `20 skipped`, `1 failed`.
- The one failure is the historical G000 README/TRACKER textual-state assertion. It is unrelated to
  the final Experiment 04/05 code and is retained as an honest archive baseline.

## Public release relationship

The clean release is built from this archive commit on `release/dissertation-v1`. Moving files out
of the release tree does not remove them from this archive branch, tag or Git history. The release
must never rewrite this tag or force-push this branch.
