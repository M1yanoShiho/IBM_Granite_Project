# Dissertation reproducibility map

This map connects each paper-facing claim to its frozen implementation, configuration, input
identity, result, and recovery source. It distinguishes technical completion from scientific
support and includes negative findings.

## Runtime method

| Component | Code | Frozen configuration / identity | Verification |
|---|---|---|---|
| Hybrid Retriever | `src/evidence_rag/retriever/` | `configs/runtime/final_seed13.toml`; `configs/models/final_seed13.json` | runtime architecture and CPU-smoke tests |
| NLI risk Selector | `src/evidence_rag/selector/`; `src/evidence_rag/cli/run_selector_lean.py` | `configs/selector/lean_v3.toml`; seed13 checkpoint SHA in final model manifest | Selector focused tests; two public result gates |
| Grounded GR-C Generator | `src/evidence_rag/generator/`; eight Generator reproduction scripts | `experiments/generator/frozen_provenance.json`; seed13 adapter SHA in final model manifest | Generator focused tests and qualification scripts |
| Three-module composition | `src/evidence_rag/pipeline/`; `src/evidence_rag/composition.py` | `configs/runtime/final_seed13.toml` | offline CPU smoke and API contract tests |

## Scientific claims and results

| Claim or table | Code and statistical rule | Input identity | Public result | Scientific decision |
|---|---|---|---|---|
| Selector removes misleading evidence selectively | Lean v3 final evaluation in `run_selector_lean.py`; 10,000 bootstrap samples, seed 13 | R005AB L003 archive artifact, SHA `b031b6f2…2495c` | `results/selector/misleading_evidence_summary.json` | Evidence gate `PASS` |
| Selector improves final answers | Same frozen policy; equal-weight NIAH/2Wiki answer delta | Same L003 source and SHA | `results/selector/blind_answer_gate.json` | `FAIL`; decision `KEEP_TOPK10` |
| Exp04 Table 1: four baselines vs Ours seeds 13/42/73 | `experiment04_goal3.py`; five frozen metrics; mean/sample SD over Generator seeds | sealed three-dataset Goal3 bundle; source hashes in `final_audit.json` | `results/experiment04/table1.json`; `results/experiment04/final_results.json` | Ours-RAR superiority `NOT_SUPPORTED` |
| Exp04 Table 2: Retriever/Selector/Generator ablations | `experiment04_goal4.py`; Full-minus-ablation paired component bootstrap | sealed Goal4 bundle and reused Full seed13 rows | `results/experiment04/table2.json`; `results/experiment04/final_results.json` | Direct Generator higher on HotpotQA/MuSiQue; other registered RAR differences absent/inconclusive |
| Exp04 audit and table build | `experiment04_goal5.py`; pure public renderer in `public_tables.py` | raw archive bundle for full recomputation or `final_results.json` for table-only verification | `results/experiment04/final_audit.json` and six rebuilt files | Technical `FINAL PASS` |
| Exp05 Table 1: five reported systems | Experiment05 data/retrieval/generation/scorer modules; Ours mean/SD over three seeds | 3 datasets × 7 main arms × 400; frozen external runroot | `results/experiment05/table1.json` | Mixed trade-off; no blanket superiority claim |
| Exp05 Table 2: three module ablations | Same scorer; Full seed13 vs one substitution at a time | 3 datasets × 3 new ablations × 400 plus Full reuse | `results/experiment05/table2.json` | Selector inactive on ordinary datasets; Generator/Retriever effects are dataset-dependent |
| Exp05 registered Claim A/B | `experiment05_compile_results.py`; 10,000 bootstrap resamples, seed 13, frozen harm/noninferiority/superiority rules | 12,000 outputs and 12,000 query scores; 0 scorer errors | `results/experiment05/bootstrap.json`, `results/experiment05/claim_labels.json`, `results/experiment05/final_audit.json` | Both `NOT SUPPORTED`; technical `FINAL PASS` |

## Rebuild commands

```bash
python experiments/experiment04/build_tables.py --output-dir build/experiment04
python experiments/experiment05/build_tables.py --output-dir build/experiment05
pytest -q tests/evaluation/test_public_table_rebuild.py \
  tests/evaluation/test_public_selector_results.py \
  tests/experiments
```

The builders operate only on small committed aggregate inputs and compare against frozen
bytes/hashes. They do not recalculate model scores or change claim labels.

## External and archived inputs

Model weights, dataset caches, indexes, raw generations, per-query scores, and full training
bundles are not committed to the public release tree. Content hashes are recorded in the final
model manifest, Generator provenance, experiment audits, and result JSON. The immutable Git
reference `research-archive-2026-08-25` recovers the complete development-time manifests and
per-query experiment records; public download locations and licenses are maintained in the
release asset manifest rather than embedded as personal HPC paths.
