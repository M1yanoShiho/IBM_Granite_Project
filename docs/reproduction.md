# Evidence RAG Reproduction

The release supports three reproducibility levels. Public table verification is fully contained in
Git; full raw recomputation and real-model training require separately authorized assets.

## 📋 Reproducibility levels

| Level | Publicly runnable | Inputs |
|---|---|---|
| Runtime contract | Yes | Source, CPU doubles, fixtures, and frozen configuration |
| Dissertation table rebuild | Yes | Small audited aggregate JSON files under `results/` |
| Full scoring and training | Conditional | External model weights, raw datasets, indexes, generations, and run bundles |

The root [reproducibility map](../REPRODUCIBILITY_MAP.md) connects every paper-facing claim to its
implementation, frozen configuration, result, and immutable archive source.

## 🧪 Verify the runtime contract

```bash
python -m pip install -e '.[dev,api]'
evidence-rag-smoke
pytest -q tests/runtime tests/api tests/pipeline tests/contracts
```

This level proves module composition, evidence identity, API schemas, offline behavior, and
fail-closed configuration. It does not claim that CPU doubles reproduce model quality.

## 🔧 Run the reference baseline workflow

The small reference fixture persists frozen artifacts for every module and runs a live Pipeline.
It is an interface/provenance baseline using BM25, Top-K, and an extractive Generator;
it is not the final trained method.

Run one stage at a time or the complete workflow:

```bash
python -m evidence_rag.cli.experiment --config configs/experiments/reference_baseline.toml prepare
python -m evidence_rag.cli.experiment --config configs/experiments/reference_baseline.toml retriever
python -m evidence_rag.cli.experiment --config configs/experiments/reference_baseline.toml selector
python -m evidence_rag.cli.experiment --config configs/experiments/reference_baseline.toml generator
python -m evidence_rag.cli.experiment --config configs/experiments/reference_baseline.toml pipeline
python -m evidence_rag.cli.experiment --config configs/experiments/reference_baseline.toml all
```

Preparation writes the validated index identity and corpus snapshot to:

```text
runs/reference-baseline/index/index_manifest.json
runs/reference-baseline/index/corpus_snapshot.json
```

Later stages fail if these files or their provenance do not match, rather than rebuilding an
unregistered index silently.

## 📊 Rebuild dissertation tables

```bash
python experiments/experiment04/build_tables.py --output-dir build/experiment04
python experiments/experiment05/build_tables.py --output-dir build/experiment05
pytest -q tests/evaluation/test_public_table_rebuild.py \
  tests/evaluation/test_public_selector_results.py \
  tests/experiments
```

Experiment 04 reads only [`final_results.json`](../results/experiment04/final_results.json) and
rebuilds six checked-in table files. Experiment 05 reads only its audited table and claim JSON
files and rebuilds both CSVs, the LaTeX output, and the final report. Golden tests compare bytes and
claim labels; the builders cannot rescore queries or rewrite registered decisions.

## 🧠 Reproduce Selector training and evaluation

The final training CLI and recipe remain public:

```bash
evidence-rag-selector-train --help
```

[`configs/selector/lean_v3.toml`](../configs/selector/lean_v3.toml) freezes data roles, schedule,
NLI backbone, risk threshold selection, bootstrap settings, and the separate blind answer gate.
The complete datasets and trained checkpoints are external. The two public Selector result files
share one immutable source SHA so the evidence PASS cannot be separated from the answer FAIL.

## ⚙️ Reproduce Generator and full experiments

The public Generator entry-point map and frozen three-seed provenance are under
[`experiments/generator/`](../experiments/generator/). Experiment 04 and Experiment 05 document
their formal stages in their own README files:

- [Experiment 04](../experiments/experiment04/README.md)
- [Experiment 05](../experiments/experiment05/README.md)

Raw recomputation requires the external bundles registered in
[`ARTIFACT_MANIFEST.json`](../ARTIFACT_MANIFEST.json). The manifest distinguishes immutable public
sources from restricted derived assets and mixed-license data. Verify every local asset before use.

## 🔒 Archive and integrity boundary

The immutable `research-archive-2026-08-25` Git reference retains development-time manifests,
per-query records, and superseded stages that are intentionally absent from the release tree.
Public aggregate files record their source hashes and archive paths. This allows an authorized
reviewer to recover the full chain without making restricted data or personal infrastructure part
of the public runtime.

`FINAL PASS` is a protocol execution status. Scientific claim labels are evaluated separately and
must be read from [results](results.md).
