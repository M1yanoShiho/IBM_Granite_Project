# Evidence RAG Limitations

This release is research software with restricted real-model assets and mixed experimental
findings. The public CPU smoke proves interfaces and invariants, not model effectiveness.

## 📊 Scientific limitations

- The Selector's positive result is limited to an evidence-level misleading-evidence stress test.
  The separate blind answer gate failed and retained the `KEEP_TOPK10` decision.
- The Selector was mostly inactive on ordinary Experiment 05 inputs. The experiment therefore does
  not support a general ordinary-data Selector improvement.
- Experiment 04 did not support the registered whole-system RAR superiority claim. Its Generator
  ablation identified GR-C seed 13 as a limiting component on two tested datasets.
- Experiment 05 registered Claims A and B were both `NOT SUPPORTED`.
- `FINAL PASS` denotes protocol execution and integrity completion only; it is not evidence that a
  scientific claim or production-quality threshold passed.
- Dataset coverage is limited to the frozen benchmarks and sample sizes. Results should not be
  generalized to other domains, languages, time periods, or retrieval corpora without evaluation.

## 🔒 Asset and reproducibility limitations

The trained Selector checkpoint and three GR-C adapters are registered by exact size and SHA-256,
but redistribution authorization has not been recorded. They are therefore marked
`restricted-not-published`. A public reader can run the CPU smoke, API schemas, table builders, and
most tests, but cannot run the real final pipeline without authorized derived weights.

Full raw recomputation also requires external datasets, indexes, generation bundles, and per-query
scores. The immutable archive preserves provenance and hashes, while the public Git tree contains
only compact aggregates. See [models and data](models-and-data.md) and
[reproduction](reproduction.md).

## ⚖️ Third-party licensing limitations

Upstream models and datasets have different terms. In particular:

- Provence carries conflicting upstream license labels and additional terms, so this project
  applies a conservative link-only boundary and does not mirror the weights.
- RGB is non-commercial under CC-BY-NC-SA-4.0.
- KILT and ALCE repository software licenses do not replace the terms inherited from their
  aggregated datasets and Wikipedia content.

The project license covers only material for which the project can grant rights. It does not
relicense third-party models, datasets, or restricted derived assets.

## 🖥️ Resource limitations

The final runtime depends on multiple upstream snapshots, including a 3B Generator and a large
TRUE verifier. Storage and accelerator requirements are materially larger than the Git checkout.
CPU real-model inference may be impractical; actual throughput and memory depend on operator
hardware, model precision, sequence lengths, and index size.

## 🌐 Deployment limitations

The provided API is a research integration boundary. It does not include authentication, rate
limiting, multi-tenant isolation, persistent session storage, or production observability. Backend
operators must add these controls, restrict CORS origins, protect asset paths, and comply with
institutional security policy before deployment.

Generated answers may still be incorrect, incomplete, or weakly supported. Citations expose the
evidence used by the pipeline; they do not guarantee that the underlying source is true or that the
answer is safe for high-stakes decisions.
