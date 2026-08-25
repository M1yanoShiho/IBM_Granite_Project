# Three-module runtime handoff

The shared Git repository contains the reproducible code and metadata for:

```text
Hybrid Retriever -> trained NLI Selector -> grounded GR-C Generator
```

Large model weights, caches, indexes, datasets, and raw job outputs remain in
HPC/model storage. They must not be committed to Git.

## Repository contents

- Runtime implementations and module registration under `src/evidence_rag/`.
- Portable configuration in
  `configs/experiments/systemf_three_module_smoke_seed13.toml`.
- Model identities, revisions, checksums, and frozen policy values in
  `configs/models/three_module_seed13.json`.
- Mocked integration tests that do not require model weights or a GPU.
- `.env.example`, which documents the three external runtime paths.

## External runtime assets

| Asset | Environment variable | Storage |
|---|---|---|
| Public Retriever, Selector, Granite, and TRUE snapshots | `EVIDENCE_RAG_MODEL_CACHE` | HPC Hugging Face cache |
| Trained Selector seed-13 checkpoint | `EVIDENCE_RAG_SELECTOR_CHECKPOINT` | HPC model storage |
| Trained GR-C seed-13 adapter directory | `EVIDENCE_RAG_GENERATOR_ADAPTER` | HPC model storage |

The committed model manifest is the authority for revisions and SHA-256
values. Personal HPC paths are deliberately absent from committed files.

## Backend setup

```bash
cp .env.example .env.local
# Edit .env.local to point at readable HPC paths.
set -a
. ./.env.local
set +a
```

The runtime config expands explicit `${NAME}` references when it constructs the
real Selector and Generator. Missing variables fail with a clear error before
model loading. `HF_HOME` should point at the same shared model cache so the
Retriever resolves its public embedding model without adding personal paths to
the TOML file.

## Front-end boundary

The browser front end does not load `.safetensors` files. It should call a
backend process that owns the three-module pipeline. For UI-only development,
use a mock response with candidate evidence, selected evidence, answer, and
citation fields. This repository currently provides the Python pipeline but no
HTTP API, so an API adapter remains a separate integration task.

## Commit boundary

Commit source code, portable configuration, manifests, tests, small fixtures,
and aggregate reports. Keep model files, full datasets, indexes, caches,
virtual environments, secrets, and raw per-query/job outputs outside Git.

## Recommended first pull request

Keep the deployable three-module wiring separate from the large Experiment 04
and Experiment 05 result archives. The first pull request should contain only:

- `.env.example`, `.gitignore`, `README.md`, and `pyproject.toml`;
- `src/evidence_rag/composition.py`;
- the required Retriever/Selector/Generator runtime modules under
  `src/evidence_rag/`;
- `configs/experiments/systemf_three_module_smoke_seed13.toml` and
  `configs/models/three_module_seed13.json`;
- the three-module smoke fixture and focused runtime tests;
- this handoff document and the three-module wiring smoke report.

Experiment 04 and Experiment 05 scripts, per-query outputs, audits, and report
archives should be reviewed in separate commits. This prevents front-end
integration from depending on unrelated experimental history.
