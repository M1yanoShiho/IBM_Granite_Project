# Three-module runtime handoff

The frozen runtime is:

```text
Hybrid Retriever -> trained NLI Selector -> grounded GR-C Generator
```

The repository contains its source, versioned API contract, portable configuration,
model manifest, CPU smoke, and tests. Model weights, datasets, indexes, caches, and raw
job output remain outside Git.

## Canonical files

- `configs/runtime/final_seed13.toml`: deployable seed-13 runtime configuration.
- `configs/runtime/cpu_smoke.toml`: offline contract smoke using the final module classes.
- `configs/models/final_seed13.json`: model revisions, SHA-256 values, and storage policy.
- `src/evidence_rag/composition.py`: three-module construction and asset validation.
- `src/evidence_rag/api/`: stable HTTP schema, service, and application factory.
- `examples/mock_frontend_response.json`: model-free frontend fixture.

## External assets

| Asset | Environment variable |
|---|---|
| Dataset manifest | `EVIDENCE_RAG_DATASET_MANIFEST` |
| Writable output directory | `EVIDENCE_RAG_OUTPUT_DIR` |
| Frozen public model snapshots | `EVIDENCE_RAG_MODEL_CACHE` |
| Trained Selector seed-13 checkpoint | `EVIDENCE_RAG_SELECTOR_CHECKPOINT` |
| Trained GR-C seed-13 adapter directory | `EVIDENCE_RAG_GENERATOR_ADAPTER` |
| Optional prebuilt Retriever index | `EVIDENCE_RAG_INDEX_DIR` |

The manifest is the authority for model revisions and checksums. Startup checks all
required variables before model loading and verifies the committed checksums against the
external files. Personal HPC paths and model binaries must not be committed.

## Verification and launch

The CPU smoke downloads nothing and needs no GPU:

```bash
python -m pip install -e '.[dev,api,data-prep]'
evidence-rag-smoke
```

To run the real backend, copy `.env.example` to an ignored `.env.local`, set the five
required paths, load them into the shell, and start the service:

```bash
set -a
. ./.env.local
set +a
evidence-rag-serve --host 127.0.0.1 --port 8000
```

Use `--allow-origin http://localhost:3000` when a browser frontend is served from that
origin. Do not use a wildcard origin for an authenticated deployment.

## Frontend boundary

The frontend calls `POST /v1/query`; it never reads `.safetensors`, adapters, caches, or
HPC paths. `GET /health` is intentionally cheap and does not load models. The backend
loads the pipeline on the first query and reuses it for later requests.

See `docs/frontend-integration.md` for the frozen request/response contract. When the
backend is unavailable, use `examples/mock_frontend_response.json` in the frontend.

## Git boundary

Commit source code, portable configuration, manifests, tests, small fixtures, aggregate
reports, and documentation. Keep model files, full datasets, indexes, caches, virtual
environments, secrets, and raw per-query/job outputs outside Git.
