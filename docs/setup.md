# Evidence RAG Setup

This guide covers the public offline smoke, development installation, and authorized real-model
runtime. Python 3.11 is required by the frozen package metadata.

## 🚀 Install and verify

```bash
git clone https://github.com/M1yanoShiho/IBM_Granite_Project.git
cd IBM_Granite_Project
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,api]'
evidence-rag-smoke
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1` instead. The smoke command uses
the final module classes with deterministic CPU doubles. It must run without a network connection,
model cache, GPU, or private data. A successful JSON trace shows that the misleading fixture was
not selected and that the answer cited only retained evidence.

## 🧪 Run development checks

```bash
pytest
ruff check src tests scripts experiments
mypy
python -m build
python -m pip check
```

The complete release validation also performs clean-clone, documentation-link, personal-path,
secret, artifact, and HPC smoke checks. Those results are published with the release candidate.

## 📦 Prepare real model assets

Real execution requires public upstream snapshots plus authorized copies of the trained Selector
and GR-C adapter. The derived weights are not public downloads. Inspect the manifest first:

```bash
evidence-rag-artifacts list
evidence-rag-artifacts verify \
  --asset selector-seed13 \
  --file /path/to/authorized/model.safetensors
```

For upstream files that have a download URL, `evidence-rag-artifacts download` writes atomically
and validates both bytes and SHA-256. Multi-file models are downloaded one registered shard at a
time. Read [models and data](models-and-data.md) before acquiring any third-party asset.

## ⚙️ Configure the real runtime

Copy the template to an ignored local file and set paths available to the backend process:

```bash
cp .env.example .env.local
```

| Variable | Purpose |
|---|---|
| `EVIDENCE_RAG_DATASET_MANIFEST` | Compatible dataset/corpus manifest |
| `EVIDENCE_RAG_OUTPUT_DIR` | Writable runtime output directory |
| `EVIDENCE_RAG_MODEL_CACHE` | Cache containing frozen upstream snapshots |
| `EVIDENCE_RAG_SELECTOR_CHECKPOINT` | Authorized seed-13 Selector checkpoint |
| `EVIDENCE_RAG_GENERATOR_ADAPTER` | Authorized seed-13 GR-C adapter directory |
| `EVIDENCE_RAG_INDEX_DIR` | Optional compatible prebuilt Retriever index |

Do not commit `.env.local`, account names, hostnames, private paths, access tokens, model files, or
dataset caches.

Load the variables and start the API:

```bash
set -a
. ./.env.local
set +a
evidence-rag-serve --host 127.0.0.1 --port 8000
```

For a browser frontend on another local port, repeat `--allow-origin` for every explicit origin.
Avoid wildcard CORS in an authenticated deployment.

## ✅ Check the service

```bash
curl http://127.0.0.1:8000/health
python examples/api_request.py
```

`GET /health` returns before real models load. The first query validates and loads the configured
pipeline; missing or mismatched assets produce a generic `503` response. See
[front-end integration](frontend-integration.md) for the frozen schema.

## 🖥️ Cluster use

Install the repository in a project environment on the cluster, keep large assets in storage
authorized for the backend operator, and expose only the HTTP endpoint required by the frontend.
Never share a personal account or embed private storage paths in browser code. Scheduler and GPU
settings are site-specific; the repository intentionally does not hard-code them.
