# Evidence RAG Architecture

The final system composes three independently testable modules behind a versioned HTTP service.
The Pipeline coordinates data movement but does not implement retrieval, selection, or generation
algorithms.

## 🏗️ Module responsibilities

| Component | Input | Output | Frozen implementation |
|---|---|---|---|
| Hybrid Retriever | Query and corpus/index | Ranked immutable candidates | Strong BM25 and Granite dense retrieval fused with reciprocal rank fusion |
| Trained NLI Selector | Query and candidates | Selected evidence IDs, scores, and ranks | Seed-13 risk-controlled NLI model with a safe threshold and two-deletion cap |
| Grounded GR-C Generator | Query, checklist, and selected evidence | Answer and cited selected-evidence IDs | Granite 4.1 3B with seed-13 GR-C adapter and frozen verification/annotation |
| Pipeline | Module outputs | End-to-end trace | Contract enforcement and evidence-ID resolution |
| HTTP API | Versioned JSON | Health and query responses | FastAPI app with lazy pipeline loading |

The canonical runtime identity is stored in
[`configs/runtime/final_seed13.toml`](../configs/runtime/final_seed13.toml) and
[`configs/models/final_seed13.json`](../configs/models/final_seed13.json).

## 🔄 Request lifecycle

```mermaid
flowchart LR
    accTitle: Three Module Request Lifecycle
    accDescr: A validated query moves through retrieval, evidence selection, generation, and response validation while the Pipeline preserves evidence identity

    validate[🌐 Validate API query] --> retrieve[🔍 Retrieve candidates]
    retrieve --> select[🧠 Select evidence IDs]
    select --> resolve[🔗 Resolve original evidence]
    resolve --> generate[⚙️ Generate cited answer]
    generate --> validate_response[✅ Validate response]

    classDef api fill:#f3f4f6,stroke:#6b7280,stroke-width:2px,color:#1f2937
    classDef module fill:#dbeafe,stroke:#2563eb,stroke-width:2px,color:#1e3a5f
    classDef success fill:#dcfce7,stroke:#16a34a,stroke-width:2px,color:#14532d

    class validate api
    class retrieve,select,resolve,generate module
    class validate_response success
```

## 🔒 Runtime invariants

The contracts and tests enforce the following boundaries:

1. Candidates have stable evidence, document, and chunk identities.
2. The Selector returns IDs and scores; it does not rewrite evidence text.
3. The Pipeline resolves selected IDs against the original CandidateSet.
4. The Generator receives only selected evidence and may cite only those IDs.
5. A response exposes candidates, selected evidence, citations, and module diagnostics so failures
   can be localized.
6. Missing or mismatched real assets fail before model execution rather than silently falling back
   to a different method.

Architecture tests under [`tests/architecture/`](../tests/architecture/) prevent concrete module
implementations from importing one another. Contract tests under
[`tests/contracts/`](../tests/contracts/) freeze the shared data schemas.

## 💾 External assets and loading

Model weights, dataset manifests, indexes, and writable outputs are supplied through explicit
environment variables. Startup validates every required setting and compares external files with
the committed SHA-256 identities before the real pipeline is built.

`GET /health` does not load models. The first valid `POST /v1/query` request constructs the real
pipeline once; subsequent queries reuse it. This keeps health checks cheap while preserving
fail-closed behavior for real requests.

See [models and data](models-and-data.md) for availability and licensing, and the
[runtime handoff](three-module-runtime-handoff.md) for the environment contract.

## 🌐 Front-end boundary

The front end calls the backend API and never reads checkpoints or HPC storage. The backend may run
on a workstation or a cluster node as long as the authorized assets are available there. During UI
development, the committed [mock response](../examples/mock_frontend_response.json) provides the
same response schema without a running model service.

The exact endpoint fields and error behavior are defined in
[front-end integration](frontend-integration.md).
