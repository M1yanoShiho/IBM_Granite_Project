# Contributing to Evidence RAG

Contributions are welcome when they preserve the module contracts, research provenance, and honest
reporting boundaries of the release.

## 🚀 Development setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,api]'
evidence-rag-smoke
```

Create a focused branch from `main`, keep unrelated changes separate, and include tests for new or
changed behavior. Open an issue before changing a frozen experiment definition, public schema, or
artifact identity.

## 🧪 Required checks

```bash
pytest
ruff check src tests scripts experiments
mypy
python -m build
python -m pip check
```

Documentation changes must keep relative links valid, use portable paths, and update model cards or
limitations when behavior or access boundaries change.

## 🔒 Research integrity

- Do not alter frozen scorer definitions, statistical gates, claim labels, or aggregate results to
  make a result appear stronger.
- Report positive and negative findings together. Keep `FINAL PASS` distinct from scientific claim
  support.
- Record seeds, revisions, hashes, dataset roles, and provenance for any new formal experiment.
- Add new results only when code and inputs are traceable and the appropriate tests pass.
- Preserve the immutable research archive; do not force-push or rewrite its history.

## 📦 Data and model artifacts

Do not commit model weights, raw datasets, indexes, caches, secrets, private paths, per-query
outputs, or full job logs. Register an external asset in
[`ARTIFACT_MANIFEST.json`](ARTIFACT_MANIFEST.json) with its source, revision, size, SHA-256,
license, availability, and redistribution boundary.

Never assume that access to a shared file grants permission to publish it. Redistribution of
derived weights or mixed-license datasets requires explicit authorization.

## 🏗️ Module boundaries

Retriever, Selector, and Generator implementations depend only on shared contracts, not on each
other's concrete classes. The Pipeline resolves selected evidence IDs and enforces citation
membership. Changes that cross these boundaries require an architecture test and coordinated API
review.

## 📝 Pull requests

A pull request should state:

1. The problem and scope.
2. The implementation or documentation change.
3. Tests and commands actually run.
4. Scientific claims affected, including negative or null outcomes.
5. External assets, licenses, or API compatibility affected.

Use the repository [issue tracker](https://github.com/M1yanoShiho/IBM_Granite_Project/issues) for
bugs and design discussions. Do not post credentials, private infrastructure details, or restricted
data in an issue.
