# Changelog

All release-facing changes to Evidence RAG are recorded here. Development-stage history remains
available through the repository's archive branch and immutable research tag.

## 1.0.0-dissertation — 2026-08-25

### Added

- Final Hybrid Retriever → trained NLI Selector → grounded GR-C Generator runtime.
- Versioned `GET /health` and `POST /v1/query` HTTP API with lazy model loading.
- Offline CPU smoke using final module classes and deterministic doubles.
- Frozen runtime/model configuration and fail-closed external-asset validation.
- Public Experiment 04/05 table builders, aggregate results, audits, and claim labels.
- Machine-readable artifact manifest with revisions, sizes, SHA-256, licenses, and availability.
- Public setup, architecture, reproduction, result, limitation, model-card, citation, and
  contribution documentation.

### Changed

- Replaced development-stage TopK/Corroboration documentation with the final trained NLI Selector
  method and explicit baseline/ablation boundaries.
- Reduced the release tree to the tested final runtime, research reproduction closure, aggregate
  evidence, and public documentation while retaining full history in the archive.

### Known limitations

- Derived Selector and GR-C weights are frozen but not publicly redistributed.
- Selector evidence filtering passed its dedicated stress gate, but its blind answer gate failed.
- Experiment 04 whole-system superiority and Experiment 05 Claims A/B were not supported.
- `FINAL PASS` in experiment audits means protocol execution, not scientific superiority.
