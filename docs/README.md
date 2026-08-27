# Evidence RAG Documentation

This is the public documentation index for the final Evidence RAG research release. New users
should begin with setup and architecture; reviewers should continue to reproduction, results, and
limitations.

## Use the system

| Document | Purpose |
|---|---|
| [Setup](setup.md) | Install the package, run the offline CPU smoke, and configure authorized model assets. |
| [Architecture](architecture.md) | Understand the Hybrid Retriever, trained NLI Selector, grounded GR-C Generator, Pipeline, and API boundaries. |
| [Front-end integration](frontend-integration.md) | Use the versioned HTTP request/response contract or model-free mock fixture. |
| [Runtime handoff](three-module-runtime-handoff.md) | Operate the frozen seed-13 backend and its environment contract. |

## Review the research

| Document | Purpose |
|---|---|
| [Reproduction](reproduction.md) | Rebuild public tables, verify frozen claims, and locate restricted raw-recompute inputs. |
| [Models and data](models-and-data.md) | Inspect upstream revisions, hashes, licenses, download rules, and external-asset boundaries. |
| [Results](results.md) | Read supported and unsupported findings together. |
| [Limitations](limitations.md) | Understand generalization, resource, access, evaluation, and deployment constraints. |
| [Selector model card](model-cards/selector.md) | Review the trained Selector's identity, intended use, evidence gate, and failed answer gate. |
| [Generator model card](model-cards/generator.md) | Review the GR-C recipe, adapter provenance, access, evaluation, and known weakness. |

The release [reproducibility map](release/REPRODUCIBILITY_MAP.md) is the claim-to-artifact
authority. The compact results are under [`results/`](../results/), while complete development-time
records remain recoverable from the immutable `research-archive-2026-08-25` Git reference.

## Maintain the release

- [Contributing](../.github/CONTRIBUTING.md) defines testing and research-integrity requirements.
- [Authors](release/AUTHORS.md) explains contributor identities and authorship boundaries.
- [Changelog](release/CHANGELOG.md) records release-facing changes.
- [Release validation](release/RELEASE_VALIDATION_REPORT.md) records the final verification gates.
- [Citation metadata](../CITATION.cff) provides the machine-readable software citation.

Historical cleanup plans are not part of the user documentation. They remain in the research
archive and release work records so the development process can still be audited.
