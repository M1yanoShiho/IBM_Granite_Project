# Release Validation Report

## Candidate

- Repository: `M1yanoShiho/IBM_Granite_Project`
- Branch: `release/dissertation-v1`
- Full clean-clone basis commit: `aeb592663600a0ffe6cf1dd28c5c33ac947842c7`
- Validation date: 2026-08-25
- Overall status: **PASS — fresh clean-clone, GitHub CI, and authorized HPC real-model smoke all passed**

This status describes technical release validation only. It does not change the negative or mixed
scientific outcomes reported in `docs/results.md` and `docs/limitations.md`.

## Summary

| Gate | Evidence | Status |
|---|---|---|
| Fresh remote clone | Clone resolved exactly to the candidate commit with a clean worktree | PASS |
| README installation | New Python 3.11 virtual environment installed `.[dev,api,data-prep]` | PASS |
| Full test suite | 1,659 passed, 20 skipped | PASS |
| Lint and types | Ruff passed; strict mypy passed for 154 source files | PASS |
| Package build | Version 1.0.0 sdist and wheel built in isolation | PASS |
| CPU three-module smoke | Model-free JSON trace parsed; Selector removed misleading evidence | PASS |
| Dependency health | `pip check` reported no broken requirements | PASS |
| GitHub Actions | Node 24 workflow run [32812730363](https://github.com/M1yanoShiho/IBM_Granite_Project/actions/runs/32812730363) passed on the exact candidate commit | PASS |
| Public table rebuild | Six Experiment 04 and four Experiment 05 outputs matched byte for byte | PASS |
| Manifest and metadata | Artifact manifest and CFF parsed; MIT license packaged once | PASS |
| Repository safety | No personal account, obvious secret, model-weight, or non-exempt large-file finding | PASS |
| Authorized HPC smoke | Job `18709546` loaded the pinned real models and restricted seed-13 assets; all seven live hashes and trace invariants passed | PASS |

## Clean-clone procedure

The candidate was cloned from the remote branch into a new temporary directory. No existing
project virtual environment was reused. The documented setup path was then executed with Python
3.11:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,api,data-prep]'
pytest
ruff check src tests scripts experiments examples
mypy src/evidence_rag tests/typecheck.py
python -m build
python -m pip check
evidence-rag-smoke --json
```

The smoke output contained ten Retriever candidates, nine selected evidence items, no misleading
fixture after selection, and a Generator citation that was a subset of the selected evidence.

## Result integrity

The public table builders ran only from the checked-in compact aggregate inputs. The following ten
outputs matched their frozen public counterparts byte for byte:

- Experiment 04: `table1.json`, `table2.json`, `summary_metrics.csv`, `table1.tex`, `table2.tex`,
  and `final_tables.md`.
- Experiment 05: `table1.csv`, `table2.csv`, `tables.tex`, and the frozen final report.

The full test suite also exercised the registered claim labels, result audits, public Selector
evidence/answer gates, artifact identities, and document-link contracts.

## Repository and archive integrity

- Tracked release tree: 472 files, 4,350,918 bytes.
- Largest tracked file: 143,646 bytes.
- No tracked file exceeded the 1 MiB release gate.
- No tracked model-weight extension was present.
- The remote release branch resolved to the candidate commit.
- `archive/full-research-history-2026-08-25` and the peeled
  `research-archive-2026-08-25` tag both still resolved to
  `ea4d617753aff868fff8f846964f7cfb050414bb`.

## Issues found and corrected during validation

1. The first fresh clone exposed a stale README sentence that still described the license as
   pending. The homepage now states the user-selected MIT license.
2. The second fresh clone exposed an incomplete CI/development dependency closure: the full
   Experiment 05 tests require NumPy and PyArrow. The development lock, CI install metadata, and
   public setup/reproduction commands now include the `data-prep` extra. A new clean environment
   and GitHub Actions both passed after this correction.
3. A successful CI run warned that the older checkout/setup actions used the retired Node 20
   runtime. Their official current v7 releases use Node 24, so the workflow was upgraded. The
   subsequent run passed every step with zero annotations.
4. The first authorized CPU real-model run exposed that the Selector loader passed the portable
   config value `device = "auto"` directly to PyTorch. A regression test was added and the loader
   now resolves `auto` to CUDA when available and CPU otherwise, while preserving explicit device
   values. The final clean clone, CI, and HPC run all passed after this correction.

## Authorized HPC real-model smoke

The release intentionally contains no trained weights or model cache, so the final real-model
check ran against the restricted assets already retained on the authorized university cluster.
The validator recomputed the seven required SHA-256 identities before execution: the trained
seed-13 Selector checkpoint, GR-C adapter weights and config, and the four pinned upstream model
configs all matched the release manifests.

The final scheduler job `18709546` ran candidate
`aeb592663600a0ffe6cf1dd28c5c33ac947842c7` to completion in 4 minutes 32 seconds with exit code
zero. It used the committed three-module smoke manifest and constructed the same
config/dataset/corpus/composition path as the public runtime loader. The public API wrapper itself
was separately exercised in the fresh `.[api]` clean-clone environment.

The redacted trace summary was:

- Retriever: `HybridRetriever`; 10 candidates.
- Selector: trained `NliRiskControlledSelector`; 10 selected and zero deleted for this ordinary
  query. This is a model decision, not a Top-K fallback: the trained class and checkpoint hash were
  both verified.
- Generator: grounded `VerifyAnnotateGenerator`; non-empty answer with one citation.
- Selected evidence was a subset of candidates, and citations were a subset of selected evidence.
- All execution was offline; no private filesystem path or restricted weight was written to Git.

The university A100 allocated for the first attempt could not create a CUDA tensor, and the same
failure reproduced with the cluster's official PyTorch probe. The completed fallback therefore ran
on a 32-core, 160 GiB scheduler node with the identical real weights and runtime config. This node
condition is recorded as infrastructure evidence rather than misreported as a model failure.
