# Release Validation Report

## Candidate

- Repository: `M1yanoShiho/IBM_Granite_Project`
- Branch: `release/dissertation-v1`
- Full clean-clone basis commit: `044978fe69aa9db9a66785ce8ea1cce648591a80`
- Validation date: 2026-08-25
- Overall status: **PARTIAL — local clean-clone and GitHub CI pass; authorized HPC real-model smoke pending cluster connectivity**

This status describes technical release validation only. It does not change the negative or mixed
scientific outcomes reported in `docs/results.md` and `docs/limitations.md`.

## Summary

| Gate | Evidence | Status |
|---|---|---|
| Fresh remote clone | Clone resolved exactly to the candidate commit with a clean worktree | PASS |
| README installation | New Python 3.11 virtual environment installed `.[dev,api,data-prep]` | PASS |
| Full test suite | 1,656 passed, 20 skipped | PASS |
| Lint and types | Ruff passed; strict mypy passed for 154 source files | PASS |
| Package build | Version 1.0.0 sdist and wheel built in isolation | PASS |
| CPU three-module smoke | Model-free JSON trace parsed; Selector removed misleading evidence | PASS |
| Dependency health | `pip check` reported no broken requirements | PASS |
| GitHub Actions | Node 24 workflow run [32808187612](https://github.com/M1yanoShiho/IBM_Granite_Project/actions/runs/32808187612) completed successfully with zero annotations | PASS |
| Public table rebuild | Six Experiment 04 and four Experiment 05 outputs matched byte for byte | PASS |
| Manifest and metadata | Artifact manifest and CFF parsed; MIT license packaged once | PASS |
| Repository safety | No personal account, obvious secret, model-weight, or non-exempt large-file finding | PASS |
| Authorized HPC smoke | Login endpoint was not reachable from the validation network | PENDING |

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

## Pending authorized HPC smoke

The release contains no trained checkpoint, model cache, dataset manifest, or index. Real-model
validation must therefore run on an authorized backend host. During this validation session the
configured University VPN service was disconnected, so the cluster login endpoint was network
unreachable; an unconfigured guessed alternate endpoint did not resolve. No HPC pass is claimed.

When cluster connectivity is available, the validator must:

1. clone or fast-forward the candidate commit on the cluster;
2. verify the Selector checkpoint and GR-C adapter bytes and SHA-256 values against
   `ARTIFACT_MANIFEST.json`;
3. set the six documented `EVIDENCE_RAG_*` paths without recording their private values;
4. run one real query through Hybrid Retriever → trained NLI Selector → grounded GR-C Generator;
5. confirm every Generator citation belongs to the Selector output; and
6. append the scheduler/job identity, candidate commit, asset verification result, and redacted
   smoke summary to this report.

G7 remains incomplete until this final row is changed from `PENDING` to `PASS` using fresh HPC
evidence. Any code, configuration, lock, or result change after that smoke requires the affected
validation gates to be rerun.
