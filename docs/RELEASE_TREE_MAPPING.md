# Release tree archive mapping

This file records the first public-release boundary applied to the complete research snapshot.
Removing a path from `release/dissertation-v1` does not delete it from Git history.

## Recovery reference

- Archive branch: `archive/full-research-history-2026-08-25`
- Immutable tag: `research-archive-2026-08-25`
- Frozen commit: `ea4d617753aff868fff8f846964f7cfb050414bb`

Recover a removed file without changing branches:

```bash
git show research-archive-2026-08-25:path/to/file
```

Restore the full historical tree in a separate worktree:

```bash
git worktree add ../evidence-rag-research-archive research-archive-2026-08-25
```

## Paths removed from the release tree

| Path group | Release decision | Archive reason |
|---|---|---|
| `.aris/` | remove from public tree | Internal planning/tool state |
| `refine-logs/` | remove from public tree | Internal refinement logs |
| `runs/` | external/archive | Generated runtime material and source-parent dump |
| historical top-level `results/` | external/archive | Raw/per-query/intermediate results, many files above 1 MiB |
| `docs/selector/` | archive | Eight Selector development routes, artifacts, snapshots and trackers |
| `docs/full-flow/experiments/01_*` | archive | Historical Selector–Generator bridge route |
| `docs/full-flow/experiments/02_*` | archive | Historical Generator–Selector alignment route |
| `docs/full-flow/experiments/03_*` | consolidate/archive | GR-C training/qualification provenance retained by archive and later model card |
| Experiment 04 plans, trackers, snapshots, readiness manifests and per-query CSV | archive/external | Final report/audit/tables are sufficient for the public tree |
| Experiment 05 plans, trackers and intermediate readiness material | archive | Final report, audit and aggregate tables retained |
| `docs/presentations/` | external/archive | Presentation binaries are not runtime or reproduction dependencies |
| other historical docs and internal handoffs | archive | Preserved for chronology, not part of public navigation |
| `ARCHIVE_INDEX.md` | archive only | The immutable archive already contains the full navigation index |

## Files retained or relocated

| Archive source | Release destination |
|---|---|
| `docs/three-module-runtime-handoff.md` | `docs/three-module-runtime-handoff.md` |
| `docs/full-flow/experiments/04_*/FINAL_REPORT.md` | `docs/research/experiment04.md` |
| `docs/full-flow/experiments/04_*/reports/GOAL3_MAIN_SYSTEM_RESULTS.md` | `docs/research/experiment04-main-results.md` |
| `docs/full-flow/experiments/04_*/reports/GOAL4_MODULE_ABLATION_RESULTS.md` | `docs/research/experiment04-ablation-results.md` |
| `docs/full-flow/experiments/05_*/results/final/FINAL_REPORT.md` | `docs/research/experiment05.md` |
| `docs/full-flow/experiments/05_*/findings.md` | `docs/research/experiment05-findings.md` |
| Experiment 04 final aggregate files | `results/experiment04/` |
| Experiment 05 final aggregate files and audit | `results/experiment05/` |

The relocation preserves file bytes. Scientific values, claim labels, scorer definitions,
checkpoint hashes and evaluation configurations are not changed in this cleanup goal.

## Deferred dependency boundary

`src/`, `tests/`, final Experiment 04/05 scripts and Generator/Selector training inputs are not
removed merely because they are old or numerous. Their final boundary is decided only after import,
test and reproduction-entry analysis in Goals 3 and 4.

At the G2 acceptance gate, the tracked release tree contains 682 files and 5,684,279 bytes. There
are no tracked files above 1 MiB. The largest remaining groups are intentionally deferred code
surfaces: 146 source files, 150 scripts, 231 tests, and 112 configurations. Goals 3 and 4 reduce
those groups only after final runtime and reproduction dependency tests identify the required
closure.

The G2 audit also verified:

- all 896 removed or relocated source paths can be read from the immutable archive tag;
- all 25 public report/result relocations are byte-identical to their archive sources;
- model weights, caches, indexes, per-query outputs, raw generations and logs are absent;
- account names and personal absolute HPC paths are absent from the public tracked tree;
- `src/`, frozen Experiment 04/05 Python entry points, metrics, claim labels and checkpoint hashes
  were not changed by the file-boundary cleanup.

## Final-tree note

This mapping is an execution record. Before the dissertation release is integrated into `main`, its
essential recovery instructions may be condensed into the public models/data and reproduction
documentation. The complete copy remains permanently available from the archive tag.
