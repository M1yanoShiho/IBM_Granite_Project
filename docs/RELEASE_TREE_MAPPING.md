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

## Final-tree note

This mapping is an execution record. Before the dissertation release is integrated into `main`, its
essential recovery instructions may be condensed into the public models/data and reproduction
documentation. The complete copy remains permanently available from the archive tag.
