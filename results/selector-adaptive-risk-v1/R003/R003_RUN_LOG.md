# R003 run log

## Run identity

- Run: `R003 — topk-and-count-controls`
- Branch/commit: `refactor/three-module-baseline@c23c4df71ba61c95b1acb72411684e6a62ff3e77`
- Server worktree at execution: clean
- Device: CPU-only metric/protocol generation; GPU and peak VRAM are not applicable
- Exact wall time: not captured by the R003 CLI, so no duration is invented here
- Evidence seal time: `2026-08-11T23:12:37Z`

## Formal generation

1. Generated TopK10/9/8/7 control artifacts for `niah-train`: PASS.
2. Generated TopK10/9/8/7 control artifacts for `niah-dev`: PASS.
3. Generated TopK10/9/8/7 control artifacts for `2wiki-train`: PASS.
4. Generated TopK10/9/8/7 control artifacts for `2wiki-dev`: PASS.
5. Generated the input-free 100-repeat count-matched protocol: PASS.

Formal generation total: `5/5 PASS`.

Every TopK job required the matching frozen Selector-v2 pool manifest and R002 component bundle. NIAH additionally required its frozen assignment subset. The count-matched job generated only the deterministic protocol and seeds; it did not generate experimental outcomes.

## Independent reproduction checks

- Re-ran all four TopK jobs and the protocol job with `--verify-only` against the same frozen inputs: `5/5 PASS`.
- The temporary generation root contained exactly the 14 expected raw files and no extras, then was atomically moved to the formal R003 path.
- Copied the formal artifacts to the local repository and compared every raw file with the server: `14/14 SHA-256 MATCH`.
- Directly re-hashed every upstream input named by the four TopK manifests on the server: `54/54 input pins MATCH`.
- Independently parsed and recomputed all `30,008` trace rows: complete K quartets, strict prefixes, suffix drops, selected-plus-dropped counts, and R002 role/component/chain bindings had zero anomalies; 2Wiki emitted zero harmful fields.
- Local [`CHECKSUMS.sha256`](CHECKSUMS.sha256) covers those 14 raw files only. Summary/config/manifest/log files were written after raw-artifact verification and are excluded to avoid recursive checksums.

## Tests

- Server relevant tests: `60 passed`.
- Local new targeted tests: `10 passed`.
- Local full repository: `1268 passed`.
- Ruff on `src` and `tests`: `PASS`.
- mypy on `src` and `tests/typecheck.py`: `PASS (112 source files)`.

## Required absent/deferred outputs

| Output | Status | Reason |
|---|---|---|
| Learned candidate scores | `NOT_APPLICABLE` | R003 does not train or run a scorer. |
| Selector decision trace | `NOT_APPLICABLE` | TopK controls are literal prefixes, not Selector decisions. |
| Selector selected sets | `NOT_APPLICABLE` | No candidate Selector exists in R003. |
| CRC calibration artifact | `NOT_APPLICABLE` | CRC is applied only after a trained scorer and a frozen policy family. |
| Count-matched random/bottom-rank results | `DEFERRED` | They require the real per-query deletion counts from a future Selector trace. |

The generated `topk_controls.jsonl` files are baseline metric traces. They must not be relabeled as learned scores or Selector decision traces.

## Decision

`R003 = COMPLETE / BASELINE-PROTOCOL PASS`; `Gate 1 = PASS`; next run is `R004`.

This decision certifies that the baseline and comparison protocol are complete and reproducible. It does not certify that a Selector, threshold, or deletion policy is safe or useful.
