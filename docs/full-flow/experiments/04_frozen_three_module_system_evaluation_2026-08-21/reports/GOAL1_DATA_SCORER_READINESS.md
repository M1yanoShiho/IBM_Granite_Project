# Experiment 04 Goal 1 — Data and scorer readiness

**Date:** 2026-08-22  
**Final status:** `PASS / STOPPED AT GOAL 1`  
**Held-out answers generated or scored:** `NO`

## Decision

Goal 1 passes. The three frozen ordered query sets are intact, every runtime
record owns its candidate pool, runtime inputs contain no gold fields, the
scorer-only sidecars are physically separate, and all five metrics plus failure
denominator rules pass known-answer tests on a revealed synthetic fixture.

This report is the mandatory stop point. No baseline was connected, no 10-arm
smoke was started, no Retriever/Selector/Generator/MiniCheck model was run on
formal held-out data, and Goal 2 was not started.

## Frozen data audit

The committed machine-readable source is
[`artifacts/goal1_data_manifest.json`](../artifacts/goal1_data_manifest.json).
Only counts, schemas, ordered-ID hashes, file hashes, and boolean audit results
were emitted; no question, answer, support label, or held-out passage was logged.

| Dataset | Expected / observed | Ordered ID SHA-256 | Relation to pre-registration | Result |
|---|---:|---|---|---|
| HotpotQA | 400 / 400 | `1c1b822c8942b3e14979de187b63d06f5b64ebcdd2303cf5b6b56c9bea92755d` | Direct match | PASS |
| MuSiQue answerable | 400 / 400 | `37626d4b1890bdc239d0904c9f406a06023f73382ccd9aabfa44bf65189761fb` | Ordered `#ans` projection of the frozen 400-ID / 800-record paired manifest; the parent paired hash also revalidates | PASS |
| RGB noise | 300 / 300 | `25ed22988cb9684a8de9fc74c603e79ef44792f0e3bfd4c3b9e74c2e2a301c87` | Direct match | PASS |

The MuSiQue hash is a deterministic Goal 1 projection, not a re-draw: the old
manifest's 800 ordered records first revalidate against their pre-registered
hash, contain exactly 400 complete `#ans/#unans` pairs, and are then filtered in
place to the 400 `#ans` records required by the v4 plan.

## Runtime / scorer isolation

Formal sealed files were materialized under the repository-ignored root
`artifacts/experiment04_goal1/sealed/`:

- `runtime/<dataset>.jsonl` uses schema `experiment04.runtime.v1` and contains
  only `schema_version`, `dataset`, `query_id`, `question`, and nested
  `candidates`. Candidate fields are `source_id`, `title`, `text`, and unlabeled
  `units`. Recursive forbidden-gold field count is zero.
- `scorer_only/<dataset>.jsonl` uses schema `experiment04.scorer.v1` and contains
  gold answer aliases, minimal support units, their integrity hashes, and an
  opaque `component_id`.
- Runtime and scorer-only files have distinct physical parent directories and
  identical ordered query IDs. Their only alignment key is `query_id`.
- The system-facing reader accepts one runtime file path only and imports no
  sidecar reader. The scorer sidecar is therefore absent from the runtime entry
  contract.
- Candidate pools are nested directly inside each query record. Local
  `source_id`/`unit_id` values are resolved only within that record; no global
  corpus or cross-query candidate lookup exists.

Minimal official support units are HotpotQA supporting sentences, MuSiQue
supporting paragraphs, and RGB positive documents. Every sidecar support unit
was programmatically resolved to a unit in the same query's runtime pool, with
matching source ID, text, and SHA-256.

## Five-metric validation

The machine-readable validation is
[`artifacts/goal1_scorer_validation.json`](../artifacts/goal1_scorer_validation.json).
It uses eight revealed synthetic cases and pre-supplied synthetic MiniCheck
precision/recall values; no MiniCheck model was loaded.

| Metric | Frozen implementation checked | Known-answer result |
|---|---|---|
| Ret. | Per-query official support-unit recall in Retriever final Top10; macro mean | PASS |
| Sel. | Per-query retained support-unit recall using the same full gold denominator as Ret.; macro mean | PASS |
| Ans. | HotpotQA/MuSiQue token F1; RGB normalized exact accuracy | PASS |
| Cit. | Per-query harmonic F1 of MiniCheck citation precision/recall; macro mean | PASS |
| RAR | Normalized gold-alias exact match + legal citation index + MiniCheck precision = recall = 1 | PASS |

The fixture's expected and observed aggregate vector is identical:
`Ret.=0.8125`, `Sel.=0.75`, `Ans.=0.4375`, `Cit.=1/3`, `RAR=0.25`.
These are synthetic validation values, not held-out results.

## Failure and common-denominator rules

All five fixture denominators remain eight. The following cases remain as
explicit per-query rows with machine-readable reasons:

- empty answer: `Ans./Cit./RAR = 0`;
- illegal citation: `Cit./RAR = 0` while independently computable `Ans.` remains;
- generation failure: `Ans./Cit./RAR = 0`;
- scoring failure: `Ans./Cit./RAR = 0`;
- missing output: all five metrics are `0`;
- duplicate or unexpected query IDs: hard contract error rather than silent
  deletion or denominator drift.

## Automated evidence

- Sealed materialization plus synthetic scorer validation:
  `PYTHONPATH=src ... python scripts/experiment04_goal1.py all` — PASS for all
  three data bundles and the scorer fixture.
- Goal 1 focused tests: `14 passed`.
- Style checks on all Goal 1 implementation and tests: PASS.
- Strict type check on both new source modules: PASS, no issues.
- Repository regression run excluding one unrelated known state-contract test:
  `1767 passed, 20 skipped, 1 deselected`.

The unfiltered repository run has one non-Goal-1 failure:
`tests/scripts/test_full_flow_g000_freeze.py::test_g000_without_server_audit_stops_before_next_stage`.
It expects the separate Experiment 03 README/TRACKER to advertise an active
G000 state. Those files and that route are outside Goal 1 and were not changed
to force a green result. No Goal 1 test failed.

## PASS checklist

| Condition | Status |
|---|---|
| Frozen ID counts are 400 / 400 / 300 | PASS |
| Ordered ID hashes match the direct or deterministic pre-registered source | PASS |
| Runtime contains no gold answer/support/component fields | PASS |
| Runtime and scorer-only sidecar are physically separate | PASS |
| Runtime/sidecar order matches and alignment uses `query_id` | PASS |
| Every candidate/support unit stays within its query pool | PASS |
| Ret./Sel./Ans./Cit./RAR known-answer tests pass | PASS |
| Empty/error/invalid/missing samples remain in the denominator | PASS |
| No held-out model execution, MiniCheck scoring, or content exposure occurred | PASS |
| Goal 2 remained unstarted | PASS |

**Final Goal 1 result: `PASS`. Stop here.**
