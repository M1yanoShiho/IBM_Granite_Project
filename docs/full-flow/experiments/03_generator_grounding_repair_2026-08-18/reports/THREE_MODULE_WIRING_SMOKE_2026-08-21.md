# Three-Module Wiring Smoke

**Date:** 2026-08-21
**Status:** `PASS / WIRING ONLY / NO FORMAL EXPERIMENT`

## Connected Route

```text
Hybrid RRF Retriever (StrongBM25 + Granite dense)
-> NLI harm/protect Selector (seed 13, q=.99, cap=2)
-> GQ GR-C Generator (seed 13) + TRUE verifier
-> one answer with verified citation
```

The runnable configuration is `configs/experiments/systemf_three_module_smoke_seed13.toml`.

## Checks Performed

- The Selector checkpoint file matched SHA-256 `86622bd9...72bf`.
- The GR-C adapter weights matched SHA-256 `492d336c...707`.
- The GR-C adapter configuration matched SHA-256 `1918333d...942`.
- The Granite base configuration matched SHA-256 `9a0e589b...25c`.
- The TRUE configuration matched SHA-256 `69f08955...6e9`.
- The real server process instantiated `HybridRetriever`, `NliRiskControlledSelector`, and `VerifyAnnotateGenerator` together.
- The Retriever returned 10 candidates.
- The Selector produced 10 live harm/protect decisions and selected 10 candidates; it deleted none on this fixture.
- The Generator trace was present and returned `IBM acquired Red Hat in 2019.` with one verified evidence citation.

## Boundary

The smoke used one synthetic question and ten fixture documents under `tests/fixtures/three_module_smoke_dataset/`.  It did not read a formal development bundle, did not read held-out data, did not score system quality, and did not retrain or retune any component.  Its only claim is that the three frozen methods can now execute sequentially through one pipeline entry point.
