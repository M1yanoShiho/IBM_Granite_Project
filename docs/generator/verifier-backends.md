# Verifier (B1) backends

Which NLI/grounding model the Generator's verification pass runs, and the
consequences that follow from the G1 six-arm triage (job 18200000,
`docs/generator/verifier-triage-results-hpc.md`, decision in the **G2** ledger
entry of `docs/hpc-run-log.md`).

## Backends

All three satisfy the `NLIModel` protocol (`classify(premise, hypothesis) -> NLILabel`)
and load weights lazily on first `classify`, so constructing one is free in tests
and CLI paths. Select with the `NLI_BACKEND` env var (`true` | `minicheck` |
`deberta`) or `build_nli_model(name)`; the default is **`true`**.

| backend | class | model | contradiction? | where |
|---|---|---|---|---|
| **TRUE** (default) | `TrueNLIModel` | `google/t5_xxl_true_nli_mixture`, 11B, ~21GB bf16, GPU-only | no (binary) | production main path |
| MiniCheck | `MiniCheckNLIModel` | `lytang/MiniCheck-Flan-T5-Large`, grounding-specific | no (binary) | CPU fallback |
| DeBERTa | `DebertaNLIModel` | `cross-encoder/nli-deberta-v3-*` | yes (3-way) | parity with earlier calibration |

TRUE is the default because it won the G1 triage on both axes that matter — ASQA
entailment recall **0.747** and derived citation precision **0.966**, versus
MiniCheck's 0.620 / 0.886 — and its probability sweep is usable
(recall 0.593–0.880 against FP 0.000–0.053) where Granite-as-judge degenerated to
near 0/1. MiniCheck stays selectable so verification can run without a GPU.

## Operating threshold

Both binary backends turn a support probability into a label with one threshold,
`DEFAULT_BINARY_ENTAIL_THRESHOLD = 0.50`, read off the G1 TRUE P(entail) sweep:
at 0.50, TRUE holds recall **0.747** at a hard-neutral false-positive rate of
**0.007** — the max-recall point that still keeps FP ≤ 0.007, i.e. the
derived-citation-precision 0.966 operating point the G2 decision was made on.
The scoring math in `TrueNLIModel`/`MiniCheckNLIModel` is ported **verbatim** from
the corresponding arms of `scripts/verifier_triage.py`; any drift there silently
invalidates the G1 measurement.

## `ClaimVerification.contradicted` under a binary backend — not computed

TRUE and MiniCheck are **binary**: they emit `entailment` or `neutral` only, never
`contradiction` (see `_binary_entailment_label`). Under either of them,
`ClaimVerification.contradicted` is therefore **not computed** — it stays `False`
for lack of a contradiction judgement, *not* because a contradiction was checked
and ruled out. This is the annotation the team agreed to:

- Any downstream consumer of `contradicted` must treat "the default (TRUE) backend
  never sets it" as the normal state, and must not read a `False` as "no
  contradiction found".
- The G1 grid measured the contradiction signal explicitly: on ASQA hard neutrals,
  `contradicted` was populated only by Granite-as-judge (3b 0.020, 8b 0.087) and by
  the 3-way DeBERTa arms — and the DeBERTa arms fired it on 28% (base) / 44.7%
  (large) of merely-unrelated passages, which is why the 3-way flag was judged
  unusable as a reported result. Adopting a binary main-path backend removes the
  signal cleanly rather than degrading it.
- Only `DebertaNLIModel` (`produces_contradiction = True`) genuinely computes it.
  Code that needs a live contradiction judgement must select that backend
  explicitly and accept its false-positive rate.
