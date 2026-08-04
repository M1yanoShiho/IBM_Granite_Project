
## Paired comparisons (ALCE sentence-level, MiniCheck, randomization + bootstrap CI)

| comparison | metric | delta | p | 95% CI | n |
|---|---|---|---|---|---|
| **verify-annotate vs verify-only** | cite precision | **+0.070** | **0.0006** | [+0.029, +0.111] | 248 |
| verify-annotate vs verify-only | cite recall | **−0.065** | 0.0029 | [−0.105, −0.023] | 248 |
| verify-annotate vs verify-only | correctness | **+0.010** | 0.0019 | [+0.004, +0.017] | 387 |
| verify-annotate vs verify-only | coverage | **0.000** | 1.00 | [0.000, 0.000] | 387 |
| verify-annotate vs baseline | cite precision | +0.142 | <0.0001 | [+0.082, +0.203] | 245 |
| verify-annotate vs baseline | cite recall | +0.067 | 0.039 | [+0.003, +0.132] | 245 |
| verify-annotate vs baseline | correctness | −0.053 | <0.0001 | [−0.079, −0.027] | 387 |
| verify-only vs baseline | cite precision | +0.073 | 0.019 | [+0.012, +0.136] | 245 |

## Against the pre-registration

The registered criterion (commit `730e14f`, written before any arm ran):

> The redesign fails if citation precision on cited claims regresses to baseline
> level ... or if coverage does not improve over `verify-only`.

- **Precision criterion: passed, and better than expected.** Precision was
  expected merely to *hold* near verify-only; it rose to **0.830**, above
  verify-only's 0.760 (+0.070, p=0.0006) and far above baseline's 0.613. It did
  not regress to baseline, so annotation is not letting unverified content
  through as cited.
- **Coverage criterion: triggered.** Coverage is **identical** to verify-only
  (delta exactly 0.000, p=1.00).
- Citation recall fell as designed (−0.065), the intended and visible cost.
- Correctness rose over verify-only by **+0.010 (p=0.0019)** — statistically
  significant but small, and still 0.053 below baseline.

**The coverage criterion was written against a `verify-only` that no longer
exists.** It cites 0.552, the all-or-nothing figure; by the time this ran,
partial answering had already been landed and verify-only scores **0.641**. Both
arms now abstain on the *same* condition — zero verified claims — so annotation
**cannot** move coverage by construction. Coverage did improve on the published
method (0.552 → 0.641, +0.089), but that is attributable to partial answering,
not to annotation. Both readings are recorded; neither is discarded.

## Two limitations that bound these numbers

**1. The precision comparison is confounded by citation convention.**
`verify-annotate` records an exact sentence → verified-citation mapping (mean
1.13 citations per answer); `verify-only` has no such record, so it falls back to
the flat-list convention where every sentence carries all of the answer's
citations (mean 1.92). Under ALCE, extra citations on a sentence trigger the
redundancy ablation and cost precision. **Part of the +0.070 is therefore a
scoring artifact, not a quality difference** — the same class of confound that
inflated the answer-level numbers in G3. Settling it needs `verify-only` to
record its per-claim citation mapping too, and a re-run.

**2. Every number here was measured on a pipeline whose only content-destroying
path is 80% wrong.** The entity-conflict audit
(`docs/generator/g5-entity-conflict-audit.md`) puts the false-veto rate at 0.700
with a further 0.100 that should have been annotated. Roughly 58 of the 72
dropped claims should have survived, most of them as *cited* sentences.
