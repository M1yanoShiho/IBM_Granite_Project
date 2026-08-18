# G010 Power/Scope Sensitivity

**日期：** 2026-08-18
**状态：** `PASS`

## Scope

- G230 full TopK rows: `739`
- G230 components: `534`
- Design effect: `1.3839`
- Observed power was not computed.

## Mean G230 Discordance MDE

| Family | Metric | Mean discordance | MDE |
|---|---|---:|---:|
| GC | answer_match | 22.64% | 5.77% |
| GC | citation_all_supported | 26.12% | 6.20% |
| GC | correct_and_cited | 25.71% | 6.15% |
| GM | answer_match | 25.30% | 6.10% |
| GM | citation_all_supported | 26.61% | 6.25% |
| GM | correct_and_cited | 27.24% | 6.33% |

## Interpretation

- G010 does not relax any gate or margin.
- A null or wide CI in later stages must be reported as insufficient evidence for small effects, not equivalence.
- Next allowed work is G100 citation attribution; training remains blocked.
