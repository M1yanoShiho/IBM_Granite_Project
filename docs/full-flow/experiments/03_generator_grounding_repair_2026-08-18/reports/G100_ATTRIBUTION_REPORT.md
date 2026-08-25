# G100 Citation Attribution

**日期：** 2026-08-18
**状态：** `PASS`

## Scope

- Common answered full TopK tasks: `596`
- Claim-level regression rows: `501`
- Rows are automated attribution candidates for G110 audit, not final repair decisions.

## Failure Stage Counts

| Stage | Rows |
|---|---:|
| UNSUPPORTED_DRAFT_CLAIM | 0 |
| DRAFT_CITATION_MISSING_OR_WRONG | 8 |
| SPLITTER_BOUNDARY_OR_REWRITE | 6 |
| TRUE_ROUTING_OR_ATTACHMENT | 268 |
| EVALUATOR_DISAGREEMENT | 219 |
| NO_REGRESSION | 0 |

## Next Step

G110 must audit these rows before activating splitter or routing repair. No training has been started.
