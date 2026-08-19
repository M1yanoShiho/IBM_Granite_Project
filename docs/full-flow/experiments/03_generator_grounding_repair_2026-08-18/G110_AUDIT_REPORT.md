# G110 Independent Audit

**日期：** 2026-08-18
**状态：** `PASS`

## Double Review

- Primary rows: `120`
- Double-reviewed rows: `24`
- Agreement: `22/24` (91.67%)
- Route decision: `G130_ROUTING_ATTACHMENT_REPAIR_CANDIDATE_REQUIRES_SEPARATE_IMPLEMENTATION`
- Conditional repair activated: `True`

## Final Label Counts

| Label | Rows |
|---|---:|
| DRAFT_CITATION_MISSING_OR_WRONG | 8 |
| EVALUATOR_DISAGREEMENT | 37 |
| SPLITTER_BOUNDARY_OR_REWRITE | 22 |
| TRUE_ROUTING_OR_ATTACHMENT | 44 |
| UNSUPPORTED_DRAFT_CLAIM | 9 |

G110 only chooses the next repair path. Training remains blocked by the later data/materialization/training gates.
