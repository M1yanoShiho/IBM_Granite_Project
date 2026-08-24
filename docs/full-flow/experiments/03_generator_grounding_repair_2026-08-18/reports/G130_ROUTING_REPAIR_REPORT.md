# G130 Routing/Attachment Repair

**日期：** 2026-08-18
**状态：** `PASS`

## Runtime Repair

- TRUE hypothesis now uses the final output sentence that reaches the user.
- Trace and G230 routing export now expose declared verification, scan rescue, review flag, routing hypothesis, and final attachment verification.
- Observe-only entity gate warnings remain non-destructive and are not treated as failed attachment.
- TRUE model, TRUE threshold, Retriever, Selector, gold/reference inputs, held-out data, and training remain unchanged.

## G110 Revealed Diagnostic

| Category | Rows |
|---|---:|
| total_true_routing_or_attachment | 44 |
| true_unverified_no_attachment | 33 |
| verified_attachment_with_observe_gate_warning | 11 |

G130 is a deterministic shared runtime repair. It does not rerun G230 metrics and does not claim Generator qualification.
