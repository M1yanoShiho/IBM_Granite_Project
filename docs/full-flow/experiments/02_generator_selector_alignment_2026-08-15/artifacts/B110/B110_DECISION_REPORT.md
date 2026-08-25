# B110 Bottleneck Decision

**Status:** `COMPLETE`

## Decision

Primary route: **Generator evidence utilization and context robustness**.

- Retriever: 12/739 (1.62%) decision-dev queries have no official relevant document in TopK10; record separately.
- Generator support-only: 68/218 (31.19%) fail even though the normalized reference string is visible in every O context.
- Generator robustness: among 150 O-correct queries, 23 become wrong under benign or harmful matched noise.
- Legacy Selector: 7 K-right/S-wrong changed queries; every removed item in those regressions is labelled harmful.
- Splitter: 12/68 O failures are zero-claim outcomes; this is not the majority bottleneck.

## Route Gates

- G200/G220 Generator draft-path work: unblocked.
- G210 splitter fallback as the primary branch: not activated.
- S300 utility-Selector training: remains blocked until the Generator gate passes.
- Legacy Selector remains an evidence-risk baseline, not an answer-utility solution.

This decision routes responsibility only. It does not claim that a new method works.
