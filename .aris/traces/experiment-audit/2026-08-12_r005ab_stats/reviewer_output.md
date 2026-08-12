# Statistical-design reviewer output

**Reviewer task:** `/root/amendment_stats_audit`  
**Mode:** independent read-only specialist audit  
**Project files edited:** no

## Verdict

The four-role design is sound after the following requirements are incorporated. The primary endpoint should be one composite success value per component-representative query, not candidate rows treated as independent.

## Mandatory reveal rules

1. All V0/V1/V2 code, initialization, objective, batch schedule, threshold algorithm, sample manifest and gate evaluator must be frozen in one clean commit before the first A-screen read.
2. All three A-fit checkpoints and thresholds must be frozen first. Reveal A-screen only in `V0 → V1 → V2` order; the first PASS becomes the unique variant, and later variants are not revealed.
3. Both B-fit checkpoints, both fit-only thresholds and all execution/hash evidence must be frozen before the first B-confirm read.
4. One command must score both seeds on B-confirm. Every seed × dataset cell must pass. Do not select a seed, average seeds, or pool to n=256.
5. Any B-confirm failure stops this amendment. Trying another variant requires a new amendment and a new untouched confirmation set.

## Recommended endpoints

- `NIAH_success(q)`: canonical clean/cf four classifications, three strict directions and all TopK10 active required/protect-positive evidence are all correct.
- `2Wiki_success(q)`: TopK10 contains at least one official support and every TopK10 official support is protected.

Constituents should be reported on the same query denominator; candidate-level metrics are diagnostic only.

## Exact pass counts

For one-sided exact 95% lower bound `Beta^-1(0.05;k,n-k+1)` and observed rate at least 0.95:

- fit n=64: 61/64 when no confidence claim is made; 62/64 would be needed if the bound were required;
- screen n=96: 92/96, lower `0.907188`;
- confirm n=128: 122/128, lower `0.909583`.

The four B cells form an all-must-pass gate. Each bound is marginal, not a simultaneous four-bound 95% statement. The target population is only the frozen eligible fresh train-fit components, not all queries.

Sanity thresholds and checkpoints may not be reused by R006A/R007. PASS only unlocks R006A; it does not establish an improvement over TopK10.

## Issues

- P0 before revision: confirm one-shot and all-pre-freeze rules were mandatory. They are now incorporated.
- P1: fix threshold enumeration/tie-breaks, endpoint candidate domains, 2Wiki non-vacuous eligibility, and precise isolation units before execution. These are now incorporated in the plan.
- P2: the gate is intentionally conservative and may stop even near a true 95% rate; do not relax it after seeing results.
