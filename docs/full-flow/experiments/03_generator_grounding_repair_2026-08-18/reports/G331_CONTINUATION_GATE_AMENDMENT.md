# G331 Continuation Gate Amendment

**Stage:** G331  
**Date:** 2026-08-19  
**Status:** `COMPLETE / PASS`  
**Scope:** planning and decision-rule amendment only  
**Next stage:** G400 locked NIAH qualification

## Why This Amendment Was Needed

The previous wording could make a numeric threshold sound like an absolute judgment on the whole route. That is too strict for this project.

A number such as 76/100 or 97/100 should answer a narrower question: can this exact artifact be cleanly frozen or used for a strong claim? It should not automatically answer the broader question: is the overall Generator/Selector route meaningful?

The corrected interpretation is:

- missing a clean-freeze or strong-claim threshold limits what we can claim;
- it does not by itself prove the route is useless;
- positive, localized, reproducible signal should continue into controlled validation;
- technical invalidity, forbidden-data leakage, or serious safety/citation regression still stops the affected route.

## Revised Decision Layers

Future stages must report four separate outcomes.

1. **Technical validity**

   The run finished, rows and hashes match, runtime did not read gold/reference, and no held-out or sealed data was touched.

2. **Responsibility usability**

   The component is usable for the next controlled stage if the main responsibility metrics do not show practically unacceptable regression and at least one main or mechanism metric gives an interpretable positive signal.

3. **Strong claim**

   Stronger language is allowed only when the frozen statistical rule supports it. Failing this layer downgrades wording; it does not automatically stop controlled work.

4. **Controlled continuation**

   If strong freeze/strong claim is not met, but the signal is positive, failures are local and explainable, and no safety/leakage tripwire fires, the next planned diagnostic or qualification stage can continue with restricted claims.

## What Remains Hard

The following still block the affected route:

- runtime reads prohibited held-out, sealed, gold, provenance, or reference data;
- hashes, row counts, model identity, or command records cannot be reproduced;
- a serious unsupported-answer, citation, or safety tripwire fires;
- utility labels cannot form a usable training target;
- the complete frozen system later fails its responsibility gate.

## What Changed For G400/G410/G420

G400 is not a standalone judgment that the entire route succeeds or fails. It is part of the Generator qualification bundle.

If G400 misses a numeric responsibility margin but shows clear positive signal and no tripwire, the correct result is a restricted positive/continuation report, then G410/G420 can still complete the planned combined assessment.

If the new Generator is not qualified as GQ, G430 can still freeze `GQ=G0` and move to Selector utility work. In that case, we do not claim Generator repair success.

## What Did Not Change

- No held-out run is authorized.
- No final test set is used for tuning.
- No seed, threshold, dataset boundary, or metric is changed after seeing results to force a pass.
- `GR-C` remains the only frozen Generator recipe for G330/G400.
- Retriever remains frozen.
- Runtime Selector and Generator still cannot read gold/reference.

## User-Facing Interpretation

The short version is:

> A threshold decides how strong the claim can be. It does not automatically decide whether the research direction has value.

So if a result is 76 instead of 100, or 97 instead of 100, the right question is:

- Is the run technically valid?
- Is the positive signal real enough to justify the next controlled check?
- Are the failures concentrated and explainable?
- Did any safety or leakage stop condition fire?

If the answers support continuation, we continue. If they show the signal is not useful or the risks are unacceptable, we stop or fall back.

## Files Updated

- `PLAN.md`
- `README.md`
- `TRACKER.md`
- `G331_CONTINUATION_GATE_AMENDMENT.md`
- `artifacts/G331/G331_CONTINUATION_GATE_MANIFEST.json`
