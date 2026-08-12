# Model/loss implementation reviewer output

**Reviewer task:** `/root/amendment_code_audit`  
**Mode:** independent read-only specialist audit  
**Project files edited:** no

## Verdict

The design is implementable after two mandatory changes: use pair-preserving batches for every variant, and use a variant-aware model builder/checkpoint verifier while preserving old R005 behavior.

## Verified snapshot facts

- Snapshot architecture: `DebertaV2ForSequenceClassification`.
- Labels: 0 contradiction, 1 entailment, 2 neutral.
- `AutoModel` loads `DebertaV2Model` and ignores pretrained pooler/classifier parameters.
- `AutoModelForSequenceClassification` loads with zero missing or unexpected keys.
- The viable wrapper registers shared DeBERTa, pretrained ContextPooler, one shared classification dropout invocation and two independent deep-copied 3-logit classifiers.
- The two classifier copies begin numerically equal but share no parameter storage; strict state reload is feasible.

## Exact one-vs-rest transform

```text
protect_logit = z_entailment - logsumexp(z_contradiction, z_neutral)
harm_logit    = z_contradiction - logsumexp(z_entailment, z_neutral)
```

The sigmoids equal the original NLI entailment and contradiction softmax probabilities at initialization; server numerical error was about `1e-11`. A single linear “target row minus mean of rest” is not exact and should not be used.

## Pairwise objective

```text
L_pair = mean_q 0.5 * [
  softplus(-(protect_clean - protect_cf)) +
  softplus(-(harm_cf - harm_clean))
]
L_total = L_masked_BCE + 0.5 * L_pair
```

At zero margins raw `L_pair=log(2)`. Gradients raise clean protect and cf harm while lowering cf protect and clean harm. Use scalar logits, average by unique strict query pair, and fail on missing/duplicate/ambiguous pairs.

## Mandatory fair batching

The current single-candidate shuffle can split a pair. V0/V1/V2 must share one frozen pair-preserving batch manifest: each NIAH pair stays in one microbatch; candidate order, steps and accumulation are identical; V0/V1 use pair weight 0 and V2 uses 0.5; V2 reuses the same forward and does not get extra compute/dropout views. 2Wiki remains under the same deterministic schedule with graph-connected exact-zero pair contribution.

## Compatibility and tests

- New architecture needs a versioned builder and upgraded fingerprint binding pooler, dropout semantics, label map, two 3-logit heads, objective and batch SHA.
- Old loader/fingerprint/verifier must remain unchanged; V0↔V1 cross-load must fail explicitly.
- Required tests include exact initialization equivalence, independent storage, one dropout call, zero-margin loss/gradient directions, pair batch completeness, masked 2Wiki gradient, strict checkpoint roundtrip, mutation-sensitive fingerprints and a full old-R005 verify-only regression.

## Issues

- P0 before revision: old batching split pairs; old verifier could not construct/load the new architecture. The plan now requires both fixes.
- P1: bind architecture/objective/batch in artifacts, preserve old R005 semantics, share dropout once and save decomposed loss traces. The plan incorporates these requirements.
- P2: added parameters are negligible relative to the existing checkpoint and A4000 capacity.
