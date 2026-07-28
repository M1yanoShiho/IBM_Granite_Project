# Verifier triage results (Generator Part B, Half 1.6)

Measurement only -- `nli.py`, `attribution.py`, `entity_check.py` and every production config are unchanged; no threshold is committed. Premise = evidence, hypothesis = claim, the direction B1 is built on.

**Hold-out respected:** HotpotQA, RGB and MuSiQue-Full are never loaded. Calibration data only -- ALCE/ASQA, 2WikiMultihopQA, and a synthetic entity-swap slice.

## Shared pair set

Every arm judges exactly these pairs, in this order.

| cell | gold | pairs |
|---|---|---|
| 2wiki-atomic | entailment | 240 |
| 2wiki-atomic-union | entailment | 120 |
| 2wiki-composed | entailment | 120 |
| 2wiki-composed-union | entailment | 60 |
| 2wiki-neutral | neutral | 240 |
| asqa | entailment | 150 |
| asqa | neutral | 150 |
| counterfactual | neutral | 109 |
| counterfactual (reported separately) | not-supported | 109 |

Near-verbatim guard: an ASQA claim is dropped if it shares 8 or more contiguous content tokens with its own gold passage. Median longest shared run in the surviving set: **5 tokens** against a median claim length of ~14 content tokens -- reported because a claim copied out of its premise makes entailment trivially detectable and inflates recall into meaninglessness.

## Arms

- **true** -- google/t5_xxl_true_nli_mixture (TRUE mixture, binary 1/0)
- **granite3b** -- ibm-granite/granite-4.1-3b (LLM-as-a-judge, fixed prompt)
- **granite8b** -- ibm-granite/granite-4.1-8b (LLM-as-a-judge, fixed prompt)

## Main grid

| arm | ASQA entail recall | ASQA hard-neutral -> entail | ASQA hard-neutral -> contra | 2Wiki atomic recall (any single chunk) | 2Wiki composed recall (single chunk) | 2Wiki composed recall (union) | residual multi-hop | union diag. (residual) | 2Wiki distractor -> entail | ms/pair |
|---|---|---|---|---|---|---|---|---|---|---|
| true | 0.747 | 0.007 | 0.000 | 0.875 | 0.467 | 0.683 | 0.125 | 0.000 | 0.000 | 263.9 |
| granite3b | 0.767 | 0.067 | 0.020 | 0.925 | 0.217 | 0.633 | 0.075 | 0.000 | 0.000 | 114.2 |
| granite8b | 0.900 | 0.240 | 0.087 | 0.908 | 0.317 | 0.850 | 0.092 | 0.182 | 0.029 | 256.2 |

`residual multi-hop` = atomic claims (from the gold decomposition) that no single gold chunk entails. `union diag.` = of those, the share the same verifier recovers when both gold chunks are concatenated -- diagnostic only, it separates 'decomposition did not reduce this claim' from 'the verifier cannot see support that is present'.

## Arm C -- threshold sweep on P(entail) (ASQA cell)

### true

| threshold | entail recall | hard-neutral FP | precision proxy |
|---|---|---|---|
| 0.05 | 0.880 | 0.053 | 0.943 |
| 0.10 | 0.840 | 0.033 | 0.962 |
| 0.20 | 0.793 | 0.027 | 0.967 |
| 0.30 | 0.787 | 0.027 | 0.967 |
| 0.40 | 0.760 | 0.020 | 0.974 |
| 0.50 | 0.747 | 0.007 | 0.991 |
| 0.60 | 0.713 | 0.007 | 0.991 |
| 0.70 | 0.693 | 0.007 | 0.990 |
| 0.80 | 0.653 | 0.007 | 0.990 |
| 0.90 | 0.593 | 0.000 | 1.000 |

### granite3b

| threshold | entail recall | hard-neutral FP | precision proxy |
|---|---|---|---|
| 0.05 | 0.767 | 0.067 | 0.920 |
| 0.10 | 0.767 | 0.067 | 0.920 |
| 0.20 | 0.767 | 0.067 | 0.920 |
| 0.30 | 0.767 | 0.067 | 0.920 |
| 0.40 | 0.767 | 0.067 | 0.920 |
| 0.50 | 0.767 | 0.067 | 0.920 |
| 0.60 | 0.767 | 0.067 | 0.920 |
| 0.70 | 0.767 | 0.067 | 0.920 |
| 0.80 | 0.767 | 0.067 | 0.920 |
| 0.90 | 0.767 | 0.067 | 0.920 |

### granite8b

| threshold | entail recall | hard-neutral FP | precision proxy |
|---|---|---|---|
| 0.05 | 0.900 | 0.240 | 0.789 |
| 0.10 | 0.900 | 0.240 | 0.789 |
| 0.20 | 0.900 | 0.240 | 0.789 |
| 0.30 | 0.900 | 0.240 | 0.789 |
| 0.40 | 0.900 | 0.240 | 0.789 |
| 0.50 | 0.900 | 0.240 | 0.789 |
| 0.60 | 0.900 | 0.240 | 0.789 |
| 0.70 | 0.900 | 0.240 | 0.789 |
| 0.80 | 0.900 | 0.240 | 0.789 |
| 0.90 | 0.900 | 0.240 | 0.789 |

## Derived citation precision estimate

**Stated assumption:** the Selector hands the Generator 5 chunks per query and, for any one atomic claim, 1 of them genuinely support it. Expected correct-to-total citation ratio = `recall*1 / (recall*1 + FP*4)` using each arm's ASQA numbers at its own operating point.

| arm | operating point | recall | hard-neutral FP | expected citation precision |
|---|---|---|---|---|
| true | argmax / 0.5 | 0.747 | 0.007 | 0.966 |
| granite3b | argmax / 0.5 | 0.767 | 0.067 | 0.742 |
| granite8b | argmax / 0.5 | 0.900 | 0.240 | 0.484 |

## Counterfactual slice (entity swaps, reported separately)

Built by swapping one entity in an ASQA claim that its gold passage actually states, for a same-type entity from a different question. Correct verdict is always 'not supported'.

| arm | caught by verifier alone | caught by verifier OR entity check |
|---|---|---|
| true | 0.963 | 1.000 |
| granite3b | 0.945 | 1.000 |
| granite8b | 0.872 | 1.000 |

Entity-check catch rate on its own, by swapped entity type:

| entity type | swaps | caught by entity_check |
|---|---|---|
| date | 42 | 1.000 |
| name | 59 | 1.000 |
| number | 8 | 1.000 |

Examples of the swaps made:

- `name` -- National Commission for Scheduled Castes -> Things I Hate About You
- `name` -- Academy Award for Best Supporting Actor -> How the Grinch Stole Christmas
- `date` -- 2001 -> 1964
- `name` -- All-Ireland Senior Hurling Championship -> Country Music Hall of Fame
- `date` -- 2016 -> December 29, 1845
- `name` -- Wine Festival -> Britain's Got Talent
- `name` -- Vietnam War. -> Los Angeles
- `number` -- 22,211 -> 1.0

## Granite judge prompt (arm E), verbatim

```text
You are checking whether a passage supports a claim.
Answer with exactly one word: SUPPORTS, CONTRADICTS, or NEITHER.
SUPPORTS  - the passage states or directly implies the claim.
CONTRADICTS - the passage states something incompatible with the claim.
NEITHER - the passage neither supports nor contradicts the claim.
Judge only from the passage; do not use outside knowledge.

Passage:
{premise}

Claim:
{hypothesis}

Answer:
```

Multi-hop cell built from 60 2Wiki compositional questions; the gold `evidences` triples are used as a decomposition oracle (a perfect A2) so the verifier is measured without the claim splitter's own quality as a confound.

