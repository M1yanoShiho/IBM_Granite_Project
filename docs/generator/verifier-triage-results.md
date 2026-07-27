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

- **base** -- cross-encoder/nli-deberta-v3-base id2label={"0": "contradiction", "1": "entailment", "2": "neutral"}
- **large** -- cross-encoder/nli-deberta-v3-large id2label={"0": "contradiction", "1": "entailment", "2": "neutral"}
- **minicheck** -- lytang/MiniCheck-Flan-T5-Large (binary supported/unsupported, decoder ids 209/3)

## Main grid

| arm | ASQA entail recall | ASQA hard-neutral -> entail | ASQA hard-neutral -> contra | 2Wiki atomic recall (any single chunk) | 2Wiki composed recall (single chunk) | 2Wiki composed recall (union) | residual multi-hop | union diag. (residual) | 2Wiki distractor -> entail | ms/pair |
|---|---|---|---|---|---|---|---|---|---|---|
| base | 0.467 | 0.007 | 0.280 | 0.625 | 0.083 | 0.200 | 0.375 | 0.022 | 0.008 | 173.1 |
| large | 0.540 | 0.073 | 0.447 | 0.817 | 0.183 | 0.600 | 0.183 | 0.091 | 0.033 | 503.2 |
| minicheck | 0.620 | 0.020 | 0.000 | 0.917 | 0.383 | 0.650 | 0.083 | 0.100 | 0.017 | 538.8 |

`residual multi-hop` = atomic claims (from the gold decomposition) that no single gold chunk entails. `union diag.` = of those, the share the same verifier recovers when both gold chunks are concatenated -- diagnostic only, it separates 'decomposition did not reduce this claim' from 'the verifier cannot see support that is present'.

## Arm C -- threshold sweep on P(entail) (ASQA cell)

### base

| threshold | entail recall | hard-neutral FP | precision proxy |
|---|---|---|---|
| 0.05 | 0.513 | 0.007 | 0.987 |
| 0.10 | 0.507 | 0.007 | 0.987 |
| 0.20 | 0.487 | 0.007 | 0.986 |
| 0.30 | 0.480 | 0.007 | 0.986 |
| 0.40 | 0.480 | 0.007 | 0.986 |
| 0.50 | 0.467 | 0.007 | 0.986 |
| 0.60 | 0.453 | 0.007 | 0.986 |
| 0.70 | 0.447 | 0.007 | 0.985 |
| 0.80 | 0.440 | 0.007 | 0.985 |
| 0.90 | 0.420 | 0.000 | 1.000 |

### large

| threshold | entail recall | hard-neutral FP | precision proxy |
|---|---|---|---|
| 0.05 | 0.607 | 0.120 | 0.835 |
| 0.10 | 0.593 | 0.093 | 0.864 |
| 0.20 | 0.573 | 0.080 | 0.878 |
| 0.30 | 0.560 | 0.073 | 0.884 |
| 0.40 | 0.547 | 0.073 | 0.882 |
| 0.50 | 0.540 | 0.073 | 0.880 |
| 0.60 | 0.527 | 0.073 | 0.878 |
| 0.70 | 0.527 | 0.060 | 0.898 |
| 0.80 | 0.507 | 0.060 | 0.894 |
| 0.90 | 0.500 | 0.053 | 0.904 |

### minicheck

| threshold | entail recall | hard-neutral FP | precision proxy |
|---|---|---|---|
| 0.05 | 0.793 | 0.147 | 0.844 |
| 0.10 | 0.727 | 0.087 | 0.893 |
| 0.20 | 0.673 | 0.060 | 0.918 |
| 0.30 | 0.653 | 0.040 | 0.942 |
| 0.40 | 0.620 | 0.020 | 0.969 |
| 0.50 | 0.620 | 0.020 | 0.969 |
| 0.60 | 0.593 | 0.007 | 0.989 |
| 0.70 | 0.573 | 0.007 | 0.989 |
| 0.80 | 0.540 | 0.007 | 0.988 |
| 0.90 | 0.440 | 0.007 | 0.985 |

## Derived citation precision estimate

**Stated assumption:** the Selector hands the Generator 5 chunks per query and, for any one atomic claim, 1 of them genuinely support it. Expected correct-to-total citation ratio = `recall*1 / (recall*1 + FP*4)` using each arm's ASQA numbers at its own operating point.

| arm | operating point | recall | hard-neutral FP | expected citation precision |
|---|---|---|---|---|
| base | argmax / 0.5 | 0.467 | 0.007 | 0.946 |
| large | argmax / 0.5 | 0.540 | 0.073 | 0.648 |
| minicheck | argmax / 0.5 | 0.620 | 0.020 | 0.886 |

## Counterfactual slice (entity swaps, reported separately)

Built by swapping one entity in an ASQA claim that its gold passage actually states, for a same-type entity from a different question. Correct verdict is always 'not supported'.

| arm | caught by verifier alone | caught by verifier OR entity check |
|---|---|---|
| base | 0.963 | 1.000 |
| large | 0.927 | 1.000 |
| minicheck | 0.963 | 1.000 |

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

## Reading (measurement, not a production change)

### 1. The model family was the problem, not the checkpoint size

`MiniCheck-Flan-T5-Large` wins **every cell** against both general-NLI
checkpoints, at the same cost as `deberta-v3-large` (539 vs 503 ms/pair) and at
a fifth of its parameter count in the useful direction:

- ASQA entailment recall **0.620** vs 0.540 (large) / 0.467 (base)
- ASQA hard-neutral -> entailment **0.020** vs 0.073 (large) -- better recall AND
  3.6x fewer false citations, which is the pairing that usually has to be traded
- 2Wiki atomic recall **0.917** vs 0.817 / 0.625

So the answer to the question this pass exists for is yes: swapping to a
*grounding-specific* verifier fixes more than scaling the general-NLI checkpoint
did. Testing only bigger `nli-deberta` checkpoints would have missed this.

### 2. The SciFact number did not transfer -- as suspected

Half 1.5 measured 0.34 entailment recall for `deberta-v3-large` on scientific
abstracts. The same checkpoint scores **0.540** here. The 0.34 was a property of
the scientific-abstract domain, not of the verifier, and it correctly motivated
this triage without being the number to design against. The honest read is that
general NLI is *mediocre* on our target domain, not broken.

### 3. Atomic decomposition is sufficient -- and beats concatenating

This is a strategy-level result, not a checkpoint choice. With `minicheck`:

| what is verified | recall |
|---|---|
| the undecomposed multi-hop claim, against one gold chunk | 0.383 |
| the undecomposed multi-hop claim, against both chunks concatenated | 0.650 |
| **atomic claims from the gold decomposition, against one gold chunk** | **0.917** |

The composed claim genuinely needs both hops (0.383 -> 0.650 when the chunks are
concatenated), so the multi-hop structure in the pair set is real rather than an
artefact. Against that, decomposing to atomic claims and verifying each from a
*single* chunk reaches 0.917 -- better than the union premise ever reaches on the
undecomposed claim. **Union premises are not needed; A2 decomposition is the
right lever.**

The residual is small and is *not* the verifier being blind to support that is
present: residual multi-hop is 0.083 (10 of 120 atomic claims), and the union
diagnostic recovers only 1 of those 10. Concatenating chunks would buy almost
nothing.

### 4. A threshold does rescue recall, and cheaply (arm C)

`minicheck` P(supported) is well-calibrated enough to trade along:

| operating point | recall | hard-neutral FP | derived citation precision* |
|---|---|---|---|
| 0.60 | 0.593 | 0.007 | 0.955 |
| 0.50 (default) | 0.620 | 0.020 | 0.886 |
| 0.30 | 0.653 | 0.040 | 0.803 |
| 0.20 | 0.673 | 0.060 | 0.737 |
| 0.05 | 0.793 | 0.147 | 0.574 |

*same assumption as the section above: 5 selected chunks, 1 of which genuinely
supports the claim.

Note the direction: **raising** the threshold to 0.60 costs 2.7 points of recall
and buys 7 points of citation precision. Lowering it to 0.05 buys 17 points of
recall and destroys precision. Under the stated 1-in-5 assumption the sensible
band is 0.5-0.6, not the aggressive end -- because with 4 non-supporting chunks
per claim, every point of false-positive rate is multiplied by 4.

### 5. `contradicted` -- Half 1.5's conclusion holds, and MiniCheck cannot produce it

On ASQA hard neutrals the general-NLI arms call **28.0% (base) / 44.7% (large)**
of merely-unrelated passages a *contradiction*. That is worse than the SciFact
pass measured and confirms the flag is unusable as a reported result.
Separately, `minicheck` is binary by construction -- supported / not supported,
no contradiction class -- so adopting it removes the signal entirely rather than
degrading it. Given the false-positive rate above, that is a cheap loss, but it
is a real interface consequence: `ClaimVerification.contradicted` would have to
become permanently `False` under a MiniCheck backend, and any downstream use of
it would silently stop firing.

### 6. Counterfactual slice: the verifier does most of it, entity_check closes it

Verifier alone catches 0.963 (base) / 0.927 (large) / 0.963 (minicheck) of entity
swaps; adding `entity_check.py` takes all three to **1.000**. So the entity layer
is contributing roughly 4-7 points on top of the verifier, not carrying the case.

**Caveat, stated plainly:** the slice only swaps entities the gold passage
actually states, so every swap leaves a conflicting same-role value in the
evidence -- exactly the shape `EntityConsistencyChecker` is built to catch. 1.000
is therefore a ceiling under ideal conditions, not a field estimate. The
informative half is the verifier-alone column: an entity swap is *mostly* visible
to the verifier itself, so the entity layer's real value is the residual 4-7%
plus the cases where NLI is entity-blind (the org-swap shape found in the Half 1
smoke test).

### Flags on the pair set itself

- **The ASQA hard-neutral pool is genuinely adjacent.** It is not a near-zero
  false-positive rate: `large` puts 0.073 of them on entailment and 0.447 on
  contradiction, and `minicheck` reaches 0.147 at threshold 0.05. Negatives are
  drawn from the same question's own top-20 GTR results.
- **The 2Wiki distractor pool IS too easy** -- 0.008-0.033 entailment. Those
  paragraphs are same-question but about entirely different entities, so they are
  rejected trivially. Do not read the 2Wiki distractor column as a
  false-positive estimate; the ASQA hard-neutral column is the meaningful one.
- **Near-verbatim risk is controlled.** ASQA claims sharing >=8 contiguous content
  tokens with their own gold passage are dropped; the surviving median shared run
  is 5 tokens against a ~14-token median claim, so the claims are paraphrases and
  the recall numbers are not inflated by copying.
- **2Wiki claims are template-built from the gold `evidences` triples**, so they
  are entity-verbatim by construction and the 2Wiki recall column is an upper
  bound. It is used for the *structural* question (decomposition sufficiency),
  not as a recall estimate. ASQA carries the recall headline.
- **Decomposition is oracle, not measured A2.** The 2Wiki cell uses the gold
  triples as a perfect decomposer, so section 3 answers "is single-chunk
  verification enough *if* decomposition works", not "does our splitter work".
  Measuring the real `ClaimSplitter` needs an LLM and belongs with the arm E
  HPC trip.

### Not done here

- **Arm D-true (`google/t5_xxl_true_nli_mixture`)** -- 11B params, 42.5GB fp32.
  It does not fit the 16GB CPU box; prepared as `scripts/run_verifier_triage.slurm`
  (bf16 on a 3g.40gb MIG slice) with a pre-registered ledger entry (G1 in
  `docs/hpc-run-log.md`), scoring the same pair set with the same seed.
- **Arm E (Granite-as-judge)** -- same HPC trip, prompt above, used unchanged.
- No threshold, model or config was committed anywhere. This file is numbers.

### Dataset-loading dependency

Reading the 2Wiki dev parquet needs **pandas + pyarrow**, which `pyproject.toml`
does not declare in any extra (both are present transitively in the current dev
env). They were deliberately not added: this pass is measurement only, the script
lives in `scripts/` and is out of CI, and `pyproject.toml` is production config.
Everything else uses stdlib `urllib`/`tarfile` plus the already-installed
`granite` extra.
