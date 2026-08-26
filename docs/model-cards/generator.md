# Grounded GR-C Generator Model Card

The final Generator uses IBM Granite 4.1 3B with a grounding-and-citation repair adapter (GR-C).
It drafts an answer from selected evidence, applies frozen verification/annotation logic, and
returns cited evidence IDs.

## Model details

| Field | Frozen value |
|---|---|
| Runtime name | `grounded-grc` |
| Base model | `ibm-granite/granite-4.1-3b` |
| Base revision | `c0650403e44e78ec0262dab1c90914c65b196c4e` |
| Runtime adapter | GR-C seed 13 |
| Adapter weights bytes | 62,332,992 |
| Adapter weights SHA-256 | `492d336c7acc32785becded707220dbdf1a8acf9a895630147fc4e8cd28d0707` |
| Adapter config bytes | 1,274 |
| Verification model | `google/t5_xxl_true_nli_mixture` |
| Public availability | `restricted-not-published` |

Three adapters were trained with seeds 13, 42, and 73. Their exact weights/config hashes and
shared recipe are in
[`frozen_provenance.json`](../../experiments/generator/frozen_provenance.json); seed 13 is the final
runtime identity.

## Training recipe

The frozen recipe uses one epoch, learning rate `5e-5`, query-group equalization, and LoRA with
rank 8, alpha 16, dropout 0.05, and seven attention/MLP target-module classes. It records 9,207
training examples, 2,370 train groups, 302 validation groups, and 297 optimizer steps.

The adapter applies only to the draft generation call. Claim splitting uses the base model with
adapters disabled, and the verifier is frozen rather than trained. Decision-development and
held-out roles were not read during training according to the frozen provenance record.

## Intended use

- Generate answers and citations from evidence already selected by the Pipeline.
- Support reproducible GR-C training, three-seed evaluation, and module ablations.
- Provide a backend response for research interfaces where citations and stage traces are visible.

It is not intended to retrieve evidence, override Selector decisions, provide guaranteed factual
answers, or support unaudited high-stakes decisions.

## Evaluation

Experiment 04 completed its registered protocol, but the final system did not establish Ours-RAR
superiority. The Direct Generator ablation was significantly higher than GR-C seed 13 on HotpotQA
and MuSiQue under the registered comparison; the RGB result was directional but inconclusive.

Experiment 05 registered Claims A and B were both `NOT SUPPORTED`, and module effects were
dataset-dependent. These findings make the frozen Generator an important limitation rather than a
universally improved component. See [results](../results.md).

## Limitations

- A 3B base model plus the TRUE T5-XXL verifier has substantial storage and accelerator cost.
- Verification and citations reduce neither all hallucinations nor source-level misinformation.
- Performance is sensitive to selected evidence, decoding limits, corpus quality, and domain shift.
- The public aggregate release cannot reproduce real generation without authorized model and data
  assets.

## Access and license

The Granite base and TRUE verifier are registered as Apache-2.0 at their frozen upstream revisions.
The project-trained GR-C adapters are not published because redistribution authorization has not
been recorded. The project license does not relicense these external or restricted model assets.
