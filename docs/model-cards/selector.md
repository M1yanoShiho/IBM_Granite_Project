# Trained NLI Selector Model Card

The final Selector is a seed-13 risk-controlled NLI evidence filter. It receives a query and ranked
candidate evidence, then returns selected evidence IDs, scores, and ranks without rewriting text.

## Model details

| Field | Frozen value |
|---|---|
| Runtime name | `nli-risk-controlled` |
| Backbone | `cross-encoder/nli-deberta-v3-base` |
| Backbone revision | `6c749ce3425cd33b46d187e45b92bbf96ee12ec7` |
| Checkpoint bytes | 737,731,768 |
| Checkpoint SHA-256 | `86622bd9ab6391c9eb560133b01b0cf3744c3706638ff8b0bd38925b84bf72bf` |
| Safe threshold | 0.9212157130241394 |
| Maximum deletions | 2 |
| Public availability | `restricted-not-published` |

The canonical identities are duplicated intentionally across
[`configs/models/final_seed13.json`](../../configs/models/final_seed13.json) and
[`ARTIFACT_MANIFEST.json`](../../ARTIFACT_MANIFEST.json), with contract tests requiring agreement.

## Intended use

- Filter a small number of high-confidence misleading candidates after Hybrid retrieval.
- Support research on evidence selection, risk control, module swaps, and end-to-end attribution.
- Operate inside the Evidence RAG Pipeline, which resolves selected IDs to original evidence.

The model is not a general truth detector, content-safety classifier, legal or medical decision
tool, or proof that answer quality will improve.

## Training and evaluation

[`configs/selector/lean_v3.toml`](../../configs/selector/lean_v3.toml) freezes the backbone, data
roles, training schedule, threshold selection, statistics, and separate answer gate. The public CLI
is `evidence-rag-selector-train`; full training inputs and checkpoints remain external.

The misleading-evidence gate used two tested seeds, 10,000 bootstrap samples, a 0.99 quantile
policy, and a two-deletion cap. Seed 13 reduced harmful evidence by 12.95% and seed 42 by 13.10%,
with zero observed required-recall and required-chain loss. This evidence-level gate passed.

The separate blind answer gate failed: macro answer-match delta was −0.1353 percentage points with
a 95% interval from −0.5618 to +0.2770 points. The frozen overall decision was `KEEP_TOPK10`.

See [results](../results.md) and the two source-bound public summaries:

- [`misleading_evidence_summary.json`](../../results/selector/misleading_evidence_summary.json)
- [`blind_answer_gate.json`](../../results/selector/blind_answer_gate.json)

## Limitations

- The positive result is conditional on the tested misleading-evidence distribution.
- The point estimate for the blind answer gate was negative and its interval crossed zero.
- The Selector was mostly inactive on ordinary Experiment 05 inputs.
- NLI confidence can be miscalibrated under domain, language, corpus, or model shift.
- The two-deletion cap favors evidence retention but can leave harmful evidence in place.

## Access and license

The upstream backbone is registered as Apache-2.0 in its model card at the frozen revision. The
trained project checkpoint is not published because redistribution authorization has not been
recorded. Possession of a shared-storage copy does not itself grant redistribution rights.
