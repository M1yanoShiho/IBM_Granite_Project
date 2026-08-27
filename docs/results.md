# Evidence RAG Results

The final evidence is mixed. The trained Selector showed a conditional evidence-level benefit in a
misleading-evidence stress test, but neither the dedicated answer gate nor the broader system
experiments established a general final-answer improvement.

## Selector gates

| Gate | Frozen outcome | Interpretation |
|---|---|---|
| Misleading-evidence stress | `PASS` | Harmful-evidence reduction was 12.95% for seed 13 and 13.10% for seed 42, with zero observed required-recall and required-chain loss. |
| Blind answer gate | `FAIL` | Macro answer-match delta was −0.1353 percentage points; the 95% interval was −0.5618 to +0.2770 points. The decision remained `KEEP_TOPK10`. |

The stress result supports a narrow claim: under the tested misleading-evidence condition, the NLI
policy selectively removed harmful candidates. It does not establish a final-answer benefit. Both
gates come from the same frozen archive artifact and SHA-256:

- [`misleading_evidence_summary.json`](../results/selector/misleading_evidence_summary.json)
- [`blind_answer_gate.json`](../results/selector/blind_answer_gate.json)

## Experiment 04

Experiment 04 compared four baselines with the three-seed final system on HotpotQA,
MuSiQue-answerable, and RGB-noise, then evaluated Retriever, Selector, and Generator ablations.

- The registered Ours-RAR superiority claim was `NOT_SUPPORTED`.
- Replacing the trained Selector with Top-10 produced no observed RAR difference on the frozen
  samples.
- Replacing the Hybrid Retriever with the dense ablation also produced no observed RAR difference.
- The Direct Generator was significantly higher than GR-C seed 13 on HotpotQA and MuSiQue for the
  registered comparison; the RGB direction was inconclusive because its interval included zero.

The technical audit status is `FINAL PASS`, meaning all registered stages and integrity checks
completed. It is not a performance claim. See the
[main results](research/experiment04-main-results.md),
[ablation results](research/experiment04-ablation-results.md), and
[`final_results.json`](../results/experiment04/final_results.json).

## Experiment 05

Experiment 05 evaluated ordinary RAG data from KILT-NQ, KILT-TriviaQA, and ALCE-ASQA across ten
arms. The frozen audit records 12,000 generation outputs, 12,000 query scores, zero scorer errors,
and 10,000 bootstrap resamples.

- Registered Claim A was `NOT SUPPORTED`.
- Registered Claim B was `NOT SUPPORTED`.
- The trained Selector was mostly inactive on ordinary Experiment 05 inputs, so its ablation did
  not provide evidence of an ordinary-data system improvement.
- Retriever and Generator contributions were dataset-dependent rather than uniformly positive.

The audit is `FINAL PASS` for protocol execution only. The scientific decisions are the two `NOT
SUPPORTED` labels in [`claim_labels.json`](../results/experiment05/claim_labels.json). See the
[findings](research/experiment05-findings.md) and
[`final_audit.json`](../results/experiment05/final_audit.json) for the frozen record.

## How to cite the findings

Report module-level and whole-system evidence separately. A faithful summary is:

> The trained NLI Selector reduced harmful evidence in a dedicated stress test without observed
> required-evidence loss, but its blind answer gate failed and the broader registered system
> superiority claims were not supported.

Do not convert a technical `FINAL PASS`, an evidence-level gate, or a successfully connected
pipeline into an unqualified answer-quality improvement claim.
