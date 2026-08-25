# Experiment 05 — Result-to-Claim Findings

**Date:** 2026-08-24  
**Verdict:** `PARTIAL / CONDITIONAL SUPPORT`  
**Integrity:** `WARN` (documentation/dead-helper notes only; no main-result integrity failure)

## What the evidence supports

The Selector has a supported conditional, module-level contribution in the dedicated misleading-evidence
evaluation. Across two seeds it reduced known harmful evidence by `12.95% / 13.10%`, achieved deletion
precision of `75.44% / 80.56%`, and caused `0 pp` observed required-evidence recall loss and `0 pp`
multi-hop-chain loss. The seed-13 harmful-reduction 95% CI was `10.34%–15.84%`, fully above zero and well
above random- and tail-deletion controls.

This supports the scoped claim: when misleading evidence is present in the tested setting, the Selector can
selectively suppress harmful passages while preserving required evidence.

## What the evidence does not support

- The Selector did not improve end-answer accuracy in the dedicated blind test: NIAH was `−0.2706 pp`,
  2Wiki was `0 pp`, and the confidence interval crossed zero.
- In Experiment 05 ordinary NQ/TriviaQA/ASQA, the frozen Selector had `0.0` selection and presentation
  reduction. Near-identical Full-vs-keep-all scores therefore describe inactivity under those conditions and
  cannot be used as causal evidence of an ordinary-dataset uplift.
- The broader Ours-vs-Hybrid citation gains cannot be assigned to Selector alone.
- The preregistered whole-system Claim A/B labels remain `NOT SUPPORTED`; they test stronger, different
  claims than the conditional Selector mechanism.

## Recommended thesis framing

Lead with the observed system and module findings, then report the confirmatory gate separately:

> The full system improved citation precision and citation coverage across all three ordinary RAG datasets.
> The dedicated misleading-evidence evaluation further showed a conditional Selector contribution: harmful
> passages were removed with high precision while required evidence was preserved. On ordinary datasets, the
> conservative Selector rarely had an opportunity to intervene, so the near-zero Full-vs-keep-all difference
> should be interpreted as inactivity rather than evidence of no module value. The stronger preregistered
> whole-system superiority claims did not cross their confirmatory gates and remain NOT SUPPORTED.

No frozen Experiment 05 result or preregistered label is changed by this interpretation.
