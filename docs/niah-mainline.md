# How NIAH fits the mainline — the storyline (read this before the results)

A one-page bridge so anyone can read `results-summary.md` and see *why* each part exists
and how it connects. If you only read one doc before the summary, read this.

## One-sentence thesis

We build and rigorously evaluate a **Granite retrieval system that finds rare, relevant
evidence buried in massive, misleading data** — "finding the needle in the haystack."

## "NIAH" is used for three things — don't confuse them

- **NIAH the *problem* (our motivation).** Finding one correct piece of evidence when the
  corpus is huge and full of near-duplicate, misleading passages. This is *why* the project
  exists — the abstract pitch, not an experiment.
- **Our NIAH *evaluation task* (findings 12–18) — the one that matters here.** A task *we*
  built: plant one designated needle among convincing counterfactual distractors in a large
  corpus, then measure whether the system ranks it to the top (needle-found@k). This is *how
  we prove* the system solves the problem — the **core evaluation of the mainline**. When the
  summary says "NIAH" from finding 12 on, it means *this*.
- **The long-context "*Lost in the Middle*" NIAH — same name, not our thing.** The famous
  probe that hides one fact deep in a *single long document* and tests recall by depth. We
  listed it as an optional stress test and essentially did not run it; it only shares the
  name. When it comes up, skip it or say plainly "that is not the NIAH we did."

So NIAH is **not a second track next to the retrieval work — it *is* the evaluation of the
retrieval system** (sense 2). Keep the three apart and the "why is NIAH here" confusion goes away.

## The storyline — why each part exists, in order

1. **Why Granite (the bake-off).** First we show Granite's dense retriever is a competitive
   first stage vs BM25 and open peers on standard BEIR sets. This **justifies choosing
   Granite** — it is no longer the headline (Granite ≈ strong open peers on easy sets, wins
   on the harder FiQA). → *foundation.* (Findings 1–7.)
2. **The real problem: at scale, with misleading distractors, the bottleneck is RANKING,
   not recall.** Plant near-duplicate **counterfactual** distractors (the "misleading hay").
   The needle is almost always retrieved into the top-100 pool (R@100 = 0.87) but **buried at
   ranks 11–100**. Ranking failure ≈ 3.4× recall failure. This is NIAH-the-problem, measured.
   (Finding 12.)
3. **What fixes the ranking bottleneck.** Query reformulation (**q2d**) significantly helps
   (+0.070); relevance rerankers do **not** (a counterfactual *is* relevant, so relevance
   can't demote it). Our novel **Corroboration Reranking** — rank by cross-source answer
   agreement — is the one reranking approach that helps (+0.037). (Findings 13, 15, 18.)
4. **Does it survive scale?** Needle-finding degrades gracefully as the haystack grows
   10k → 5M; q2d's edge holds at every scale. (Finding 14 + the in-progress scale run.)
5. **Does better retrieval → better answers?** End-to-end RAG over the haystack: retrieval
   gains propagate to significantly better *cited answers* (F1) once the generator's context
   window matches the retrieval cutoff. (Findings 16–17.)
**Steps 2–5 above *are* our NIAH evaluation** — that is how "NIAH" and the retrieval mainline
combine: step 1 builds a good Granite retriever, steps 2–5 put it on the real needle-finding
task. (The long-context *Lost in the Middle* probe shares the name but is the separate,
optional, not-run stress test from the three senses above — do not conflate them.)

## Map — where each summary finding sits

| Findings | What it is | Role |
|---|---|---|
| **1–11** | Retrieval bake-off, efficiency, hybrid/SPLADE, reranking, failure analysis, + RAG on NQ/TriviaQA | **Foundation** — "Granite is a good retriever, and it generalises" |
| **12–18** | Diagnosis (ranking bottleneck) → q2d → corroboration → scale → end-to-end | **Headline story** — "finding the rare needle among misleading look-alikes" |

So when NIAH "appears" at finding 12, it is **not a new topic** — it is where the foundation
(a good Granite retriever) is turned on the actual thesis (find the rare needle among
misleading distractors). Everything before it earns the right to run that experiment.

## One-line reading guide

Read the summary as: *Granite is a good retriever (1–11) → but at scale the hard part is
ranking the needle above look-alike misinformation (12) → here is what fixes that (13–15, 18)
→ it holds at scale (14) and improves the final answers (16–17).*
