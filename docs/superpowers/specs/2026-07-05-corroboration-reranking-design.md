# Corroboration Reranking — design spec

- Date: 2026-07-05
- Owner: P6 (Weikai Mao)
- Status: design approved (brainstorm 2026-07-05); pending spec review → writing-plans.
- Context: Phase-2/3 "make-it-stronger" upside on the certified NIAH pipeline. Sibling
  of `AstuteRAGPipeline` (source-aware consolidation) moved from the **generation**
  stage to the **rank** stage. Related: [[progress-2026-07-02]].

## 1. Problem

On the NIAH task the needle is almost always retrievable (R@100 = 0.87) but only
~0.49–0.56 reach top-10; ~0.38 are **buried at ranks 11–100 by distractors**. The
worst offender is the Source-A **counterfactual** (`make_counterfactual` = LLM picks a
type-consistent wrong entity, then a pure `str.replace` swaps it into the needle). It
is *relevant* (on-topic, answer-shaped) but factually wrong. Every reranker we tried
(cross-encoder, 3B listwise, stacked-on-q2d) failed: **they rank by relevance, and the
counterfactual IS relevant.** Certified: query-transform (Q2D, +0.07 p=.0002) helps at
the query stage; nothing at the rank stage does.

## 2. Goal & novelty

A **training-free reranker that ranks by cross-source answer corroboration, not query
relevance**, so a lone factually-wrong passage is demoted even though it is
semantically relevant. The true answer is corroborated by other retrieved passages
and the model's own knowledge; the counterfactual's fabricated entity is a lone
outlier.

**Novelty (honest):** a *new combination/application*, not a new primitive — answer
voting (open-domain QA) + source-aware consolidation (Astute) recast as a **rank-stage
signal targeting knowledge-conflict distractors in rare-needle retrieval**. Defensible
as an MSc contribution; must NOT be framed as "solving counterfactual robustness".

## 3. Method

First stage (q2d_granite or granite_dense) returns a top-K pool. Rerank the top-N
(N ≪ K, configurable, cost control):

1. **Answer extraction** — per candidate `c_i`, prompt Granite: *"Using ONLY this
   passage, answer the question; if it does not answer it, say NONE."* → `a_i` (short
   string or NONE). N LLM calls/query.
2. **Parametric vote (anchor)** — one no-passage elicitation → `a_0` (reuse Astute's
   ELICIT). One extra independent voter / tie-breaker.
3. **Corroboration score** (pure, no LLM — see §4) — how many *other* sources support
   `a_i`, by exact-match voting **and** entity-presence in other passages' text.
4. **Additive-boost rerank** — `final_i = relevance_i + λ·corroboration_i`; relevance
   is the base, corroboration a boost (λ=0 ⇒ plain first stage; λ tuned offline,
   reusing `eval/tune_alpha.py`). Sort top-N by `final_i`, keep the rest of the pool
   below. NONE-answer candidates get corroboration 0 (fall back to relevance).

The counterfactual `Y` appears in no other passage (it was swapped only into its own
text) and is corroborated by nobody ⇒ its boost is 0 while a genuinely relevant needle
`X` (echoed by other golds / the corpus / parametric) gets a positive boost ⇒ the
needle rises above its counterfactual twin.

## 4. Corroboration score (the crux — pure & unit-testable)

`corroboration_scores(answers, texts, parametric) -> List[float]`:
- `a_i == NONE` → `0.0`.
- else `score_i = |{ j≠i : norm(a_j) == norm(a_i) }|            # exact-answer votes`
  `           + |{ j≠i : norm(a_i) ⊆ norm(texts_j) }|          # entity-presence side-clues`
  `           + (1 if parametric and norm(parametric)==norm(a_i) else 0)`
- `norm` = lowercase, strip punctuation + leading articles.

Pure function, no LLM ⇒ tested directly: a lone answer scores 0; an answer echoed by
other passages/parametric scores high; a counterfactual entity present only in its own
passage scores 0.

## 5. Tie-break — the known failure mode (measure first)

**Honest weakness (raised in review):** when the correct answer appears in *only* the
needle within the top-N window (sparse-gold query, or small N), `X` and `Y` both score
0, the additive boost cannot separate them, and the fallback to relevance favours the
near-identical counterfactual. Multi-gold NQ + unlabeled `X`-bearing passages make this
a *subset*, not the norm — but it is exactly the hardest subset.

**Do not over-engineer before measuring.** Mandatory diagnostics (§7): the **tie rate**
and the **counterfactual-demotion rate**. Only if ties dominate, add:

**Entailment tie-breaker (external, cost-bounded):** on a tie among answer-bearing
candidates with *different* answers, ask Granite which answer the pool's collective
context better supports. This leverages the counterfactual's **logical seam** — its
un-swapped context (e.g. "the turn of the century") still entails the true `X` and
contradicts its own swapped `Y` — but reads it as an **external** contradiction
(other/own context vs the answer), NOT single-doc artifact detection. Applied only to
ties, so cost stays bounded.

## 6. Integration (minimal, follows existing patterns)

- `CorroborationReranker` in `src/retrieval/reranker.py`, beside `Reranker` /
  `LLMListwiseReranker`; same `rerank(query, candidates, top_k)` interface. Holds the
  injected LLM (reused, no second load), `top_n`, `lambda_`.
- Units: `corroboration_scores(...)` (pure, §4) · `extract_answer(passage, query, llm)`
  helper (prompt + parse) · the class assembling them. Each independently testable.
- Wrapped by the existing `TwoStageRetriever(first_stage, CorroborationReranker(llm), …)`.
- `run_benchmark` spec `q2d_corroborate` (rerank the q2d first stage), reusing the
  shared-LLM gate (`retrievers_need_llm`) so it drops into `run_niah` unchanged.
- λ tuned via `eval/tune_alpha.py` (same offline-sweep pattern as the hybrid).

## 7. Evaluation

- `run_niah` on the frozen 300q task: needle-found@10 + MRR.
- **Ablation (the result table):** `q2d_corroborate` vs cross-encoder rerank vs listwise
  rerank vs q2d-alone — same first stage — → "corroboration beats relevance-reranking".
  Significance vs q2d and vs the cross-encoder (`eval/significance.py`).
- **Mechanism diagnostics (report figure):** using the task's known
  `parent_needle_id`, measure (a) **counterfactual-demotion rate** (how often the
  Source-A distractor is pushed out of top-10) and (b) the **tie rate** (§5). These
  show the method works by design, not as a black box.

## 8. Bounded optional sub-experiment — internal consistency

Single-doc seam detection (LLM-as-judge: is the extracted answer consistent with the
rest of *its own* passage?). **Deliberately NOT core.** Because `make_counterfactual`
is a naive `str.replace`, seams are guaranteed, so a single-doc detector would win
**because our generator is naive** — an artifact of construction that would not
generalise to fluent, LLM-rewritten misinformation. If run at all, it must be reported
as a **bounded** finding ("effective on mechanical entity-swap counterfactuals") and
tested against §9's hardened seamless counterfactuals — never claimed as general
counterfactual robustness.

## 9. Parallel track — harden Source-A (task validity)

The seam analysis exposes that Source-A counterfactuals carry obvious logical seams
(naive `str.replace`). Independently of this method, an **LLM-rewrite** counterfactual
(swap the entity AND smooth the surrounding context) is a stronger, more realistic
adversary. Hardening it improves the whole task's defensibility **and** is the honest
test of whether Corroboration Reranking's gain is real (external corroboration should
survive seamless counterfactuals; a seam-detector would not). Recommended as a sibling
task, gated separately.

## 10. Risks & honest limitations

- **Cost:** N answer-extractions/query → rerank only top-N; entailment tie-breaker only
  on ties.
- **Coordinated misinformation:** if many passages share the *same wrong* answer,
  corroboration is fooled. The task's counterfactual is a lone outlier, so this holds
  here; note as a real-world limitation.
- **Extraction quality:** 3B may mis-extract; 8B is more reliable but slower — an
  8B-vs-3B point.
- **Tie subset (§5):** the method's weakest region; measured, not hidden.
- **Novelty is a combination/application** (§2), not a new primitive — frame as such.

## 11. References (verify before citing)
- Entity-based knowledge conflicts — Longpre et al., EMNLP 2021 (Source A lineage).
- Astute RAG — Wang et al., 2024 (source-aware consolidation; generation-stage sibling).
- Self-consistency / answer voting in open-domain QA.
- RankGPT / listwise reranking (the relevance-ranking baseline this departs from).
