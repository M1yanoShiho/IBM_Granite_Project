# Embedder Hard-Negative Fine-Tuning (WS-14) — design spec

- Date: 2026-07-08
- Owner: TBD (WS-14); spec author P6 (Weikai)
- Status: design approved (brainstorm 2026-07-08); pending spec review → writing-plans.
- Context: the representation-level attack on the NIAH ranking bottleneck. Sibling of the
  **query-stage** lever (q2d) and the **rank-stage** lever (Corroboration Reranking) — this
  is the **first-stage / representation** lever. Explicitly STRETCH / high-risk; gated on
  WS-5. Related: [[progress-2026-07-05]], the corroboration spec (2026-07-05), work plan
  WS-5/WS-8/WS-14.

## 1. Problem

On the NIAH task the needle is almost always in the pool (R@100 = 0.87) but only ~0.49
(dense) / ~0.56 (q2d) / ~0.61 (+corroboration) reach top-10; the dominant failure is
**~0.26 of needles buried at ranks 11–100** (ranking headroom) vs ~0.13 unreachable
(recall headroom). The worst distractor is the Source-A **counterfactual** — the needle's
own text with the answer entity swapped to a type-consistent wrong one. Needle and
counterfactual differ by **~1 token**, so the dense embedder maps them to **near-collinear
vectors** and cannot separate them.

The two levers already certified attack this indirectly:
- **q2d (query-stage, +0.070):** enriches the *query* representation; leaves the needle/cf
  vectors untouched in **doc space**, so it can only nudge the query toward the needle's
  barely-different vector.
- **Corroboration (rank-stage, +0.037, MRR-flat):** re-orders a top-N pool by cross-source
  agreement; a coarse, boundary-specific correction.

Neither fixes the **root cause: doc-space collinearity of near-duplicates.** That is what
representation fine-tuning attacks directly.

## 2. Goal & novelty (honest)

Fine-tune `granite-embedding-english-r2` (~150M, the current dense retriever) with the
counterfactual as an **explicit hard negative**, so the first-stage embedding space itself
**pushes the needle apart from its entity-swapped near-dup**. Because it changes the
ranking at the retrieval stage (not a post-hoc re-order), it is the one lever that can move
**MRR**, not just found@10.

**Novelty (honest):** a *known technique* — hard-negative contrastive fine-tuning / embedder
domain adaptation — applied to the **near-duplicate counterfactual** retrieval pathology,
with **construction-aware, artifact-avoiding** negatives (§4). Not a new primitive. The
defensible MSc framing is a **decomposition of the ranking bottleneck by intervention
stage** — representation (this) vs query (q2d) vs rank (corroboration) — and which stages
*compose*. It must NOT be framed as "solving counterfactual robustness".

## 3. Success criteria & the q2d-complementarity thesis (reframed)

Head-to-head "pure FT beats q2d (0.563)" is a steep, uncertain bar and is NOT the pass
line. The bar is complementarity:

- **Primary (validates the approach):** fine-tuned dense first stage significantly beats
  **baseline dense (0.493)** on needle-found@10 (frozen 300q, `eval.significance`), and
  ideally moves **MRR** (which rerankers/corroboration structurally cannot).
- **Core question (the real deliverable):** is FT **complementary to q2d** — i.e. does it
  attack a *different* failure mode? Two measurements settle it: (a) **WS-5 pool-union
  recall** — does q2d work by expanding recall (new needles into the pool) or by re-ranking
  the existing pool? and (b) the **FT × q2d ablation**. If q2d is a recall lever and FT a
  ranking lever, they stack; if q2d is *also* a ranking lever, FT and q2d compete for the
  same ~0.26 and the stack shows diminishing returns. **This is the hypothesis to TEST — do
  not pre-assume the stack breaks through.**
- **Bonus (not required):** FT alone ≥ q2d. Plausible *on this specific pathology* (FT is
  the more direct fix for near-dup collinearity) but risked by §10's entity-hypersensitivity.

## 4. Data construction (the crux — artifact-avoiding)

Per training query: **1 positive** (the true needle passage) + **K≈4–8 hard negatives** +
in-batch negatives.

- **Hard negatives via Longpre-style entity substitution** (arXiv:2109.05052), NOT the
  `str.replace` Source-A (so the model cannot shortcut on a logical seam):
  - **Corpus substitution (MVP):** replace the answer entity with a **same-type wrong
    entity sampled from the dataset's answer set** (spaCy NER for the entity type). Cheap,
    no external deps.
  - **Popularity-controlled substitution (stretch):** same-type Wikidata entity drawn from
    a popularity band — teaches the model to prefer the *contextually correct* entity over
    the merely *popular* one (Longpre: models are popularity-biased).
- **Positives / augmentation:** the needle; **alias substitution is a *correct* paraphrase
  → optional positive augmentation, NEVER a negative** (this was the one correction to the
  brainstorm — "alias/popularity" is not a single negative source).

Two query variants (both built, for the ablation and the synergy test):
- **short:** the natural query.
- **long (q2d-expanded):** the q2d pseudo-document appended, matching the FT+q2d inference
  distribution — otherwise an encoder fine-tuned only on short queries sees q2d's long
  representation off-distribution at test time and the synergy is lost.

## 5. Training

- Base: `granite-embedding-english-r2`; loss = **MultipleNegativesRankingLoss** (InfoNCE)
  with explicit hard negatives + in-batch negatives (sentence-transformers).
- **LoRA by default** — cheap, reversible, less catastrophic forgetting on a 150M encoder;
  full fine-tune only as a fallback if LoRA underperforms (YAGNI otherwise).
- Two checkpoints (short-query-trained, long-query-trained) so the ablation isolates the
  q2d-synergy effect.

## 6. Rigor — train/test disjointness + generalisation validity

- **No train-on-test:** training queries drawn from NQ/dpr-w100 **disjoint from the frozen
  300q** (filter by qid). Same corpus, different queries. This is make-or-break; a leak
  invalidates the whole result.
- **Generalisation validity check:** train the negatives on corpus/popularity substitutions
  only, then **test on the frozen task's `str.replace` Source-A counterfactuals the model
  never trained on**. If it still separates them, it learned real entity-sensitivity, not
  the construction artifact. If it only separates the substitution type it trained on, say
  so — that is the honest negative.

## 7. Evaluation — the ablation matrix

Swap each fine-tuned checkpoint in as the first stage in `run_niah`; frozen 300q;
**needle-found@10 + MRR**; significance vs baseline dense and vs q2d.

Ablation grid = **FT-checkpoint {short, long} × stack {alone, +q2d, +corroboration}**, all
against the baseline-dense and q2d rows already certified:

|  | alone | + q2d | + q2d + corroboration |
|---|---|---|---|
| baseline dense | 0.493 (ref) | 0.563 | 0.607 |
| FT (short-trained) | ? | ? | ? |
| FT (long / q2d-trained) | — | ? (synergy cell) | ? |

(Columns are the certified pipeline stages — dense→q2d→corroborate = 0.493→0.563→0.607;
corroboration always sits on top of q2d. FT(long) has no `alone` cell — feeding a short
query to a q2d-query-trained encoder is off-distribution.) The **FT(long)+q2d cell** is
where the hoped complementarity shows up; read it together with WS-5's pool-union recall
(§3). Report whatever happens — including "FT overlaps q2d, no stack gain", which is a real
finding about the ranking bottleneck.

## 8. Integration & components (isolated, TDD-able)

- `src/niah/entity_substitution.py` — Longpre-style corpus/popularity substitution
  (spaCy NER + same-type sampling). Pure logic, unit-tested (given an entity + type, yields
  a same-type wrong entity; never yields the original; alias path yields a correct paraphrase).
- `eval/build_finetune_data.py` — assemble disjoint `(query, positive, [hard-negs])` triplets
  (short + long variants) → JSONL. Tested: disjointness from the frozen qids; K negatives;
  no positive-as-negative leakage.
- `scripts/finetune_embedder.py` (+ `scripts/run_finetune_embedder.slurm`) — the
  sentence-transformers LoRA training loop; checkpoints to `/user/work/$USER/`.
- `eval/run_niah.py` — a small addition to point the dense embedder at a **local fine-tuned
  checkpoint path** (env var / `--embedder-path`), reusing the existing retriever factory.

## 9. Gating & sequencing (STRETCH discipline)

- **Gated on WS-5:** run the oracle headroom decomposition FIRST. If it shows little
  reachable ranking headroom (q2d already captures it), scope WS-14 down or drop it — WS-5
  also validates §3's complementarity premise before any GPU is spent.
- **Long-pole background only:** never displaces the spine (WS-1a/1b/10) or the report for
  GPU (work-plan GPU-contention rule). MVP ≈ 1–2 GPU-days; the full grid + popularity +
  full-FT is multi-week.
- **Freeze 08-03:** if not certified by then it is **Future Work, not a headline**.
- Runs in parallel with WS-8 (recall arm) — different headroom, plausibly additive; the two
  first stages may eventually be composed (fine-tuned dense + SPLADE convex hybrid).

## 10. Risks & honest limitations

- **Entity-hypersensitivity (the real reason FT may lose to q2d):** training the encoder to
  split near-dups on the answer token can over-weight entity tokens and **degrade general
  semantic retrieval** (normal near-synonym matching). Guard: hold out a standard retrieval
  metric (a small BEIR slice) as a regression check; if general nDCG drops, the FT is
  overfit to the pathology.
- **Artifact overfitting:** mitigated by fluent substitution negatives (§4) + the §6
  generalisation check.
- **Distribution mismatch:** short-trained FT + long q2d queries loses synergy — handled by
  the long-query training variant (§4/§5).
- **May not beat q2d, may not stack:** both are acceptable, reportable outcomes under the
  §3 reframing — the deliverable is the *decomposition by stage*, not a single number.
- **Novelty is an application** (§2), not a new primitive — frame as such.

## 11. References (verify before citing)
- Entity-based knowledge conflicts — Longpre et al., EMNLP 2021 (substitution policies:
  corpus / popularity / alias; the hard-negative source).
- Query2Doc — Wang et al., 2023 (the query-stage lever this complements).
- sentence-transformers MultipleNegativesRankingLoss (hard-negative contrastive training).
- Dense passage retrieval / hard-negative mining (DPR; the standard training recipe).
