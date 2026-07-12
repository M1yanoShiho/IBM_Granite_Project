# Prompt Centralization — design spec

- Date: 2026-07-12
- Owner: TBD
- Status: design approved (decision session 2026-07-12); owner to be assigned.
- Context: prompts have accreted across 8 files with 4 confirmed duplicate groups; this spec
  converts them into a single architectural control layer (`src/prompts/`) with a registry,
  preserving all call-site signatures (`from src.prompts import ...`) and zero behavior
  change for non-duplicate prompts. Sibling of the contract system documented in
  [interfaces.md](../../interfaces.md) — this spec adds the *prompt* contract; no code
  contracts change.

## 1. Problem

The repo currently defines **22 named prompt strings + 1 inline listwise-rank prompt**
across **8 files**, with **4 confirmed duplicate / near-duplicate groups**:

| # | Duplicate group | Files | Severity |
|---|---|-------|---------|
| 1 | `HYDE_PROMPT` ≡ `Q2D_PROMPT` | `src/retrieval/query_transform.py` · `eval/niah_selector_pilot.py` | **identical text** — silent consistency trap if one side is tweaked |
| 2 | `_WRONG_ENTITY_PROMPT` (src) vs `WRONG_ENTITY_PROMPT` + `SEALED_WRONG_ENTITY_PROMPT` (eval) | 3 files | same intent, divergent wording → experiments cannot self-document "which wording fired" |
| 3 | `_EXTRACT_PROMPT` (src) vs `EXTRACT_PROMPT` (eval) | 2 files | near-identical; text drift accumulated |
| 4 | `_PARAMETRIC_PROMPT` (src) vs `PARAMETRIC_PROMPT` (eval) | 2 files | near-identical; text drift accumulated |

The architectural consequence: a prompt tweet is a behavioral change, but a behavioral
change made to *one* of two sibling copies is a **silent evaluation drift** — downstream
metrics move without the experimenter knowing that richer experiments (`eval/`) fired on
a *different* prompt than the production (`src/`) code. This is an integrity risk for a
project whose marks derive from certified-method rigor (see work-plan 2026-07-09
§2 "Definition of done").

The single inline prompt (`RelevanceReranker._rank_window` in `reranker.py:190`) is a
secondary concern — there is only one copy and it lives with its parser, so its
extraction to a named constant is optional; we do it anyway for uniformity.

## 2. Goal

**A single source of truth for every prompt template** in the project. All 22+1 prompts
live in `src/prompts/` and are *imported* by their (many) consumers. The repo's prompt
surface becomes:

- numerable (`PROMPT_REGISTRY` lists every prompt with a stable key + doc)
- reviewable (one file to diff when an experiment result changes)
- sealed-aware (explicit constants — never merged by accident into the base prompt)

**Non-goal:** we do NOT unify sealed variants into one parameterized prompt. Sealed variants
exist deliberately to prevent LLM train/test leakage between split construction phases; their
independent wording is a *feature*. They become separate constants in the registry with
explicit `_SEALED` suffix names.

**Non-goal:** we do NOT change the `LLM.generate` interface, prompt-format conventions, or
any consumer's signature. This is an additive relocation.

**Non-goal:** we do NOT gate prompt access behind a function call (`get_prompt("hyde")`).
Compile-time constant resolution (`from src.prompts import HYDE_PROMPT`) is preferred —
the registry exists for *listing*, not dispatch.

## 3. Architecture

### 3.1 Package layout

```
src/prompts/
├── __init__.py      # PROMPT_REGISTRY + re-export all prompts (the public surface)
├── rag.py           # 5 prompts — RAG generation line
├── retrieval.py    # 4 prompts — retrieval enhancement (HyDE, listwise rank, extract, parametric)
├── niah.py          # 8 prompts — NIAH task construction + distractors (incl. * SEALED variants)
└── judge.py         # 5 prompts — LLM-as-judge / labeling audits
```

Splitting by function (not by file ownership) means the package shape itself explains
*when* each prompt fires. There are four logical phases of an LLM call in this project:
generate an answer (rag), augment retrieval (retrieval), fabricate distractors (niah),
and judge correctness (judge). That is the locus of every prompt — there are no
prompts that cross phases.

### 3.2 Registry shape

```python
# src/prompts/__init__.py
PROMPT_REGISTRY: dict[str, str] = {
  "rag.default":            DEFAULT_RAG_PROMPT,
  "rag.citation":           CITATION_RAG_PROMPT,
  "rag.elicit":             ELICIT_PROMPT,
  "rag.consolidate":        CONSOLIDATE_PROMPT,
  "rag.finalize":           FINALIZE_PROMPT,
  "retrieval.hyde":         HYDE_PROMPT,
  "retrieval.listwise_rank": LISTWISE_RANK_PROMPT,
  "retrieval.extract":      EXTRACT_PROMPT,
  "retrieval.parametric":   PARAMETRIC_PROMPT,
  "niah.wrong_entity":          WRONG_ENTITY_PROMPT,
  "niah.wrong_entity_sealed":   WRONG_ENTITY_PROMPT_SEALED,
  "niah.generative":             GENERATIVE_DISTRACTOR_PROMPT,
  "niah.answerability":          ANSWERABILITY_PROMPT,
  "niah.non_answer":             NON_ANSWER_PROMPT,
  "niah.non_answer_sealed":      NON_ANSWER_PROMPT_SEALED,
  "niah.reliability":            RELIABILITY_PROMPT,
  "niah.label_audit_a":          LABEL_AUDIT_PROMPT_A,
  "niah.label_audit_b":          LABEL_AUDIT_PROMPT_B,
}
```

Keys are dotted (`<phase>.<purpose>`) — this is the public naming contract. Adding a
prompt in the future requires adding a pair (`KEY` constant + `"phase.purpose"` entry),
which is the minimum friction that prevents silent accretion.

The `Q2D_PROMPT` alias is re-exported from `__init__` as `Q2D_PROMPT = HYDE_PROMPT` (no
registry entry — pure backwards-compat alias; the alias must never diverge from HYDE).

### 3.3 Prompt consolidation decisions

For each duplicate group the **authority** (which wording wins) and **legacy-tag policy**:

#### Group 1 — HyDE ≡ Query2Doc
- **Nothing to merge** — both wordings are byte-identical. Authority: `HYDE_PROMPT`
  (named after the published method). `Q2D_PROMPT` becomes a deprecated alias re-exported
  for backwards compatibility; the alias points at the same string.
- No legacy data affected.

#### Group 2 — Wrong-Entity (src) vs Wrong-Entity (eval base) vs Wrong-Entity (sealed)
- Authority for the **base** wording: `src/niah/counterfactual.py::_WRONG_ENTITY_PROMPT`
  (the implementation-line wording; src owns data construction per the contract system).
- The eval-line base wording (`eval/niah_selector_pilot.py::WRONG_ENTITY_PROMPT`) is
  **slightly different in wording**. Selector experiments already run with that wording
  and reported in `results/` — those CSVs are tagged **legacy** in `results-summary.md`:
  "eval/base-wrong-entity wording (pre-centralization)". The re-run uses the unified
  src wording.
- `SEALED_WRONG_ENTITY_PROMPT` is preserved as a separate constant
  `WRONG_ENTITY_PROMPT_SEALED` — same wording as before; deliberate independent variant
  for the test-split construction phase.

#### Group 3 — Extract (src) vs Extract (eval)
- Authority: `src/retrieval/reranker.py::_EXTRACT_PROMPT` (richer wording: "a name,
  place, date, or number" — documents the target answer class, more reproducible). Eval
  legacy CSVs tagged "eval/extract wording (pre-centralization)".

#### Group 4 — Parametric (src) vs Parametric (eval)
- Authority: `src/retrieval/reranker.py::_PARAMETRIC_PROMPT`. Eval legacy CSVs tagged.

### 3.4 The legacy-tagging rule

When wording changes affect already-reported experiments, `docs/results-summary.md` gains
a one-line **legacy note** near each affected table row pair:
> Pre-centralization run used `eval/niah_selector_pilot.py::WRONG_ENTITY_PROMPT`
> wording; replication uses the unified `src/prompts/` constant and may differ marginally.

This converts a silent drift risk into a declared limitation, which is the project's
existing convention (see work-plan WS-1b "kill last tuning caveat" — explicit pre-registration
philosophy).

### 3.5 Call-site invariant

Every consumer file swaps local prompt definitions for an import:

```python
# before
_WRONG_ENTITY_PROMPT = ("...")

# after
from src.prompts.niah import WRONG_ENTITY_PROMPT
```

**No method signature changes.** The local copy is deleted; the import takes its name. The
only name change visible to callers is the prefix drop (`_EXTRACT_PROMPT` → `EXTRACT_PROMPT`)
for the three src-side private copies that previously used the leading underscore.

`AstuteRAGPipeline.ELICIT_PROMPT` keeps its class-attribute name but points at the registry:

```python
from src.prompts.rag import ELICIT_PROMPT as _ELICIT_PROMPT

class AstuteRAGPipeline(RAGPipeline):
    ELICIT_PROMPT = _ELICIT_PROMPT        # semantic association kept; signature zero-change
    CONSOLIDATE_PROMPT = _CONSOLIDATE_PROMPT
    FINALIZE_PROMPT = _FINALIZE_PROMPT
```

This preserves (a) external readers who reach for `AstuteRAG.ELICIT_PROMPT` (semantic anchor)
and (b) any third party (the demo `app/main.py`) reading prompts from the class.

### 3.6 Inline prompt extraction (`RelevanceReranker._rank_window`)

The inline listwise-rank prompt currently in `reranker.py:190` becomes
`LISTWISE_RANK_PROMPT` in `src/prompts/retrieval.py`, with `.format(query=, listing=)`
placeholders. The `_rank_window` method calls `LISTWISE_RANK_PROMPT.format(...)`.
This is uniformity — not deduplication — but it makes the registry complete (every prompt
listed in one place, including listwise rank).

## 4. Backwards compatibility

- `eval/niah_selector_pilot.py` keeps its `Q2D_PROMPT = HYDE_PROMPT` re-export so
  `eval/ramdocs_selector.py` and `eval/financebench_selector.py` (which import
  `Q2D_PROMPT` via `eval.niah_selector_pilot`) keep working unchanged.
- `src/rag_pipeline.py` keeps symbols `DEFAULT_RAG_PROMPT` and `CITATION_RAG_PROMPT`
  importable (re-exported from the new home) because `eval/run_rag.py` imports them
  by name from `src.rag_pipeline`. **However** we opt to **change that one import** to
  `eval/run_rag.py` going `from src.prompts import CITATION_RAG_PROMPT` — a single
  explicit target; the re-export in `rag_pipeline.py` is retained for any external
  consumer.
- All class-attribute pairs (`ELICIT_PROMPT` etc.) on `AstuteRAGPipeline` remain
  accessible on the class for `app/main.py`.
- No file's public function/class signature changes.

## 5. Test plan

- New `tests/test_prompts_registry.py`:
  - `PROMPT_REGISTRY` is non-empty; every value is a `str`.
  - Every registry key is lowercase-dotted (`re.fullmatch(r"[a-z_]+\.[a-z_]+", key)`).
  - Every prompt is non-trivial (len > 0).
  - `*_SEALED` variants and their base pairs are **not equal strings** (regression guard:
    a well-intentioned future "tidy" PR must not collapse a sealed variant by accident).
  - `Q2D_PROMPT is HYDE_PROMPT` (the alias must share identity, not just equal text).
  - All placeholders are recognized `{name}` patterns; no stray `{}` or `{1}` that would
    break `.format()` — verified by attempting `prompt.format(**{name: "" for name in
    discovered_placeholders})` and asserting no `KeyError`/`IndexError`.
- Existing test suite is expected to remain **green where it was green** — the 14
  pre-existing collection errors (env / import) are out of scope for this change and
  will NOT be fixed here. The KPI is "all tests that passed before still pass", verified
  by ignoring the erroring modules via `--ignore`.

## 6. Interfaces contract update

`docs/interfaces.md` Contract 4 (RAG I/O) gains a short "Prompt Registry" sub-section
pointing to `src/prompts/` as the authoritative location and listing the dotted key
naming convention (§3.2). No contract addition — purely a registry reference.

## 7. Risk

- **Behavioral drift on legacy experiments** — addressed by the explicit legacy tagging
  (§3.4). The risk being foreclosed by centralization is *larger*: the drift already
  exists, it is currently silent; making it explicit is a reduction in net risk.
- **`from src.rag_pipeline import CITATION_RAG_PROMPT` external consumers** — a re-export
  is kept on `src.rag_pipeline` so external access (scripts, notebooks) does not break.
- **Refactor blast radius** — only 8 files modified, all under `src/` and `eval/`; every
  change is import-only. The diff is large in line count but small in logic.
- **Spec-to-code drift during implementation** — mitigated by writing the plan
  (sibling file) with task-by-task checkbox steps that mirror this spec section by section.