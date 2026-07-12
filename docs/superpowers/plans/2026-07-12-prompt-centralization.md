# Prompt Centralization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Relocate every prompt template string (22 named + 1 inline) into a single
`src/prompts/` package, eliminate 4 duplicate groups, expose a `PROMPT_REGISTRY`, and keep
all consumer files at zero behavioral change except for the deliberate wording
unification described in `docs/superpowers/specs/2026-07-12-prompt-centralization-design.md` §3.3.

**Architecture:** New `src/prompts/` package (`__init__.py` + 4 phase files). The 8
consumer files swap local definitions for imports. One new test file. Nothing else is
touched. Spec: `docs/superpowers/specs/2026-07-12-prompt-centralization-design.md`.

**Tech Stack:** Python 3.10+ stdlib (`re`, `typing`), pytest. No new dependencies.

---

## File Structure

- **Create** `src/prompts/__init__.py` — `PROMPT_REGISTRY` + re-exports
- **Create** `src/prompts/rag.py` — 5 RAG generation prompts
- **Create** `src/prompts/retrieval.py` — 4 retrieval-enhancement prompts
- **Create** `src/prompts/niah.py` — 8 NIAH construction/judge prompts (incl. sealed variants)
- **Create** `src/prompts/judge.py` — 5 LLM-as-judge / labeling prompts
- **Create** `tests/test_prompts_registry.py` — registry & invariant tests
- **Modify** `src/rag_pipeline.py` — replace 2 local prompts with imports; keep `AstuteRAGPipeline.ELICIT_PROMPT` etc. as class attrs pointing at the registry
- **Modify** `src/retrieval/reranker.py` — extract inline listwise rank prompt; replace `_EXTRACT_PROMPT` / `_PARAMETRIC_PROMPT` with imports
- **Modify** `src/retrieval/query_transform.py` — replace local `HYDE_PROMPT` with import
- **Modify** `src/niah/counterfactual.py` — replace local `_WRONG_ENTITY_PROMPT` with import
- **Modify** `src/niah/filters.py` — replace local `_ANSWERABILITY_PROMPT` with import
- **Modify** `src/niah/generative.py` — replace local `_GENERATIVE_PROMPT` with import
- **Modify** `eval/niah_selector_pilot.py` — replace 8 local prompts with imports; keep `Q2D_PROMPT` as alias re-export
- **Modify** `eval/niah_label_audit.py` — replace 2 local prompts with imports
- **Modify** `eval/run_rag.py` — change `CITATION_RAG_PROMPT` import source (still works either way)
- **Modify** `docs/interfaces.md` — Contract 4 + Prompt Registry subsection
- **Modify** `docs/dev-log.md` — Changelog row

Run tests with:
```bash
python -m pytest tests/test_prompts_registry.py -v
python -m pytest --ignore=tests/test_strong_bm25.py --ignore=tests/test_tune_alpha.py \
                --ignore=tests/test_tune_corroboration.py --ignore=tests/test_retrieval_bm25.py \
                --ignore=tests/test_run_benchmark.py --ignore=tests/test_run_niah.py \
                --ignore=tests/test_run_niah_rag.py --ignore=tests/test_run_rag.py \
                --ignore=tests/test_sparse_index.py --ignore=tests/test_sparse_retriever.py \
                --ignore=tests/test_splade_encoder.py -q
```
(The 14 pre-existing collection errors are environment-related, out of scope for this plan — §5 of the spec.)

---

## Task 1: Create the `src/prompts/` package skeleton + registry tests (TDD)

**Files:**
- Create: `src/prompts/rag.py` (placeholders — exact copies of existing prompts)
- Create: `src/prompts/retrieval.py` (placeholders)
- Create: `src/prompts/niah.py` (placeholders)
- Create: `src/prompts/judge.py` (placeholders)
- Create: `src/prompts/__init__.py` (PROMPT_REGISTRY + re-exports)
- Create: `tests/test_prompts_registry.py`

- [ ] **Step 1: Write the failing tests first**

Create `tests/test_prompts_registry.py`:

```python
"""Invariant tests for the central prompt registry. See
docs/superpowers/specs/2026-07-12-prompt-centralization-design.md."""
from __future__ import annotations

import re

import pytest

from src.prompts import PROMPT_REGISTRY
from src.prompts import (
    DEFAULT_RAG_PROMPT,
    CITATION_RAG_PROMPT,
    HYDE_PROMPT,
    Q2D_PROMPT,
    WRONG_ENTITY_PROMPT,
    WRONG_ENTITY_PROMPT_SEALED,
    NON_ANSWER_PROMPT,
    NON_ANSWER_PROMPT_SEALED,
)


def test_registry_is_non_empty():
    assert len(PROMPT_REGISTRY) >= 18


def test_every_value_is_a_non_empty_string():
    for key, prompt in PROMPT_REGISTRY.items():
        assert isinstance(prompt, str), f"{key}: not a str"
        assert prompt.strip(), f"{key}: empty prompt"


def test_registry_keys_are_dotted_phase_purpose():
    pat = re.compile(r"^[a-z_]+\.[a-z_]+$")
    for key in PROMPT_REGISTRY:
        assert pat.match(key), f"bad key shape: {key!r}"


def test_sealed_variants_differ_from_base():
    assert WRONG_ENTITY_PROMPT_SEALED != WRONG_ENTITY_PROMPT
    assert NON_ANSWER_PROMPT_SEALED != NON_ANSWER_PROMPT


def test_q2d_alias_shares_identity_with_hyde():
    assert Q2D_PROMPT is HYDE_PROMPT


def _placeholders(prompt: str) -> list[str]:
    return re.findall(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}", prompt)


def test_every_prompt_formats_with_its_placeholders():
    for key, prompt in PROMPT_REGISTRY.items():
        names = _placeholders(prompt)
        kwargs = {name: "X" for name in names}
        prompt.format(**kwargs)  # must not raise


def test_citation_prompt_mentions_evidence_format():
    assert "Evidence:" in CITATION_RAG_PROMPT
    assert "Answer:" in CITATION_RAG_PROMPT
```

- [ ] **Step 2: Run the tests — expect ImportError (no `src/prompts/` yet)**

```bash
python -m pytest tests/test_prompts_registry.py -v
```

- [ ] **Step 3: Create the 5 package files**

Copy the exact text of each prompt from its current home into the corresponding phase file.
Re-name private copies: drop the leading underscore (`_EXTRACT_PROMPT` → `EXTRACT_PROMPT`)
so the registry key is the public name.

`src/prompts/rag.py` — copy `DEFAULT_RAG_PROMPT`, `CITATION_RAG_PROMPT`, `ELICIT_PROMPT`,
`CONSOLIDATE_PROMPT`, `FINALIZE_PROMPT` verbatim from `src/rag_pipeline.py`.

`src/prompts/retrieval.py` — copy `HYDE_PROMPT` from `src/retrieval/query_transform.py`;
copy `_EXTRACT_PROMPT`/`_PARAMETRIC_PROMPT` verbatim from `src/retrieval/reranker.py`
(re-named `EXTRACT_PROMPT`/`PARAMETRIC_PROMPT`); extract the inline listwise prompt from
`reranker.py:190` into `LISTWISE_RANK_PROMPT` with `{query}`/`{listing}`/`{n}` placeholders.

`src/prompts/niah.py` — copy 8 prompts:
- `WRONG_ENTITY_PROMPT` ← `src/niah/counterfactual.py::_WRONG_ENTITY_PROMPT` (the **src** wording is authoritative per §3.3 group 2)
- `WRONG_ENTITY_PROMPT_SEALED` ← `eval/niah_selector_pilot.py::SEALED_WRONG_ENTITY_PROMPT`
- `GENERATIVE_DISTRACTOR_PROMPT` ← `src/niah/generative.py::_GENERATIVE_PROMPT`
- `ANSWERABILITY_PROMPT` ← `src/niah/filters.py::_ANSWERABILITY_PROMPT`
- `NON_ANSWER_PROMPT` ← `eval/niah_selector_pilot.py::NON_ANSWER_PROMPT`
- `NON_ANSWER_PROMPT_SEALED` ← `eval/niah_selector_pilot.py::SEALED_NON_ANSWER_PROMPT`
- `RELIABILITY_PROMPT` ← `eval/niah_selector_pilot.py::RELIABILITY_PROMPT`
- `LABEL_AUDIT_PROMPT_A`, `LABEL_AUDIT_PROMPT_B` ← `eval/niah_label_audit.py`

`src/prompts/judge.py` — Reduces to a re-export of `RELIABILITY_PROMPT` and
`LABEL_AUDIT_PROMPT_A/B` from `niah.py` if you judge that sitting these in `niah.py` is
fine; the spec splits them per phase — keep §3.1 layout (5 in judge.py) so the file
purposely holds the LLM-as-judge wording: `RELIABILITY_PROMPT`, `LABEL_AUDIT_PROMPT_A`,
`LABEL_AUDIT_PROMPT_B`, plus any future judge prompts without disturbing `niah.py`.

Revisit: the cleaner cut is judge.py holds the three judge prompts to keep the file
purposeful. Place `RELIABILITY_PROMPT` is technically a `niah`-use prompt (judge inside
the NIAH selector loop) but conceptually *judges*; spec §3.1 places it under judge. Plan
follows the spec — relocate to `judge.py`. `niah.py` will import it from `judge.py` for
re-export convenience if needed.

Final word: keep the spec's split strict — judge.py has 3 prompts, niah.py has 5 (wrong
entity x2, generative, answerability, non-answer x2). The 22-prompt count of the spec is
approximate; the registry test asserts `>= 18` precisely because of this ±3 ambiguity.

`src/prompts/__init__.py` — re-export all and define `PROMPT_REGISTRY` exactly as in spec §3.2.
Add `Q2D_PROMPT = HYDE_PROMPT` as an explicit identity alias.

- [ ] **Step 4: Run the new tests — must pass**

```bash
python -m pytest tests/test_prompts_registry.py -v
```

All 7 tests green. If any fails: re-read the spec §3.2 (key naming) and §3.1 (file split).

---

## Task 2: Swap consumers — `src/` files

Order: leaf modules first (niah), then mid-tier (retrieval, rag_pipeline), so each swap
is independently verifiable by importing the file.

- [ ] **Step 1: `src/niah/counterfactual.py`**

Delete `_WRONG_ENTITY_PROMPT = ( ... )`. Add at top, under other imports:
```python
from src.prompts.niah import WRONG_ENTITY_PROMPT
```
In `propose_wrong_entity`, replace `_WRONG_ENTITY_PROMPT.format(answer=answer)` with
`WRONG_ENTITY_PROMPT.format(answer=answer)`.

- [ ] **Step 2: `src/niah/generative.py`**

Delete `_GENERATIVE_PROMPT`. Add `from src.prompts.niah import GENERATIVE_DISTRACTOR_PROMPT`.
Replace call site.

- [ ] **Step 3: `src/niah/filters.py`**

Delete `_ANSWERABILITY_PROMPT`. Add `from src.prompts.niah import ANSWERABILITY_PROMPT`.
Replace call site.

- [ ] **Step 4: `src/retrieval/query_transform.py`**

Delete `HYDE_PROMPT = ( ... )`. Add `from src.prompts.retrieval import HYDE_PROMPT`.
The default-arg `template: str = HYDE_PROMPT` keeps working (the imported constant fills
the default at function-def time, same as before).

- [ ] **Step 5: `src/retrieval/reranker.py`**

- Delete `_EXTRACT_PROMPT` and `_PARAMETRIC_PROMPT` definitions (lines 219-231).
- Add `from src.prompts.retrieval import EXTRACT_PROMPT, PARAMETRIC_PROMPT, LISTWISE_RANK_PROMPT`.
- In `_extract_answer` body: `_EXTRACT_PROMPT` → `EXTRACT_PROMPT`.
- In `_parametric_answer` body: `_PARAMETRIC_PROMPT` → `PARAMETRIC_PROMPT`.
- In `_rank_window` body: replace the inline f-string triple-quoted prompt with
  `LISTWISE_RANK_PROMPT.format(query=query, listing=listing, n=len(docs))`.
  (The named constant must use `{query}`/`{listing}`/`{n}` placeholders, not f-string —
  see Step 3 of Task 1.)

- [ ] **Step 6: `src/rag_pipeline.py`**

This is the most delicate file because it owns 5 prompts and the new citation machinery
that exists only in the working tree. Steps:

- Delete the local `DEFAULT_RAG_PROMPT` block and the `CITATION_RAG_PROMPT` block.
- Delete the body of the `AstuteRAGPipeline.ELICIT_PROMPT`/`CONSOLIDATE_PROMPT`/`FINALIZE_PROMPT`
  class attributes; replace with imports bound to the class at module level (see below).
- Add to imports at top: `from src.prompts.rag import (DEFAULT_RAG_PROMPT, CITATION_RAG_PROMPT, ELICIT_PROMPT, CONSOLIDATE_PROMPT, FINALIZE_PROMPT)`.
- **Backwards-compat re-export**: keep `DEFAULT_RAG_PROMPT` and `CITATION_RAG_PROMPT`
  re-exportable from `src.rag_pipeline` (the import statements do this implicitly — they
  are now module-level names of `src.rag_pipeline`). External callers
  (`eval/run_rag.py`, notebooks) that do `from src.rag_pipeline import CITATION_RAG_PROMPT`
  keep working without a code change. **Optionally**, change `eval/run_rag.py` to import
  directly from `src.prompts` — minor cleanup, deferred to Task 3 step 3.
- `AstuteRAGPipeline` class body — keep three class attributes but assign from the
  imported registry constants:

```python
class AstuteRAGPipeline(RAGPipeline):
    ELICIT_PROMPT = _prompts.ELICIT_PROMPT        # class-level anchor — see spec §3.5
    CONSOLIDATE_PROMPT = _prompts.CONSOLIDATE_PROMPT
    FINALIZE_PROMPT = _prompts.FINALIZE_PROMPT
```
(Use `import src.prompts.rag as _prompts` so the class body reads cleanly without
clashing with the same-name imports at module top.)

- Verify nothing else in the file references the deleted local constants (grep
  `DEFAULT_RAG_PROMPT\|CITATION_RAG_PROMPT\|ELICIT_PROMPT\|CONSOLIDATE_PROMPT\|FINALIZE_PROMPT`
  — every match must be either an import or an `AstuteRAGPipeline.<X>` class attribute, OR
  the `if self.prompt_template == CITATION_RAG_PROMPT:` check in `query()` /
  `CorrectiveRAGPipeline.query()` — which now uses `==` rather than `is` so the
  citation path stays active even if the prompt object is loaded from JSON/YAML
  instead of imported; `==` on strings is by text, not object identity).

- [ ] **Step 7: smoke-import all modified `src/` files**

```bash
python -c "import src.niah.counterfactual, src.niah.generative, src.niah.filters, \
           src.retrieval.query_transform, src.retrieval.reranker, src.rag_pipeline; \
           print('all src imports ok')"
```

- [ ] **Step 8: harden the citation-path switch (`is` → `==`)**

`src/rag_pipeline.py` has two sites that activate the model-self-citation path by
*identity* comparison:

* `RAGPipeline.query()` (~line 278): `if self.prompt_template is CITATION_RAG_PROMPT:`
* `CorrectiveRAGPipeline.query()` (~line 376): `if self.prompt_template is CITATION_RAG_PROMPT:`

Both must change `is` → `==`. Rationale: `is` only resolves True when the two operands
are the *same interned object*; a prompt loaded from JSON/YAML in a future config-loader
refactor would text-equal `CITATION_RAG_PROMPT` but fail `is`, silently degrading the
citation path to the DEFAULT (post-hoc attribution) path — `evaluate_rag` would then
receive no `citation_*` metrics with no error raised. `==` on `str` compares by text,
so the switch is robust to object provenance while still behaving identically for the
current import-based flow (all callers fetch the same object from `src.prompts`, so
`==` returns True just as `is` did).

Verify after edit:
```bash
rg -n "prompt_template is CITATION_RAG_PROMPT" src/rag_pipeline.py
# should return nothing
rg -n "prompt_template == CITATION_RAG_PROMPT" src/rag_pipeline.py
# should return 2 lines
```

---

## Task 3: Swap consumers — `eval/` files

- [ ] **Step 1: `eval/niah_label_audit.py`**

Delete `PROMPT_A = ( ... )` and `PROMPT_B = ( ... )`. Add:
```python
from src.prompts.judge import LABEL_AUDIT_PROMPT_A, LABEL_AUDIT_PROMPT_B
```
Replace call sites: `PROMPT_A.format(...)` → `LABEL_AUDIT_PROMPT_A.format(...)`, likewise B.

- [ ] **Step 2: `eval/niah_selector_pilot.py`**

This is the biggest single change — 8 prompt definitions to replace.

- Delete all 8: `Q2D_PROMPT`, `WRONG_ENTITY_PROMPT`, `SEALED_WRONG_ENTITY_PROMPT`,
  `NON_ANSWER_PROMPT`, `SEALED_NON_ANSWER_PROMPT`, `EXTRACT_PROMPT`, `PARAMETRIC_PROMPT`,
  `RELIABILITY_PROMPT`.
- Add imports:
```python
from src.prompts.retrieval import HYDE_PROMPT, EXTRACT_PROMPT, PARAMETRIC_PROMPT
from src.prompts.niah import (
    WRONG_ENTITY_PROMPT, WRONG_ENTITY_PROMPT_SEALED,
    NON_ANSWER_PROMPT, NON_ANSWER_PROMPT_SEALED,
)
from src.prompts.judge import RELIABILITY_PROMPT
```
- Add a backwards-compat alias at the bottom of the import block:
  `Q2D_PROMPT = HYDE_PROMPT` — this preserves the public name used by sibling eval
  files (`eval/ramdocs_selector.py`, `eval/financebench_selector.py` import this). The
  semantic-intent registry entry remains `retrieval.hyde`; `Q2D_PROMPT` is a deprecated
  surface alias, NOT a registry key (already encoded in spec §3.2).
- Audit all call sites in the file: replace `Q2D_PROMPT.format(...)` references — they
  can stay as `Q2D_PROMPT` (the alias works) OR be migrated to `HYDE_PROMPT` for
  clarity. Plan choice: **migrate to `HYDE_PROMPT`** in the call sites within this file so
  the deprecation is visible to the next reader. The alias survives for external callers only.
- Replace `SEALED_WRONG_ENTITY_PROMPT` → `WRONG_ENTITY_PROMPT_SEALED`,
  `SEALED_NON_ANSWER_PROMPT` → `NON_ANSWER_PROMPT_SEALED`. (Two name changes — flagged
  in spec §3.3.)
- All other prompts (`WRONG_ENTITY_PROMPT`, `NON_ANSWER_PROMPT`, `EXTRACT_PROMPT`,
  `PARAMETRIC_PROMPT`, `RELIABILITY_PROMPT`) keep their names.

- [ ] **Step 3: `eval/run_rag.py`**

Currently imports `CITATION_RAG_PROMPT` from `src.rag_pipeline`. Since `src.rag_pipeline`
re-exports it (Task 2 Step 6), this continues to work. **However**, the spec wants this
file to import directly from the registry for clarity. Change:
```python
# before
from src.rag_pipeline import CITATION_RAG_PROMPT
# after
from src.prompts import CITATION_RAG_PROMPT
```
The two imports yield the same object identity, so the `is` check in
`RAGPipeline.query()` still holds.

- [ ] **Step 4: smoke-import the modified `eval/` files**

```bash
python -c "import eval.niah_label_audit, eval.niah_selector_pilot, eval.run_rag; \
           print('all eval imports ok')"
```

If `eval/niah_selector_pilot.py` import fails with `ModuleNotFoundError` from
pandas/torch — that's outside this plan's scope (pre-existing env). Verify instead via
`python -c "from eval.niah_selector_pilot import Q2D_PROMPT; print(Q2D_PROMPT[:30])"` or
skip if its deps aren't present.

---

## Task 4: Update docs

- [ ] **Step 1: `docs/interfaces.md` — Contract 4 add "Prompt Registry" subsection**

Append to the Contract 4 section (before the separator that starts Contract 5):

```markdown
### 4b — Prompt Registry (新增,2026-07-12)

所有 prompt 模板的**唯一权威位置**:`src/prompts/`
([`__init__.py`](../src/prompts/__init__.py) 暴露 `PROMPT_REGISTRY` + re-export)。
按 phase 分 4 个子文件:`rag.py` / `retrieval.py` / `niah.py` / `judge.py`。

约定:
- 新增 prompt = 加一个常量 + 一条 `PROMPT_REGISTRY` 条目,key 形如 `"<phase>.<purpose>"`
  (lowercase, dot-separated)。
- 不允许在 `src/` 或 `eval/` 的其他文件里再定义 prompt 字符串常量 —— 否则
  `tests/test_prompts_registry.py` 不会捕获它,但 review 时应拒绝。
- `*_SEALED` 变体是**独立的常量**(非参数化的同一模板),用于 train/test split
  的 prompt 措辞隔离,严禁合并。
- 设计 spec:`docs/superpowers/specs/2026-07-12-prompt-centralization-design.md`。
```

- [ ] **Step 2: `docs/dev-log.md` — Changelog add row**

Add to the Changelog table (top-most row):

```
| 2026-07-12 | refactor/prompts | **Prompt 集中化**:新建 `src/prompts/` 包(rag/retrieval/niah/judge 4 phase + `PROMPT_REGISTRY`);消除 4 组重复 prompt(HyDE≡Q2D、WRONG_ENTITY、EXTRACT、PARAMETRIC);8 个 consuming file 改 import;新增 `tests/test_prompts_registry.py` 7 测试。sealed 变体保留为独立常量;eval 旧措辞实验标 legacy。spec:`docs/superpowers/specs/2026-07-12-prompt-centralization-design.md` | `src/prompts/*`, `src/rag_pipeline.py`, `src/retrieval/{reranker,query_transform}.py`, `src/niah/{counterfactual,filters,generative}.py`, `eval/{niah_selector_pilot,niah_label_audit,run_rag}.py`, `tests/test_prompts_registry.py`, `docs/interfaces.md` | _TBD_ |
```

The owner is left as `_TBD_` per instruction (to be filled when assigned).

---

## Task 5: Test-suite regression check

- [ ] **Step 1: Run the new registry tests**

```bash
python -m pytest tests/test_prompts_registry.py -v
```
Expect: 7 passed.

- [ ] **Step 2: Run the unaffected, importable tests**

```bash
python -m pytest --ignore=tests/test_strong_bm25.py --ignore=tests/test_tune_alpha.py \
                --ignore=tests/test_tune_corroboration.py --ignore=tests/test_retrieval_bm25.py \
                --ignore=tests/test_run_benchmark.py --ignore=tests/test_run_niah.py \
                --ignore=tests/test_run_niah_rag.py --ignore=tests/test_run_rag.py \
                --ignore=tests/test_sparse_index.py --ignore=tests/test_sparse_retriever.py \
                --ignore=tests/test_splade_encoder.py -q
```
Goal: no NEW failures vs the pre-refactor baseline (the 14 errors are environmental,
out of scope; what matters is they are still errors at the same count — any new
error/failure means a regression and must be fixed before declaring done).

- [ ] **Step 3: Final smoke `--collect-only` on the touched test files**

```bash
python -m pytest tests/test_prompts_registry.py tests/test_rag_pipeline.py \
                 tests/test_citations.py tests/test_data_structures.py --collect-only -q
```
All four must collect cleanly; if `test_rag_pipeline.py` is among the 14 errors
pre-existing, ignore it only if its error is the same `ModuleNotFoundError` it had
pre-refactor (verify the diff in errors is empty before/after).

---

## Task 6: Self-review checklist

- [ ] `git grep -nP '(?<!\\)\\b[A-Z_]*_PROMPT[A-Z_]*\\s*=' -- 'src/**' 'eval/**' | grep -v 'src/prompts/'`
  should return zero newly-defined prompts outside `src/prompts/`. (Aliasing
  `Q2D_PROMPT = HYDE_PROMPT` is one explicit exception and must be the only one.)
- [ ] `git grep -nP '_prompt\\s*='` — likewise.
- [ ] `python -c "from src.prompts import PROMPT_REGISTRY; print(len(PROMPT_REGISTRY))"`
  — the number matches what the spec says (≥18).
- [ ] The diff of `src/rag_pipeline.py` shows lines removed (the local prompt strings)
  and a few imports added; the body of `query` / `CorrectiveRAGPipeline.query` / `ask`
  / `_build_result` is byte-identical to pre-refactor.
- [ ] The diff of `eval/run_rag.py` shows only one line changed (the import source).
- [ ] No tests deleted; no test files renamed.

---

## Out of scope

- The 14 pre-existing pytest collection errors (installed env / torch / pandas)
  — fixing these is a separate task, not part of this centralization.
- Migrating `eval/ramdocs_selector.py` / `eval/financebench_selector.py` from
  `from eval.niah_selector_pilot import Q2D_PROMPT` to `from src.prompts import HYDE_PROMPT`
  — they continue working with the alias preserved; the change is cosmetic and kept
  for a future cleanup PR.
- The legacy experiment re-runs that §3.4 of the spec mandates tagging
  (`docs/results-summary.md`) — the tagging is a doc edit that the experiment owner
  should make when re-running those cells; not part of this code refactor.