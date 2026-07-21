# Counterfactual Injector Implementation Plan (Materializer B)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Implement the deterministic counterfactual injector from `docs/superpowers/specs/2026-07-21-counterfactual-injector-design.md`: given a loaded base `DatasetBundle` (dpr-w100 NQ), inject a poisoned-twin counterfactual per eligible query (five invariants) and write the injected dataset plus a `provenance.jsonl` harmful-label sidecar.

**Architecture:** New `src/evidence_rag/materializer/` package. Pure CPU, deterministic, no GPU/network (base bundle and answer bank are inputs). Reuses `answer_norm.canonicalize_answer`, contract models, and `JsonlDatasetAdapter` for output validation. Works at pre-chunked passage granularity — no slicing (spec §11).

**Tech Stack:** Python 3.11, pydantic v2 (frozen models), pytest, mypy strict, ruff. No new deps. Run tests with `$env:PYTHONPATH='src'; python -m pytest`.

---

## File Structure

| File | Responsibility |
|---|---|
| Create `src/evidence_rag/materializer/__init__.py` | package marker |
| Create `src/evidence_rag/materializer/provenance.py` | `MutationRecord` model + jsonl read/write |
| Create `src/evidence_rag/materializer/answer_bank.py` | `string_class` + `AnswerBank` + `build_answer_bank` |
| Create `src/evidence_rag/materializer/injector.py` | eligibility, injection, invariants, materialize + write |
| Create `src/evidence_rag/materializer/cli.py` | `evidence-rag-materialize-niah` entry point |
| Modify `pyproject.toml` | register the CLI script |
| Tests | `tests/materializer/test_provenance.py`, `test_answer_bank.py`, `test_injector.py`, `test_cli.py` |

---

### Task 1: provenance.py — MutationRecord + sidecar IO

**Files:** Create `src/evidence_rag/materializer/__init__.py`, `src/evidence_rag/materializer/provenance.py`; Test `tests/materializer/test_provenance.py`

- [ ] **Step 1.1: Write failing tests**

```python
from pathlib import Path

from evidence_rag.materializer.provenance import MutationRecord, read_provenance, write_provenance


def record(query_id: str = "q1") -> MutationRecord:
    return MutationRecord(
        query_id=query_id,
        needle_document_id="doc-9",
        counterfactual_document_id="cf::q1::doc-9",
        gold_value="18",
        gold_alias_used="18%",
        replacement_value="23",
        string_class="integer",
        seed=42,
        char_span=(10, 13),
        text_hash_before="a" * 64,
        text_hash_after="b" * 64,
        answer_bank_hash="c" * 64,
    )


def test_record_round_trips_through_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "provenance.jsonl"
    write_provenance(path, (record("q1"), record("q2")))
    loaded = read_provenance(path)
    assert tuple(r.query_id for r in loaded) == ("q1", "q2")
    assert loaded[0].char_span == (10, 13)


def test_record_is_frozen_and_rejects_extra_fields() -> None:
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        MutationRecord(query_id="q", extra="x")  # type: ignore[call-arg]
```

- [ ] **Step 1.2: Run to verify failure**

Run: `$env:PYTHONPATH='src'; python -m pytest tests/materializer/test_provenance.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'evidence_rag.materializer'`

- [ ] **Step 1.3: Implement**

Create empty `src/evidence_rag/materializer/__init__.py`:

```python
"""Counterfactual dataset materializer (Selector evaluation data)."""
```

Create `src/evidence_rag/materializer/provenance.py`:

```python
"""Provenance sidecar for injected counterfactuals (spec §6).

One MutationRecord per injected query; the eval harness reads counterfactual_document_id
to compute harmful-in-context. Convention-named jsonl (spec §8 decision 1); no manifest
schema change in v1.
"""

from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class MutationRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    query_id: str
    needle_document_id: str
    counterfactual_document_id: str
    gold_value: str
    gold_alias_used: str
    replacement_value: str
    string_class: str
    seed: int
    char_span: tuple[int, int]
    text_hash_before: str
    text_hash_after: str
    answer_bank_hash: str


def write_provenance(path: Path, records: Iterable[MutationRecord]) -> None:
    lines = [record.model_dump_json() for record in records]
    Path(path).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def read_provenance(path: Path) -> tuple[MutationRecord, ...]:
    text = Path(path).read_text(encoding="utf-8")
    return tuple(
        MutationRecord.model_validate_json(line)
        for line in text.splitlines()
        if line.strip()
    )
```

- [ ] **Step 1.4: Run to verify pass**

Run: `$env:PYTHONPATH='src'; python -m pytest tests/materializer/test_provenance.py -q`
Expected: PASS (2 tests)

- [ ] **Step 1.5: Commit**

```bash
git add src/evidence_rag/materializer/__init__.py src/evidence_rag/materializer/provenance.py tests/materializer/test_provenance.py
git commit -m "feat(materializer): MutationRecord provenance sidecar"
```

### Task 2: answer_bank.py — class detection + deterministic replacement

**Files:** Create `src/evidence_rag/materializer/answer_bank.py`; Test `tests/materializer/test_answer_bank.py`

- [ ] **Step 2.1: Write failing tests**

```python
from evidence_rag.materializer.answer_bank import AnswerBank, build_answer_bank, string_class


def test_string_class_partitions_by_mechanical_type() -> None:
    assert string_class("18") == "integer"
    assert string_class("2025") == "year"
    assert string_class("45.3") == "decimal"
    assert string_class("2019-01-05") == "date"
    assert string_class("Alice") == "name-1"
    assert string_class("New York") == "name-2"
    assert string_class("chocolate") == "noun-1"


def test_select_is_same_class_excludes_gold_and_deterministic() -> None:
    bank = build_answer_bank(["18", "23", "44", "2025", "Alice"], seed=42)
    chosen = bank.select(gold_value="18", gold_aliases=("18%", "18 percent"), string_class="integer")
    assert chosen in {"23", "44"}
    assert bank.select("18", ("18%",), "integer") == chosen  # deterministic


def test_select_blocks_canonical_equivalents() -> None:
    bank = build_answer_bank(["1200000000", "5"], seed=42)
    # gold "$1.2B" canonicalizes to 1200000000, which must be excluded
    assert bank.select("$1.2B", (), "integer") == "5"


def test_select_returns_none_when_no_alternative() -> None:
    bank = build_answer_bank(["18"], seed=42)
    assert bank.select("18", (), "integer") is None


def test_content_hash_is_stable() -> None:
    a = build_answer_bank(["18", "23"], seed=42)
    b = build_answer_bank(["23", "18"], seed=42)
    assert a.content_hash == b.content_hash
    assert build_answer_bank(["18", "23"], seed=7).content_hash != a.content_hash
```

- [ ] **Step 2.2: Run to verify failure**

Run: `$env:PYTHONPATH='src'; python -m pytest tests/materializer/test_answer_bank.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'evidence_rag.materializer.answer_bank'`

- [ ] **Step 2.3: Implement**

```python
"""Frozen, seeded replacement bank (spec §5).

Replacement values are real answer strings from the corpus, grouped by mechanical class,
so a counterfactual is a plausible same-class value. Selection is deterministic
(seed + class + gold value) and excludes anything canonically equal to the gold.
"""

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from hashlib import sha256

from evidence_rag.selector.answer_norm import canonicalize_answer

_INTEGER = re.compile(r"\d+")
_DECIMAL = re.compile(r"\d+\.\d+")
_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def string_class(surface: str) -> str:
    canonical = canonicalize_answer(surface)
    if _ISO_DATE.fullmatch(canonical):
        return "date"
    if _INTEGER.fullmatch(canonical):
        value = int(canonical)
        return "year" if 1000 <= value <= 2999 else "integer"
    if _DECIMAL.fullmatch(canonical):
        return "decimal"
    tokens = surface.split()
    kind = "name" if surface[:1].isupper() else "noun"
    return f"{kind}-{len(tokens)}"


class AnswerBank:
    def __init__(self, by_class: Mapping[str, Iterable[str]], *, seed: int = 42) -> None:
        self._by_class = {key: tuple(sorted(set(values))) for key, values in by_class.items()}
        self._seed = seed
        payload = json.dumps(
            {"seed": seed, "by_class": {k: list(v) for k, v in self._by_class.items()}},
            sort_keys=True,
            ensure_ascii=True,
        )
        self._hash = sha256(payload.encode("utf-8")).hexdigest()

    @property
    def content_hash(self) -> str:
        return self._hash

    def select(
        self,
        gold_value: str,
        gold_aliases: Sequence[str],
        string_class: str,
    ) -> str | None:
        blocked = {canonicalize_answer(gold_value)} | {
            canonicalize_answer(alias) for alias in gold_aliases
        }
        candidates = [
            value
            for value in self._by_class.get(string_class, ())
            if canonicalize_answer(value) not in blocked
        ]
        if not candidates:
            return None
        digest = sha256(f"{self._seed}:{string_class}:{gold_value}".encode("utf-8")).hexdigest()
        return candidates[int(digest, 16) % len(candidates)]


def build_answer_bank(values: Iterable[str], *, seed: int = 42) -> AnswerBank:
    by_class: dict[str, list[str]] = {}
    for value in values:
        by_class.setdefault(string_class(value), []).append(value)
    return AnswerBank(by_class, seed=seed)
```

- [ ] **Step 2.4: Run to verify pass**

Run: `$env:PYTHONPATH='src'; python -m pytest tests/materializer/test_answer_bank.py -q`
Expected: PASS (5 tests)

- [ ] **Step 2.5: Commit**

```bash
git add src/evidence_rag/materializer/answer_bank.py tests/materializer/test_answer_bank.py
git commit -m "feat(materializer): seeded same-class answer bank"
```

### Task 3: injector.py — eligibility + occurrence finding

**Files:** Create `src/evidence_rag/materializer/injector.py`; Test `tests/materializer/test_injector.py`

- [ ] **Step 3.1: Write failing tests**

```python
from evidence_rag.contracts.models import Document, GoldCase
from evidence_rag.materializer.injector import InjectionTarget, find_injection_target


def doc(document_id: str, text: str) -> Document:
    return Document(document_id=document_id, text=text, source_uri=f"fixture://{document_id}")


def test_finds_target_when_alias_appears_exactly_once() -> None:
    documents = {"d1": doc("d1", "The operating margin was 18% last year.")}
    gold = GoldCase(
        query_id="q1",
        relevant_document_ids=("d1",),
        reference_answers=("18%", "18 percent"),
    )
    target = find_injection_target(gold, documents)
    assert isinstance(target, InjectionTarget)
    assert target.needle.document_id == "d1"
    assert target.alias_used == "18%"
    assert target.needle.text[target.span[0] : target.span[1]] == "18%"


def test_rejects_multi_valued_gold() -> None:
    documents = {"d1": doc("d1", "Alice or Bob won.")}
    gold = GoldCase(query_id="q1", relevant_document_ids=("d1",), reference_answers=("Alice", "Bob"))
    assert find_injection_target(gold, documents) is None


def test_rejects_when_alias_appears_more_than_once() -> None:
    documents = {"d1": doc("d1", "18% here and 18% there.")}
    gold = GoldCase(query_id="q1", relevant_document_ids=("d1",), reference_answers=("18%",))
    assert find_injection_target(gold, documents) is None


def test_rejects_when_no_alias_in_any_gold_passage() -> None:
    documents = {"d1": doc("d1", "No number appears here.")}
    gold = GoldCase(query_id="q1", relevant_document_ids=("d1",), reference_answers=("18%",))
    assert find_injection_target(gold, documents) is None
```

- [ ] **Step 3.2: Run to verify failure**

Run: `$env:PYTHONPATH='src'; python -m pytest tests/materializer/test_injector.py -q`
Expected: FAIL — `ImportError: cannot import name 'find_injection_target'`

- [ ] **Step 3.3: Implement (create injector.py with this portion)**

```python
"""Poisoned-twin counterfactual injection (spec §3-§4).

Deterministic: eligible query -> copy the gold passage -> replace the single gold-alias
occurrence with a same-class bank value -> validate invariants -> emit twin Document +
MutationRecord. Passage granularity; no slicing (spec §11).
"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256

from evidence_rag.contracts.models import Document, GoldCase
from evidence_rag.materializer.answer_bank import AnswerBank, string_class
from evidence_rag.materializer.provenance import MutationRecord
from evidence_rag.selector.answer_norm import canonicalize_answer


@dataclass(frozen=True)
class InjectionTarget:
    query_id: str
    needle: Document
    alias_used: str
    span: tuple[int, int]
    gold_value: str
    aliases: tuple[str, ...]


def _occurrences(text: str, alias: str) -> list[tuple[int, int]]:
    pattern = re.compile(rf"(?<!\w){re.escape(alias)}(?!\w)")
    return [(match.start(), match.end()) for match in pattern.finditer(text)]


def _has_any_alias(text: str, aliases: Sequence[str]) -> bool:
    return any(_occurrences(text, alias) for alias in aliases)


def find_injection_target(
    gold_case: GoldCase,
    documents: Mapping[str, Document],
) -> InjectionTarget | None:
    answers = gold_case.reference_answers or ()
    if not answers:
        return None
    normalized = {canonicalize_answer(answer) for answer in answers}
    if len(normalized) != 1:
        return None
    gold_value = next(iter(normalized))
    for document_id in gold_case.relevant_document_ids or ():
        document = documents.get(document_id)
        if document is None:
            continue
        for alias in answers:
            spans = _occurrences(document.text, alias)
            if len(spans) == 1:
                return InjectionTarget(
                    query_id=gold_case.query_id,
                    needle=document,
                    alias_used=alias,
                    span=spans[0],
                    gold_value=gold_value,
                    aliases=tuple(answers),
                )
    return None
```

- [ ] **Step 3.4: Run to verify pass**

Run: `$env:PYTHONPATH='src'; python -m pytest tests/materializer/test_injector.py -q`
Expected: PASS (4 tests)

- [ ] **Step 3.5: Commit**

```bash
git add src/evidence_rag/materializer/injector.py tests/materializer/test_injector.py
git commit -m "feat(materializer): counterfactual eligibility and occurrence finding"
```

### Task 4: injector.py — injection + invariants

**Files:** Modify `src/evidence_rag/materializer/injector.py`; extend `tests/materializer/test_injector.py`

- [ ] **Step 4.1: Write failing tests**

```python
from evidence_rag.materializer.answer_bank import build_answer_bank
from evidence_rag.materializer.injector import inject_counterfactual


def test_injection_swaps_the_single_alias_and_records_provenance() -> None:
    documents = {"d1": doc("d1", "The operating margin was 18% last year.")}
    gold = GoldCase(query_id="q1", relevant_document_ids=("d1",), reference_answers=("18%",))
    target = find_injection_target(gold, documents)
    assert target is not None
    bank = build_answer_bank(["18%", "23", "44"], seed=42)
    result = inject_counterfactual(target, bank, seed=42)
    assert result is not None
    twin, record = result
    assert twin.document_id == "cf::q1::d1"
    assert "18%" not in twin.text
    assert record.replacement_value in {"23", "44"}
    assert record.needle_document_id == "d1"
    assert record.char_span == target.span


def test_injection_is_reversible_by_mutation_log() -> None:
    documents = {"d1": doc("d1", "Margin was 18% overall.")}
    gold = GoldCase(query_id="q1", relevant_document_ids=("d1",), reference_answers=("18%",))
    target = find_injection_target(gold, documents)
    assert target is not None
    bank = build_answer_bank(["18%", "23"], seed=42)
    twin, record = inject_counterfactual(target, bank, seed=42)  # type: ignore[misc]
    start, end = record.char_span
    restored = twin.text[:start] + record.gold_alias_used + twin.text[start + len(record.replacement_value):]
    assert restored == documents["d1"].text


def test_injection_returns_none_when_bank_has_no_same_class_alternative() -> None:
    documents = {"d1": doc("d1", "Margin was 18% overall.")}
    gold = GoldCase(query_id="q1", relevant_document_ids=("d1",), reference_answers=("18%",))
    target = find_injection_target(gold, documents)
    assert target is not None
    bank = build_answer_bank(["Alice"], seed=42)  # only a name, no integer
    assert inject_counterfactual(target, bank, seed=42) is None
```

- [ ] **Step 4.2: Run to verify failure**

Run: `$env:PYTHONPATH='src'; python -m pytest tests/materializer/test_injector.py -q`
Expected: FAIL — `ImportError: cannot import name 'inject_counterfactual'`

- [ ] **Step 4.3: Implement (append to injector.py)**

```python
def inject_counterfactual(
    target: InjectionTarget,
    bank: AnswerBank,
    *,
    seed: int = 42,
) -> tuple[Document, MutationRecord] | None:
    cls = string_class(target.alias_used)
    replacement = bank.select(target.gold_value, target.aliases, cls)
    if replacement is None:
        return None
    start, end = target.span
    new_text = target.needle.text[:start] + replacement + target.needle.text[end:]
    if _has_any_alias(new_text, target.aliases):
        return None
    blocked = {canonicalize_answer(alias) for alias in target.aliases}
    if canonicalize_answer(replacement) in blocked:
        return None
    restored = new_text[:start] + target.alias_used + new_text[start + len(replacement) :]
    if restored != target.needle.text:
        return None
    twin = Document(
        document_id=f"cf::{target.query_id}::{target.needle.document_id}",
        text=new_text,
        source_uri=f"synthetic://cf/{target.query_id}/{target.needle.document_id}",
    )
    record = MutationRecord(
        query_id=target.query_id,
        needle_document_id=target.needle.document_id,
        counterfactual_document_id=twin.document_id,
        gold_value=target.gold_value,
        gold_alias_used=target.alias_used,
        replacement_value=replacement,
        string_class=cls,
        seed=seed,
        char_span=(start, end),
        text_hash_before=sha256(target.needle.text.encode("utf-8")).hexdigest(),
        text_hash_after=sha256(new_text.encode("utf-8")).hexdigest(),
        answer_bank_hash=bank.content_hash,
    )
    return twin, record
```

- [ ] **Step 4.4: Run to verify pass**

Run: `$env:PYTHONPATH='src'; python -m pytest tests/materializer/test_injector.py -q`
Expected: PASS (7 tests)

- [ ] **Step 4.5: Commit**

```bash
git add src/evidence_rag/materializer/injector.py tests/materializer/test_injector.py
git commit -m "feat(materializer): counterfactual injection with invariant guards"
```

### Task 5: injector.py — materialize + write injected dataset

**Files:** Modify `src/evidence_rag/materializer/injector.py`; extend `tests/materializer/test_injector.py`

- [ ] **Step 5.1: Write failing tests**

```python
from pathlib import Path

from evidence_rag.infrastructure.datasets import (
    DatasetBundle,
    DatasetManifest,
    JsonlDatasetAdapter,
)
from evidence_rag.contracts.models import Query
from evidence_rag.materializer.injector import materialize_counterfactuals, write_injected_dataset
from evidence_rag.materializer.provenance import read_provenance


def _bundle() -> DatasetBundle:
    documents = (
        doc("d1", "The operating margin was 18% last year."),
        doc("d2", "An unrelated distractor passage about rain."),
    )
    queries = (Query(query_id="q1", text="operating margin?"),)
    golds = (GoldCase(query_id="q1", relevant_document_ids=("d1",), reference_answers=("18%",)),)
    manifest = DatasetManifest(
        dataset_id="niah/nq",
        dataset_version="v1",
        split="dev",
        documents_file="documents.jsonl",
        queries_file="queries.jsonl",
        gold_cases_file="gold_cases.jsonl",
    )
    return DatasetBundle(
        manifest=manifest,
        dataset_signature="sig",
        documents=documents,
        queries=queries,
        gold_cases=golds,
    )


def test_materialize_produces_twin_and_record() -> None:
    bank = build_answer_bank(["18%", "23", "44"], seed=42)
    result = materialize_counterfactuals(_bundle(), bank, seed=42)
    assert len(result.injected_documents) == 3  # 2 base + 1 twin
    assert len(result.records) == 1
    assert result.records[0].counterfactual_document_id == "cf::q1::d1"


def test_write_injected_dataset_loads_back_through_adapter(tmp_path: Path) -> None:
    bank = build_answer_bank(["18%", "23"], seed=42)
    result = materialize_counterfactuals(_bundle(), bank, seed=42)
    manifest_path = write_injected_dataset(_bundle(), result, tmp_path, seed=42)
    reloaded = JsonlDatasetAdapter.load(manifest_path)
    ids = {document.document_id for document in reloaded.documents}
    assert "cf::q1::d1" in ids
    assert reloaded.manifest.dataset_version == "v1+cf42"
    provenance = read_provenance(tmp_path / "provenance.jsonl")
    assert len(provenance) == 1
```

- [ ] **Step 5.2: Run to verify failure**

Run: `$env:PYTHONPATH='src'; python -m pytest tests/materializer/test_injector.py -q`
Expected: FAIL — `ImportError: cannot import name 'materialize_counterfactuals'`

- [ ] **Step 5.3: Implement (append to injector.py)**

Add imports at the top of injector.py:

```python
import json
from collections.abc import Iterable
from pathlib import Path

from evidence_rag.infrastructure.datasets import DatasetBundle, DatasetManifest
from evidence_rag.materializer.provenance import write_provenance
```

Append:

```python
@dataclass(frozen=True)
class MaterializationResult:
    injected_documents: tuple[Document, ...]
    records: tuple[MutationRecord, ...]
    skipped_query_ids: tuple[str, ...]


def materialize_counterfactuals(
    bundle: DatasetBundle,
    bank: AnswerBank,
    *,
    seed: int = 42,
) -> MaterializationResult:
    documents = {document.document_id: document for document in bundle.documents}
    twins: list[Document] = []
    records: list[MutationRecord] = []
    skipped: list[str] = []
    for gold_case in bundle.gold_cases:
        target = find_injection_target(gold_case, documents)
        injection = inject_counterfactual(target, bank, seed=seed) if target is not None else None
        if injection is None:
            skipped.append(gold_case.query_id)
            continue
        twin, record = injection
        twins.append(twin)
        records.append(record)
    return MaterializationResult(
        injected_documents=tuple(bundle.documents) + tuple(twins),
        records=tuple(records),
        skipped_query_ids=tuple(skipped),
    )


def _write_jsonl(path: Path, records: Iterable[object]) -> None:
    lines = [json.dumps(record, ensure_ascii=True, separators=(",", ":"), sort_keys=True) for record in records]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_injected_dataset(
    bundle: DatasetBundle,
    result: MaterializationResult,
    output_directory: Path,
    *,
    seed: int = 42,
) -> Path:
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    manifest = DatasetManifest(
        dataset_id=bundle.manifest.dataset_id,
        dataset_version=f"{bundle.manifest.dataset_version}+cf{seed}",
        split=bundle.manifest.split,
        documents_file="documents.jsonl",
        queries_file="queries.jsonl",
        gold_cases_file="gold_cases.jsonl",
    )
    manifest_path = output_directory / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest.model_dump(mode="json"), ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_jsonl(
        output_directory / manifest.documents_file,
        (document.model_dump(mode="json") for document in result.injected_documents),
    )
    _write_jsonl(
        output_directory / manifest.queries_file,
        (query.model_dump(mode="json") for query in bundle.queries),
    )
    _write_jsonl(
        output_directory / manifest.gold_cases_file,
        (gold_case.model_dump(mode="json") for gold_case in bundle.gold_cases),
    )
    write_provenance(output_directory / "provenance.jsonl", result.records)
    return manifest_path
```

- [ ] **Step 5.4: Run to verify pass**

Run: `$env:PYTHONPATH='src'; python -m pytest tests/materializer/test_injector.py -q`
Expected: PASS (9 tests)

- [ ] **Step 5.5: Commit**

```bash
git add src/evidence_rag/materializer/injector.py tests/materializer/test_injector.py
git commit -m "feat(materializer): materialize and write injected dataset with provenance"
```

### Task 6: cli.py + entry point

**Files:** Create `src/evidence_rag/materializer/cli.py`; Modify `pyproject.toml:34-37`; Test `tests/materializer/test_cli.py`

- [ ] **Step 6.1: Write failing test**

```python
import json
from pathlib import Path

from evidence_rag.contracts.models import Document, Query
from evidence_rag.infrastructure.datasets import DatasetManifest, JsonlDatasetAdapter
from evidence_rag.materializer.cli import main


def _write_base(tmp_path: Path) -> Path:
    manifest = DatasetManifest(
        dataset_id="niah/nq",
        dataset_version="v1",
        split="dev",
        documents_file="documents.jsonl",
        queries_file="queries.jsonl",
        gold_cases_file="gold_cases.jsonl",
    )
    (tmp_path / "manifest.json").write_text(manifest.model_dump_json(), encoding="utf-8")
    (tmp_path / "documents.jsonl").write_text(
        Document(document_id="d1", text="Margin was 18% overall.", source_uri="x://d1").model_dump_json()
        + "\n"
        + Document(document_id="d2", text="Revenue was 23 in 2020.", source_uri="x://d2").model_dump_json()
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "queries.jsonl").write_text(
        Query(query_id="q1", text="margin?").model_dump_json() + "\n", encoding="utf-8"
    )
    from evidence_rag.infrastructure.datasets import GoldCase

    (tmp_path / "gold_cases.jsonl").write_text(
        GoldCase(query_id="q1", relevant_document_ids=("d1",), reference_answers=("18%",)).model_dump_json()
        + "\n",
        encoding="utf-8",
    )
    return tmp_path / "manifest.json"


def test_cli_materializes_and_validates(tmp_path: Path) -> None:
    base = _write_base(tmp_path)
    out = tmp_path / "injected"
    exit_code = main(("--base-manifest", str(base), "--output", str(out), "--seed", "42"))
    assert exit_code == 0
    reloaded = JsonlDatasetAdapter.load(out / "manifest.json")
    assert any(d.document_id == "cf::q1::d1" for d in reloaded.documents)
```

- [ ] **Step 6.2: Run to verify failure**

Run: `$env:PYTHONPATH='src'; python -m pytest tests/materializer/test_cli.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'evidence_rag.materializer.cli'`

- [ ] **Step 6.3: Implement cli.py**

The answer bank is built from all gold answers in the base dataset (real same-class values).

```python
"""CLI: materialize a counterfactual-injected dataset from a base manifest (spec §9)."""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from evidence_rag.infrastructure.datasets import JsonlDatasetAdapter
from evidence_rag.materializer.answer_bank import build_answer_bank
from evidence_rag.materializer.injector import materialize_counterfactuals, write_injected_dataset


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inject counterfactual twins into a base dataset")
    parser.add_argument("--base-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    bundle = JsonlDatasetAdapter.load(arguments.base_manifest)
    answers = [
        answer
        for gold_case in bundle.gold_cases
        for answer in (gold_case.reference_answers or ())
    ]
    bank = build_answer_bank(answers, seed=arguments.seed)
    result = materialize_counterfactuals(bundle, bank, seed=arguments.seed)
    manifest_path = write_injected_dataset(bundle, result, arguments.output, seed=arguments.seed)
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "injected": len(result.records),
                "skipped": len(result.skipped_query_ids),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6.4: Register the entry point in pyproject.toml**

In `[project.scripts]` (currently lines 34-37) add:

```toml
evidence-rag-materialize-niah = "evidence_rag.materializer.cli:main"
```

- [ ] **Step 6.5: Run to verify pass, full suite, typecheck, ruff**

Run:
```
$env:PYTHONPATH='src'; python -m pytest tests/materializer -q
$env:PYTHONPATH='src'; python -m pytest -q
$env:MYPYPATH='src'; python -m mypy
python -m ruff check src/evidence_rag/materializer tests/materializer
```
Expected: materializer tests pass; full suite parity (no new failures vs baseline); mypy clean; ruff clean.

- [ ] **Step 6.6: Commit**

```bash
git add src/evidence_rag/materializer/cli.py pyproject.toml tests/materializer/test_cli.py
git commit -m "feat(materializer): CLI entry point evidence-rag-materialize-niah"
```

---

## Experiment marking (running-hpc-experiments)

The injector is CPU-only and runs on the login node (no slurm). The dataset it produces feeds E2/E3 (spec-blocked on the base loader A + the §14 dataset decision). No HPC triple in this plan; when the base loader A lands and produces a real dpr-w100 NQ base, running the injector + `evidence-rag-experiment` E2 is a login-node CPU step recorded in `docs/hpc-run-log.md` under E2.

## Self-review notes

- Spec coverage: §3 eligibility→Task 3; §4 injection+invariants→Task 4; §5 answer bank→Task 2; §6 MutationRecord/provenance→Task 1; §2 output + §manifest `+cf<seed>`→Task 5; §9 CLI→Task 6; §11 no-slicing honored (passage-level Documents, no chunker touched). Base loader A is out of scope by spec §0 (sibling).
- No placeholders: every code step is complete; `string_class` bucket rule is concrete (Task 2), resolving spec §12.4's "impl detail".
- Type consistency: `MutationRecord` fields identical in Tasks 1/4; `InjectionTarget`/`inject_counterfactual`/`materialize_counterfactuals`/`write_injected_dataset` signatures consistent across Tasks 3-6; `build_answer_bank`/`AnswerBank.select` used identically in Tasks 2/4/6.
- Determinism: bank selection seeded by sha256(seed:class:gold); cf id encodes query_id (no cross-query collision); JsonlDatasetAdapter round-trip is the integration guard (Task 5/6).
