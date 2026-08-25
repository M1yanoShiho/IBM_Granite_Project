# NIAH Base Loader Implementation Plan (Materializer A)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Implement the dpr-w100 NQ base loader from `docs/superpowers/specs/2026-07-21-niah-base-loader-design.md`: materialize `dpr-w100/natural-questions` into the project dataset format with qrels-aware corpus subsampling, so the injector B has a base dataset to poison.

**Architecture:** New `src/evidence_rag/materializer/base_loader.py` mirroring `infrastructure/benchmarks.py` (provider protocol + materialize function) plus reservoir subsampling so the 21M-passage corpus never loads whole. Pure CPU, deterministic, provider-injectable (fake in tests; ir_datasets in production). Output validated by `JsonlDatasetAdapter.load`. No slicing — dpr-w100 is pre-chunked.

**Tech Stack:** Python 3.11, pydantic v2, pytest, mypy strict, ruff. No new deps (stdlib `random`; `ir_datasets` only at production CLI runtime). Run tests with `$env:PYTHONPATH='src'; python -m pytest`.

---

## File Structure

| File | Responsibility |
|---|---|
| Create `src/evidence_rag/materializer/base_loader.py` | provider protocols + `materialize_niah_base` + qrels-aware subsample |
| Create `src/evidence_rag/materializer/base_cli.py` | `evidence-rag-load-niah-base` (ir_datasets provider) |
| Modify `pyproject.toml` | register CLI |
| Tests | `tests/materializer/test_base_loader.py` |

---

### Task 1: base_loader.py — materialize + qrels-aware subsample

**Files:** Create `src/evidence_rag/materializer/base_loader.py`; Test `tests/materializer/test_base_loader.py`

- [ ] **Step 1.1: Write failing tests**

```python
from pathlib import Path

from evidence_rag.infrastructure.datasets import JsonlDatasetAdapter
from evidence_rag.materializer.base_loader import materialize_niah_base


class FakeDoc:
    def __init__(self, doc_id: str, text: str, title: str = "") -> None:
        self.doc_id = doc_id
        self.text = text
        self.title = title


class FakeQuery:
    def __init__(self, query_id: str, text: str, answers: tuple[str, ...]) -> None:
        self.query_id = query_id
        self.text = text
        self.answers = answers


class FakeQrel:
    def __init__(self, query_id: str, doc_id: str, relevance: int) -> None:
        self.query_id = query_id
        self.doc_id = doc_id
        self.relevance = relevance


class FakeDataset:
    def __init__(self, docs, queries, qrels) -> None:  # type: ignore[no-untyped-def]
        self._docs = docs
        self._queries = queries
        self._qrels = qrels

    def docs_iter(self):  # type: ignore[no-untyped-def]
        return iter(self._docs)

    def queries_iter(self):  # type: ignore[no-untyped-def]
        return iter(self._queries)

    def qrels_iter(self):  # type: ignore[no-untyped-def]
        return iter(self._qrels)


def _dataset() -> FakeDataset:
    docs = [FakeDoc(f"d{i}", f"passage {i}", title=f"T{i}") for i in range(7)]
    queries = [
        FakeQuery("q1", "question one", ("18%",)),
        FakeQuery("q2", "question two", ("Alice", "alice")),
    ]
    qrels = [
        FakeQrel("q1", "d0", 1),
        FakeQrel("q2", "d1", 1),
        FakeQrel("q1", "d2", 0),  # non-positive, ignored
    ]
    return FakeDataset(docs, queries, qrels)


def test_materialize_maps_fields_and_keeps_gold(tmp_path: Path) -> None:
    result = materialize_niah_base(_dataset(), tmp_path, corpus_size=4, seed=42)
    bundle = JsonlDatasetAdapter.load(result.manifest_path)
    ids = {d.document_id for d in bundle.documents}
    assert {"d0", "d1"} <= ids  # gold always kept
    assert len(bundle.documents) == 4  # 2 gold + 2 sampled distractors
    gold = {g.query_id: g for g in bundle.gold_cases}
    assert gold["q1"].relevant_document_ids == ("d0",)
    assert gold["q1"].reference_answers == ("18%",)
    assert gold["q2"].reference_answers == ("alice",)  # deduped, sorted
    d0 = next(d for d in bundle.documents if d.document_id == "d0")
    assert "T0" in d0.text and "passage 0" in d0.text  # title+text join


def test_subsample_is_deterministic(tmp_path: Path) -> None:
    a = materialize_niah_base(_dataset(), tmp_path / "a", corpus_size=4, seed=42)
    b = materialize_niah_base(_dataset(), tmp_path / "b", corpus_size=4, seed=42)
    ids_a = {d.document_id for d in JsonlDatasetAdapter.load(a.manifest_path).documents}
    ids_b = {d.document_id for d in JsonlDatasetAdapter.load(b.manifest_path).documents}
    assert ids_a == ids_b


def test_query_without_positive_qrel_is_dropped(tmp_path: Path) -> None:
    docs = [FakeDoc("d0", "p0"), FakeDoc("d1", "p1")]
    queries = [FakeQuery("q1", "q", ("x",)), FakeQuery("q2", "q", ("y",))]
    qrels = [FakeQrel("q1", "d0", 1)]  # q2 has none
    result = materialize_niah_base(FakeDataset(docs, queries, qrels), tmp_path, corpus_size=2, seed=1)
    bundle = JsonlDatasetAdapter.load(result.manifest_path)
    assert {q.query_id for q in bundle.queries} == {"q1"}
```

- [ ] **Step 1.2: Run to verify failure**

Run: `$env:PYTHONPATH='src'; python -m pytest tests/materializer/test_base_loader.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'evidence_rag.materializer.base_loader'`

- [ ] **Step 1.3: Implement**

```python
"""dpr-w100 NQ base loader with qrels-aware subsampling (spec §3-§5).

Mirrors infrastructure/benchmarks.py but streams the 21M-passage corpus and keeps only
gold passages plus a seeded distractor sample. Provider-injectable; no ir_datasets import
here (only the CLI pulls it in). No slicing — dpr-w100 is pre-chunked.
"""

import json
import random
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from evidence_rag.contracts.models import Document, Query
from evidence_rag.infrastructure.datasets import (
    DatasetManifest,
    GoldCase,
    JsonlDatasetAdapter,
    normalize_document,
    normalize_query,
)


class BaseDoc(Protocol):
    doc_id: str
    text: str


class BaseQuery(Protocol):
    query_id: str
    text: str


class BaseQrel(Protocol):
    query_id: str
    doc_id: str
    relevance: int


class BaseDataset(Protocol):
    def docs_iter(self) -> Iterable[BaseDoc]: ...
    def queries_iter(self) -> Iterable[BaseQuery]: ...
    def qrels_iter(self) -> Iterable[BaseQrel]: ...


@dataclass(frozen=True)
class BaseMaterialization:
    manifest_path: Path
    document_count: int
    query_count: int
    gold_case_count: int


def _document_text(doc: BaseDoc) -> str:
    title = getattr(doc, "title", "")
    parts = [part.strip() for part in (title, doc.text) if isinstance(part, str) and part.strip()]
    if not parts:
        raise ValueError(f"document text must not be blank: {doc.doc_id}")
    return "\n\n".join(parts)


def _write_jsonl(path: Path, records: Iterable[object]) -> None:
    lines = [
        json.dumps(record, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        for record in records
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def materialize_niah_base(
    dataset: BaseDataset,
    output_directory: Path,
    *,
    corpus_size: int,
    query_limit: int | None = None,
    seed: int = 42,
) -> BaseMaterialization:
    if corpus_size <= 0:
        raise ValueError("corpus_size must be positive")

    answers: dict[str, tuple[str, ...] | None] = {}
    query_by_id: dict[str, Query] = {}
    for source_query in dataset.queries_iter():
        query = normalize_query(Query(query_id=source_query.query_id, text=source_query.text))
        query_by_id[query.query_id] = query
        raw = tuple(getattr(source_query, "answers", ()) or ())
        cleaned = tuple(sorted({answer.strip() for answer in raw if answer.strip()}))
        answers[query.query_id] = cleaned or None

    selected_ids = sorted(query_by_id)
    if query_limit is not None:
        selected_ids = selected_ids[:query_limit]
    selected = set(selected_ids)

    positive: dict[str, set[str]] = {qid: set() for qid in selected_ids}
    for qrel in dataset.qrels_iter():
        if int(qrel.relevance) > 0 and qrel.query_id in selected:
            positive[qrel.query_id].add(qrel.doc_id)
    kept_query_ids = [qid for qid in selected_ids if positive[qid]]
    gold_docs = {doc_id for qid in kept_query_ids for doc_id in positive[qid]}

    distractor_budget = max(0, corpus_size - len(gold_docs))
    rng = random.Random(seed)
    kept_docs: dict[str, Document] = {}
    reservoir: list[Document] = []
    seen_distractors = 0
    for source_doc in dataset.docs_iter():
        document = normalize_document(
            Document(
                document_id=source_doc.doc_id,
                text=_document_text(source_doc),
                source_uri=f"ir-datasets://dpr-w100/nq/document/{source_doc.doc_id}",
            )
        )
        if document.document_id in gold_docs:
            kept_docs[document.document_id] = document
            continue
        if distractor_budget == 0:
            continue
        seen_distractors += 1
        if len(reservoir) < distractor_budget:
            reservoir.append(document)
        else:
            index = rng.randrange(seen_distractors)
            if index < distractor_budget:
                reservoir[index] = document

    missing = sorted(gold_docs - set(kept_docs))
    if missing:
        raise ValueError(f"gold document missing from corpus: {missing[0]}")

    documents = tuple(kept_docs[doc_id] for doc_id in sorted(kept_docs)) + tuple(
        sorted(reservoir, key=lambda item: item.document_id)
    )
    queries = tuple(query_by_id[qid] for qid in kept_query_ids)
    gold_cases = tuple(
        GoldCase(
            query_id=qid,
            relevant_document_ids=tuple(sorted(positive[qid])),
            reference_answers=answers[qid],
        )
        for qid in kept_query_ids
    )

    manifest = DatasetManifest(
        dataset_id="niah/dpr-w100-nq",
        dataset_version=f"subsample-{corpus_size}",
        split="dev",
        documents_file="documents.jsonl",
        queries_file="queries.jsonl",
        gold_cases_file="gold_cases.jsonl",
    )
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = output_directory / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest.model_dump(mode="json"), ensure_ascii=True, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    _write_jsonl(
        output_directory / manifest.documents_file,
        (document.model_dump(mode="json") for document in documents),
    )
    _write_jsonl(
        output_directory / manifest.queries_file,
        (query.model_dump(mode="json") for query in queries),
    )
    _write_jsonl(
        output_directory / manifest.gold_cases_file,
        (gold_case.model_dump(mode="json") for gold_case in gold_cases),
    )
    JsonlDatasetAdapter.load(manifest_path)
    return BaseMaterialization(
        manifest_path=manifest_path,
        document_count=len(documents),
        query_count=len(queries),
        gold_case_count=len(gold_cases),
    )
```

- [ ] **Step 1.4: Run to verify pass**

Run: `$env:PYTHONPATH='src'; python -m pytest tests/materializer/test_base_loader.py -q`
Expected: PASS (3 tests)

- [ ] **Step 1.5: Commit**

```bash
git add src/evidence_rag/materializer/base_loader.py tests/materializer/test_base_loader.py
git commit -m "feat(materializer): dpr-w100 NQ base loader with qrels-aware subsample"
```

### Task 2: base_cli.py + entry point

**Files:** Create `src/evidence_rag/materializer/base_cli.py`; Modify `pyproject.toml`; extend `tests/materializer/test_base_loader.py`

- [ ] **Step 2.1: Write failing test (append to test_base_loader.py)**

```python
def test_cli_uses_injected_provider(tmp_path: Path) -> None:
    from evidence_rag.materializer.base_cli import main

    class Provider:
        version = "fake-1"

        def load(self, dataset_id: str) -> FakeDataset:
            assert dataset_id == "dpr-w100/natural-questions/dev"
            return _dataset()

    out = tmp_path / "base"
    exit_code = main(
        ("--split", "dev", "--output", str(out), "--corpus-size", "4", "--seed", "42"),
        provider=Provider(),
    )
    assert exit_code == 0
    bundle = JsonlDatasetAdapter.load(out / "manifest.json")
    assert {"d0", "d1"} <= {d.document_id for d in bundle.documents}
```

- [ ] **Step 2.2: Run to verify failure**

Run: `$env:PYTHONPATH='src'; python -m pytest tests/materializer/test_base_loader.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'evidence_rag.materializer.base_cli'`

- [ ] **Step 2.3: Implement base_cli.py**

```python
"""CLI: materialize the dpr-w100 NQ base dataset (spec §6).

Production loads the dataset through ir_datasets; a provider can be injected for tests.
"""

import argparse
import importlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from evidence_rag.materializer.base_loader import BaseDataset, materialize_niah_base


class BaseProvider(Protocol):
    version: str

    def load(self, dataset_id: str) -> BaseDataset: ...


class _IrDatasetsProvider:
    def __init__(self) -> None:
        module = importlib.import_module("ir_datasets")
        self.version = "ir-datasets"
        self._module = module

    def load(self, dataset_id: str) -> BaseDataset:
        dataset: BaseDataset = self._module.load(dataset_id)
        return dataset


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Materialize the dpr-w100 NQ base dataset")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--corpus-size", type=int, default=100000)
    parser.add_argument("--query-limit", type=int)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Sequence[str] | None = None, *, provider: BaseProvider | None = None) -> int:
    arguments = _parser().parse_args(argv)
    active = provider if provider is not None else _IrDatasetsProvider()
    dataset = active.load(f"dpr-w100/natural-questions/{arguments.split}")
    result = materialize_niah_base(
        dataset,
        arguments.output,
        corpus_size=arguments.corpus_size,
        query_limit=arguments.query_limit,
        seed=arguments.seed,
    )
    print(
        json.dumps(
            {
                "manifest": str(result.manifest_path),
                "documents": result.document_count,
                "queries": result.query_count,
                "gold_cases": result.gold_case_count,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2.4: Register the entry point in pyproject.toml**

In `[project.scripts]` add (keep alphabetical):

```toml
evidence-rag-load-niah-base = "evidence_rag.materializer.base_cli:main"
```

- [ ] **Step 2.5: Run to verify pass, full suite, typecheck, ruff**

Run:
```
$env:PYTHONPATH='src'; python -m pytest tests/materializer/test_base_loader.py -q
$env:PYTHONPATH='src'; python -m pytest -q
$env:MYPYPATH='src'; python -m mypy
python -m ruff check src/evidence_rag/materializer tests/materializer
```
Expected: base-loader tests pass; full suite parity (no new failures); mypy clean; ruff clean.

- [ ] **Step 2.6: Commit**

```bash
git add src/evidence_rag/materializer/base_cli.py pyproject.toml tests/materializer/test_base_loader.py
git commit -m "feat(materializer): evidence-rag-load-niah-base CLI"
```

---

## Experiment marking (running-hpc-experiments)

CPU/offline code. **Real materialization is a login-node ops step**: `ir_datasets` must download dpr-w100 NQ (21M passages, large) on the login node (compute offline) before `evidence-rag-load-niah-base` runs. Record the base-dataset build + the downstream `evidence-rag-materialize-niah` (B) + E2 under E2 in `docs/hpc-run-log.md` when run. No slurm here (login-node CPU).

## Self-review notes

- Spec coverage: §3 provider/output→Task 1; §4 subsample (reservoir, gold-kept)→Task 1; §5 field mapping→Task 1; §6 CLI→Task 2; §8 no-chunking honored (pre-chunked docs, no chunker); fresh-query audit intentionally out (spec §8 defers to final run).
- No placeholders: complete code; reservoir sampling concrete; ir_datasets import isolated to the CLI provider.
- Type consistency: `BaseDataset`/`materialize_niah_base`/`BaseMaterialization` names identical across Tasks 1-2; CLI `provider.load` returns `BaseDataset`.
- Determinism: seeded reservoir; test asserts identical id sets across two runs.
