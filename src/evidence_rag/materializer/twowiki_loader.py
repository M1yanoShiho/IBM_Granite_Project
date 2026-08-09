"""2WikiMultihopQA (dev) loader → the uniform JSONL dataset bundle.

Mirrors ``base_loader.materialize_niah_base`` but sources a multi-hop QA set instead of
dpr-w100. Each question ships ~10 context paragraphs; the corpus is the *union* of the
selected questions' paragraphs (deduplicated by title), gold = the ``supporting_facts``
titles, and the reference answer is kept for downstream answer-match. Multi-hop is the
setting where query decomposition / rewriting is expected to help, so this complements the
single-hop SciFact/NQ benchmarks.

Provider-decoupled: ``materialize_2wiki`` takes an iterable of row mappings (the CLI reads
the parquet with pandas and passes ``frame.to_dict("records")``); no pandas import here.
"""

import json
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evidence_rag.contracts.models import Document, Query
from evidence_rag.infrastructure.datasets import (
    DatasetManifest,
    GoldCase,
    normalize_document,
    normalize_query,
)


@dataclass(frozen=True)
class TwoWikiMaterialization:
    manifest_path: Path
    document_count: int
    query_count: int
    gold_case_count: int


@dataclass(frozen=True)
class _ParsedQuestion:
    query_id: str
    question: str
    answer: str
    paragraphs: tuple[tuple[str, str], ...]  # (title, body)
    gold_titles: tuple[str, ...]


def _as_json(value: object) -> Any:
    """2Wiki parquet stores context/supporting_facts as JSON strings; be tolerant of
    already-decoded lists too."""

    return json.loads(value) if isinstance(value, (str, bytes)) else value


def _paragraphs(raw_context: object) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for entry in _as_json(raw_context) or []:
        title, body = entry[0], entry[1]
        text = " ".join(str(part) for part in body) if isinstance(body, Sequence) and not isinstance(body, str) else str(body)
        pairs.append((str(title).strip(), text.strip()))
    return pairs


def _gold_titles(raw_supporting: object) -> list[str]:
    titles: list[str] = []
    for fact in _as_json(raw_supporting) or []:
        titles.append(str(fact[0]).strip())
    return titles


def _parse_row(row: Mapping[str, object]) -> _ParsedQuestion | None:
    query_id = str(row.get("_id", "")).strip()
    question = str(row.get("question", "")).strip()
    answer = str(row.get("answer", "")).strip()
    if not query_id or not question or not answer:
        return None
    paragraphs = tuple((t, b) for t, b in _paragraphs(row.get("context")) if t and b)
    if not paragraphs:
        return None
    titles_present = {title for title, _ in paragraphs}
    gold = tuple(dict.fromkeys(t for t in _gold_titles(row.get("supporting_facts")) if t))
    # Drop questions whose gold paragraph is absent from the given context: the relevance
    # label would be unreachable and understate recall.
    if not gold or any(title not in titles_present for title in gold):
        return None
    return _ParsedQuestion(query_id, question, answer, paragraphs, gold)


def _write_jsonl(path: Path, records: Iterable[object]) -> None:
    lines = [
        json.dumps(record, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        for record in records
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def materialize_2wiki(
    rows: Iterable[Mapping[str, object]],
    output_directory: Path | str,
    *,
    query_limit: int,
    seed: int,
    split: str = "dev",
) -> TwoWikiMaterialization:
    """Subsample ``query_limit`` well-formed questions (seeded) and write the bundle."""

    if query_limit <= 0:
        raise ValueError("query_limit must be positive")
    split = split.strip()
    if not split:
        raise ValueError("split must not be blank")

    pool = list(rows)
    rng = random.Random(seed)
    rng.shuffle(pool)

    selected: list[_ParsedQuestion] = []
    for row in pool:
        parsed = _parse_row(row)
        if parsed is None:
            continue
        selected.append(parsed)
        if len(selected) >= query_limit:
            break

    # Corpus = union of the selected questions' paragraphs, deduplicated by title
    # (first occurrence wins for a deterministic body).
    documents: dict[str, Document] = {}
    for question in sorted(selected, key=lambda item: item.query_id):
        for title, body in question.paragraphs:
            if title in documents:
                continue
            documents[title] = normalize_document(
                Document(
                    document_id=title,
                    text=f"{title}\n\n{body}",
                    source_uri=f"2wiki://{split}/{title}",
                )
            )

    queries = tuple(
        normalize_query(Query(query_id=q.query_id, text=q.question))
        for q in sorted(selected, key=lambda item: item.query_id)
    )
    gold_cases = tuple(
        GoldCase(
            query_id=q.query_id,
            relevant_document_ids=tuple(sorted(set(q.gold_titles))),
            reference_answers=(q.answer,),
        )
        for q in sorted(selected, key=lambda item: item.query_id)
    )
    ordered_documents = tuple(documents[title] for title in sorted(documents))

    manifest = DatasetManifest(
        dataset_id="2wiki/multihop",
        dataset_version=f"{split}-subsample-{query_limit}",
        split=split,
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
        (document.model_dump(mode="json") for document in ordered_documents),
    )
    _write_jsonl(
        output_directory / manifest.queries_file,
        (query.model_dump(mode="json") for query in queries),
    )
    _write_jsonl(
        output_directory / manifest.gold_cases_file,
        (gold.model_dump(mode="json") for gold in gold_cases),
    )
    return TwoWikiMaterialization(
        manifest_path=manifest_path,
        document_count=len(ordered_documents),
        query_count=len(queries),
        gold_case_count=len(gold_cases),
    )
