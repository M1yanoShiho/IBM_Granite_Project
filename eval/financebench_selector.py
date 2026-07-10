"""Prepare the official FinanceBench evidence corpus for selector evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Iterable, Mapping, Sequence

from eval.niah_selector_pilot import GraniteBatchGenerator, Q2D_PROMPT, rank_candidates


def token_coverage(reference: str, candidate: str) -> float:
    """Return the fraction of unique reference tokens present in a candidate."""

    reference_tokens = set(re.findall(r"[a-z0-9]+", reference.casefold()))
    if not reference_tokens:
        return 0.0
    candidate_tokens = set(re.findall(r"[a-z0-9]+", candidate.casefold()))
    return len(reference_tokens.intersection(candidate_tokens)) / len(reference_tokens)


def finance_utility_grade(
    *,
    candidate_id: str,
    candidate_doc: str,
    candidate_company: str,
    target_doc: str,
    target_company: str,
    official_page_ids: set[str],
    required_page_count: int,
) -> tuple[int, str | None]:
    """Map official evidence and enterprise-style mismatches to utility and harm."""

    if candidate_id in official_page_ids:
        return (4 if required_page_count == 1 else 3), None
    if candidate_doc == target_doc:
        return 1, None
    if candidate_company == target_company:
        return 0, "wrong_period"
    return 0, "wrong_entity"


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def prepare_financebench_evidence_corpus(
    *, questions_path: Path, out: Path, model_id: str, device: str, batch_size: int
) -> None:
    """Materialize 150 questions and all 189 official evidence snippets."""

    raw_rows = _read_jsonl(questions_path)
    generator = GraniteBatchGenerator(model_id, device=device)
    expansions = generator.generate(
        [Q2D_PROMPT.format(question=row["question"]) for row in raw_rows],
        batch_size=batch_size,
        max_new_tokens=96,
    )
    corpus: list[dict[str, object]] = []
    for row in raw_rows:
        query_id = f"finance-{row['financebench_id']}"
        for evidence_index, evidence in enumerate(row["evidence"]):
            corpus.append(
                {
                    "candidate_id": f"{query_id}__evidence__{evidence_index:02d}",
                    "text": str(evidence["evidence_text"]),
                    "title": str(evidence["doc_name"]),
                    "source_parent_id": f"finance-report:{evidence['doc_name']}",
                    "candidate_doc": str(evidence["doc_name"]),
                    "candidate_company": str(row["company"]),
                    "evidence_page_num": int(evidence["evidence_page_num"]),
                    "origin_query_id": query_id,
                }
            )
    output: list[dict[str, object]] = []
    for raw, expansion in zip(raw_rows, expansions):
        query_id = f"finance-{raw['financebench_id']}"
        official_ids = [
            f"{query_id}__evidence__{index:02d}" for index in range(len(raw["evidence"]))
        ]
        output.append(
            {
                "query_id": query_id,
                "split": "finance",
                "question": raw["question"],
                "answers": [raw["answer"]],
                "company": raw["company"],
                "target_doc": raw["doc_name"],
                "question_type": raw["question_type"],
                "official_candidate_ids": official_ids,
                "query2doc": f"{raw['question']} {expansion.strip()}".strip(),
            }
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"corpus": corpus, "queries": output}, ensure_ascii=False, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def rank_financebench_pool(
    *, base_pool: Path, out: Path, embedding_model: str, device: str, batch_size: int
) -> None:
    """Freeze one top-20 evidence-snippet pool per FinanceBench question."""

    import numpy as np
    from sentence_transformers import SentenceTransformer

    payload = json.loads(base_pool.read_text(encoding="utf-8"))
    rows = payload["queries"]
    corpus = [dict(candidate) for candidate in payload["corpus"]]
    model = SentenceTransformer(embedding_model, device=device)
    document_vectors = model.encode(
        [str(candidate["text"]) for candidate in corpus],
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    query_vectors = model.encode(
        [str(row["query2doc"]) for row in rows],
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    output_rows: list[dict[str, object]] = []
    for row_index, row in enumerate(rows):
        scores = document_vectors @ query_vectors[row_index]
        official_ids = {str(value) for value in row["official_candidate_ids"]}
        candidates: list[dict[str, object]] = []
        for candidate_raw, score in zip(corpus, scores):
            candidate = dict(candidate_raw)
            grade, harm_type = finance_utility_grade(
                candidate_id=str(candidate["candidate_id"]),
                candidate_doc=str(candidate["candidate_doc"]),
                candidate_company=str(candidate["candidate_company"]),
                target_doc=str(row["target_doc"]),
                target_company=str(row["company"]),
                official_page_ids=official_ids,
                required_page_count=len(official_ids),
            )
            candidate.update(
                {
                    "source": "financebench_official_evidence_corpus",
                    "utility_grade": grade,
                    "harm_type": harm_type,
                    "relevance_score": float(score),
                }
            )
            candidates.append(candidate)
        ranked = rank_candidates(candidates, "relevance_score")[:20]
        for rank, candidate in enumerate(ranked, start=1):
            candidate["original_rank"] = rank
        output = {key: value for key, value in row.items() if key != "official_candidate_ids"}
        output["official_candidate_ids"] = sorted(official_ids)
        output["candidates"] = ranked
        output_rows.append(output)
    _write_jsonl(out, output_rows)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--questions", type=Path, required=True)
    prepare.add_argument("--out", type=Path, required=True)
    prepare.add_argument("--model-id", required=True)
    prepare.add_argument("--device", default="cuda:0")
    prepare.add_argument("--batch-size", type=int, default=32)
    rank = subparsers.add_parser("rank")
    rank.add_argument("--base-pool", type=Path, required=True)
    rank.add_argument("--out", type=Path, required=True)
    rank.add_argument("--embedding-model", required=True)
    rank.add_argument("--device", default="cuda:0")
    rank.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        prepare_financebench_evidence_corpus(
            questions_path=args.questions,
            out=args.out,
            model_id=args.model_id,
            device=args.device,
            batch_size=args.batch_size,
        )
    else:
        rank_financebench_pool(
            base_pool=args.base_pool,
            out=args.out,
            embedding_model=args.embedding_model,
            device=args.device,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()
