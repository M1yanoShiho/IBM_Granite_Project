"""Prepare official and adapted RAMDocs candidate pools for selector evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from eval.niah_selector_pilot import GraniteBatchGenerator, Q2D_PROMPT, rank_candidates


def ramdocs_utility_grade(document_type: str) -> int:
    """Map official RAMDocs types and external mined noise to canonical utility."""

    mapping = {"correct": 4, "noise": 1, "misinfo": 0, "mined_external": 1}
    try:
        return mapping[document_type]
    except KeyError as exc:
        raise ValueError(f"unsupported RAMDocs type {document_type!r}") from exc


def select_mined_documents(
    documents: Sequence[Mapping[str, object]], *, query_id: str, limit: int
) -> list[dict[str, object]]:
    """Select high-scoring cross-query documents with deterministic text deduplication."""

    selected: list[dict[str, object]] = []
    seen_texts: set[str] = set()
    for document in sorted(
        documents,
        key=lambda row: (-float(row["score"]), str(row["candidate_id"])),
    ):
        if str(document["origin_query_id"]) == query_id:
            continue
        normalized = " ".join(str(document["text"]).split()).casefold()
        if normalized in seen_texts:
            continue
        seen_texts.add(normalized)
        selected.append(dict(document))
        if len(selected) == limit:
            break
    if len(selected) != limit:
        raise ValueError(f"could mine only {len(selected)} of {limit} requested documents")
    return selected


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def prepare_ramdocs_pool(
    *, raw_path: Path, out: Path, model_id: str, device: str, batch_size: int
) -> None:
    """Materialize official candidates and one Query2Doc expansion per query."""

    raw_rows = _read_jsonl(raw_path)
    generator = GraniteBatchGenerator(model_id, device=device)
    expansions = generator.generate(
        [Q2D_PROMPT.format(question=row["question"]) for row in raw_rows],
        batch_size=batch_size,
        max_new_tokens=96,
    )
    rows: list[dict[str, object]] = []
    for index, (raw, expansion) in enumerate(zip(raw_rows, expansions)):
        query_id = f"ramdocs-{index:06d}"
        candidates = []
        for document_index, document in enumerate(raw["documents"]):
            document_type = str(document["type"])
            candidates.append(
                {
                    "candidate_id": f"{query_id}__official__{document_index:03d}",
                    "text": str(document["text"]),
                    "title": "RAMDocs official candidate",
                    "source_parent_id": f"ramdocs-source:{query_id}:{document_index:03d}",
                    "source": "ramdocs_official",
                    "official_type": document_type,
                    "official_answer": document.get("answer"),
                    "utility_grade": ramdocs_utility_grade(document_type),
                    "origin_query_id": query_id,
                }
            )
        rows.append(
            {
                "query_id": query_id,
                "split": "external",
                "question": raw["question"],
                "answers": raw["gold_answers"],
                "wrong_answers": raw["wrong_answers"],
                "disambig_entity": raw["disambig_entity"],
                "query2doc": f"{raw['question']} {expansion.strip()}".strip(),
                "official_candidate_count": len(candidates),
                "candidates": candidates,
            }
        )
    _write_jsonl(out, rows)


def rank_adapted_ramdocs_pool(
    *, base_pool: Path, out: Path, embedding_model: str, device: str, batch_size: int
) -> None:
    """Mine cross-query negatives, freeze 20 candidates, and preserve official labels."""

    import numpy as np
    from sentence_transformers import SentenceTransformer

    rows = _read_jsonl(base_pool)
    official = [
        dict(candidate, origin_query_id=row["query_id"])
        for row in rows
        for candidate in row["candidates"]
    ]
    candidate_ids = [str(candidate["candidate_id"]) for candidate in official]
    model = SentenceTransformer(embedding_model, device=device)
    document_vectors = model.encode(
        [str(candidate["text"]) for candidate in official],
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
    index_by_id = {candidate_id: index for index, candidate_id in enumerate(candidate_ids)}
    output_rows: list[dict[str, object]] = []
    for row_index, row in enumerate(rows):
        similarities = document_vectors @ query_vectors[row_index]
        scored_documents = [
            {
                **candidate,
                "score": float(similarities[index]),
            }
            for index, candidate in enumerate(official)
        ]
        official_candidates = []
        for candidate_raw in row["candidates"]:
            candidate = dict(candidate_raw)
            candidate["relevance_score"] = float(
                similarities[index_by_id[str(candidate["candidate_id"])]]
            )
            official_candidates.append(candidate)
        mined = select_mined_documents(
            scored_documents,
            query_id=str(row["query_id"]),
            limit=20 - len(official_candidates),
        )
        mined_candidates = [
            {
                "candidate_id": f"{row['query_id']}__mined__{candidate['candidate_id']}",
                "text": candidate["text"],
                "title": "RAMDocs cross-query mined negative",
                "source_parent_id": candidate["source_parent_id"],
                "source": "mined_external",
                "official_type": None,
                "official_answer": None,
                "utility_grade": ramdocs_utility_grade("mined_external"),
                "origin_query_id": candidate["origin_query_id"],
                "relevance_score": candidate["score"],
            }
            for candidate in mined
        ]
        ranked = rank_candidates(
            [*official_candidates, *mined_candidates], "relevance_score"
        )[:20]
        for rank, candidate in enumerate(ranked, start=1):
            candidate["original_rank"] = rank
        output = {key: value for key, value in row.items() if key != "candidates"}
        output["candidates"] = ranked
        output_rows.append(output)
    _write_jsonl(out, output_rows)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--raw", type=Path, required=True)
    prepare.add_argument("--out", type=Path, required=True)
    prepare.add_argument("--model-id", required=True)
    prepare.add_argument("--device", default="cuda:0")
    prepare.add_argument("--batch-size", type=int, default=32)
    rank = subparsers.add_parser("rank-adapted")
    rank.add_argument("--base-pool", type=Path, required=True)
    rank.add_argument("--out", type=Path, required=True)
    rank.add_argument("--embedding-model", required=True)
    rank.add_argument("--device", default="cuda:0")
    rank.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        prepare_ramdocs_pool(
            raw_path=args.raw,
            out=args.out,
            model_id=args.model_id,
            device=args.device,
            batch_size=args.batch_size,
        )
    else:
        rank_adapted_ramdocs_pool(
            base_pool=args.base_pool,
            out=args.out,
            embedding_model=args.embedding_model,
            device=args.device,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()
