"""Build contract-internal top-20 evidence pools from official ContractNLI spans."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from eval.niah_selector_pilot import rank_candidates


def contract_utility_grade(
    *, choice: str, evidence_spans: Sequence[int], candidate_span: int
) -> int:
    """Map official evidence spans to canonical utility without inventing negatives."""

    if choice not in {"Entailment", "Contradiction", "NotMentioned"}:
        raise ValueError(f"unsupported ContractNLI choice {choice!r}")
    if choice == "NotMentioned" or candidate_span not in evidence_spans:
        return 1
    return 4 if len(evidence_spans) == 1 else 3


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def build_contractnli_pools(
    *, contract_dir: Path, out: Path, embedding_model: str, device: str, batch_size: int
) -> None:
    """Embed clauses once and rank each hypothesis only inside its source contract."""

    import numpy as np
    from sentence_transformers import SentenceTransformer

    payloads = {
        split: json.loads((contract_dir / f"{split}.json").read_text(encoding="utf-8"))
        for split in ("train", "dev", "test")
    }
    labels = payloads["train"]["labels"]
    if any(payload["labels"] != labels for payload in payloads.values()):
        raise ValueError("ContractNLI hypothesis definitions differ across official splits")
    hypothesis_ids = sorted(labels)
    model = SentenceTransformer(embedding_model, device=device)
    hypothesis_vectors = model.encode(
        [str(labels[hypothesis_id]["hypothesis"]) for hypothesis_id in hypothesis_ids],
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    hypothesis_vector = {
        hypothesis_id: hypothesis_vectors[index]
        for index, hypothesis_id in enumerate(hypothesis_ids)
    }
    output: list[dict[str, object]] = []
    for split, payload in payloads.items():
        documents = payload["documents"]
        clause_texts = [
            str(document["text"])[int(span[0]) : int(span[1])]
            for document in documents
            for span in document["spans"]
        ]
        clause_vectors = model.encode(
            clause_texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        vector_offset = 0
        for document in documents:
            document_id = str(document["id"])
            spans = document["spans"]
            vectors = clause_vectors[vector_offset : vector_offset + len(spans)]
            vector_offset += len(spans)
            annotations = document["annotation_sets"][0]["annotations"]
            for hypothesis_id in hypothesis_ids:
                annotation = annotations[hypothesis_id]
                choice = str(annotation["choice"])
                evidence_spans = [int(value) for value in annotation["spans"]]
                scores = vectors @ hypothesis_vector[hypothesis_id]
                candidates: list[dict[str, object]] = []
                for span_index, (span, score) in enumerate(zip(spans, scores)):
                    text = str(document["text"])[int(span[0]) : int(span[1])]
                    candidates.append(
                        {
                            "candidate_id": f"contract-{document_id}__span__{span_index:04d}",
                            "text": text,
                            "title": str(document["file_name"]),
                            "source_parent_id": f"contract:{document_id}",
                            "source": "contractnli_clause",
                            "span_index": span_index,
                            "utility_grade": contract_utility_grade(
                                choice=choice,
                                evidence_spans=evidence_spans,
                                candidate_span=span_index,
                            ),
                            "relevance_score": float(score),
                        }
                    )
                ranked = rank_candidates(candidates, "relevance_score")[:20]
                for rank, candidate in enumerate(ranked, start=1):
                    candidate["original_rank"] = rank
                output.append(
                    {
                        "query_id": f"contract-{split}-{document_id}-{hypothesis_id}",
                        "split": split,
                        "question": labels[hypothesis_id]["hypothesis"],
                        "hypothesis_id": hypothesis_id,
                        "contract_id": document_id,
                        "annotation_choice": choice,
                        "eligible_for_training": choice != "NotMentioned",
                        "required_span_indices": evidence_spans,
                        "candidates": ranked,
                    }
                )
    _write_jsonl(out, output)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args(argv)
    build_contractnli_pools(
        contract_dir=args.contract_dir,
        out=args.out,
        embedding_model=args.embedding_model,
        device=args.device,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
