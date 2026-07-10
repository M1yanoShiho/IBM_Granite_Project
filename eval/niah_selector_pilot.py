"""Controlled NQ/DPR pilot for relevance-aware evidence selection."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import re
from typing import Iterable, Mapping, Sequence

from src.retrieval.corroboration import is_valid_answer, normalize_answer


LABEL_GAINS = (0, 1, 3, 7, 15)
CORE_FEATURES = (
    "relevance_score",
    "relevance_normalized",
    "original_rank",
    "reciprocal_rank",
    "exact_vote_count",
    "vote_ratio",
    "parametric_agreement",
    "extraction_failure",
)
FULL_FEATURES = CORE_FEATURES + (
    "source_dedup_vote_count",
    "dominant_answer_agreement",
    "answer_length",
)

Q2D_PROMPT = (
    "Write a short, factual passage that answers the question.\n"
    "Question: {question}\nPassage:"
)
WRONG_ENTITY_PROMPT = (
    "Return one different but same-type plausible replacement for the answer below. "
    "Output only the replacement entity.\nAnswer: {answer}\nReplacement:"
)
SEALED_WRONG_ENTITY_PROMPT = (
    "For a factual robustness test, substitute the answer with one plausible entity "
    "of the same type. Give only that different entity.\nOriginal: {answer}\nSubstitute:"
)
NON_ANSWER_PROMPT = (
    "Write a short fluent passage on the question's topic that shares key terms but "
    "does not state or imply the answer. Output only the passage.\n"
    "Question: {question}\nReference style:\n{needle}\nPassage:"
)
SEALED_NON_ANSWER_PROMPT = (
    "Produce a concise background paragraph related to the query. Discuss adjacent "
    "facts and omit the requested answer. Return only the paragraph.\n"
    "Query: {question}\nStyle sample:\n{needle}\nBackground:"
)
EXTRACT_PROMPT = (
    "Using only the passage, answer the question with the shortest exact answer. "
    "If the passage cannot answer it, reply NONE.\n"
    "Question: {question}\nPassage: {passage}\nAnswer:"
)
PARAMETRIC_PROMPT = (
    "Answer with the shortest exact answer from your own knowledge. If unsure, reply "
    "NONE.\nQuestion: {question}\nAnswer:"
)


def utility_grade(*, source: str, relevance: int | None) -> int:
    """Map controlled-NIAH provenance to the canonical five-level utility label."""

    if source == "counterfactual":
        return 0
    if source == "generative_non_answer":
        return 2
    if source != "dpr":
        raise ValueError(f"Unknown candidate source {source!r}")
    mapping = {2: 4, 1: 3, 0: 1, -1: 2}
    if relevance not in mapping:
        raise ValueError(f"Unsupported DPR relevance {relevance!r}")
    return mapping[relevance]


def minmax(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    low, high = min(values), max(values)
    if high == low:
        return [0.0] * len(values)
    return [(value - low) / (high - low) for value in values]


def rank_candidates(rows: Sequence[Mapping[str, object]], score_key: str) -> list[dict[str, object]]:
    """Rank by score with the preregistered deterministic tie-break."""

    return [
        dict(row)
        for row in sorted(
            rows,
            key=lambda row: (
                -float(row[score_key]),
                int(row.get("original_rank", 10**9)),
                str(row["candidate_id"]),
            ),
        )
    ]


def add_group_features(
    rows: Sequence[Mapping[str, object]], *, parametric_answer: str | None
) -> list[dict[str, object]]:
    """Add cost-matched corroboration and cheap reliability features."""

    answers = [str(row.get("extracted_answer", "")) for row in rows]
    normalized = [normalize_answer(answer) if is_valid_answer(answer) else None for answer in answers]
    parametric = (
        normalize_answer(parametric_answer)
        if parametric_answer and is_valid_answer(parametric_answer)
        else None
    )
    valid_count = sum(answer is not None for answer in normalized)
    frequency = Counter(answer for answer in normalized if answer is not None)
    dominant = max(frequency, key=lambda answer: (frequency[answer], answer)) if frequency else None
    relevance = [float(row["relevance_score"]) for row in rows]
    relevance_norm = minmax(relevance)
    output: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        item = dict(row)
        answer = normalized[index]
        exact_votes = 0
        source_votes: set[str] = set()
        if answer is not None:
            own_parent = str(row.get("source_parent_id", ""))
            for other_index, other_answer in enumerate(normalized):
                if other_index == index or other_answer != answer:
                    continue
                exact_votes += 1
                other_parent = str(rows[other_index].get("source_parent_id", ""))
                if other_parent != own_parent:
                    source_votes.add(other_parent)
            if parametric == answer:
                exact_votes += 1
        item.update(
            {
                "relevance_normalized": relevance_norm[index],
                "reciprocal_rank": 1.0 / max(1, int(row["original_rank"])),
                "exact_vote_count": float(exact_votes),
                "vote_ratio": float(exact_votes) / max(1, valid_count),
                "parametric_agreement": float(answer is not None and answer == parametric),
                "extraction_failure": float(answer is None),
                "source_dedup_vote_count": float(
                    len(source_votes) + (1 if answer is not None and answer == parametric else 0)
                ),
                "dominant_answer_agreement": float(answer is not None and answer == dominant),
                "answer_length": float(len(answer or "")),
            }
        )
        output.append(item)
    return output


def _dcg(grades: Sequence[int]) -> float:
    return sum(LABEL_GAINS[grade] / math.log2(rank + 2) for rank, grade in enumerate(grades))


def selector_metrics(
    groups: Mapping[str, Sequence[Mapping[str, object]]], *, score_key: str, k: int = 10
) -> dict[str, float]:
    ndcgs: list[float] = []
    required_recalls: list[float] = []
    harmful = direct = selected_count = 0
    for rows in groups.values():
        ranked = rank_candidates(rows, score_key)
        selected = ranked[:k]
        grades = [int(row["utility_grade"]) for row in selected]
        ideal = sorted((int(row["utility_grade"]) for row in rows), reverse=True)[:k]
        denominator = _dcg(ideal)
        ndcgs.append(_dcg(grades) / denominator if denominator else 0.0)
        required_total = sum(int(row["utility_grade"]) >= 3 for row in rows)
        required_selected = sum(int(row["utility_grade"]) >= 3 for row in selected)
        required_recalls.append(required_selected / required_total if required_total else 1.0)
        harmful += sum(int(row["utility_grade"]) == 0 for row in selected)
        direct += sum(int(row["utility_grade"]) == 4 for row in selected)
        selected_count += len(selected)
    return {
        f"ndcg@{k}": sum(ndcgs) / len(ndcgs) if ndcgs else 0.0,
        f"required_evidence_recall@{k}": (
            sum(required_recalls) / len(required_recalls) if required_recalls else 0.0
        ),
        f"harmful_rate@{k}": harmful / selected_count if selected_count else 0.0,
        f"direct_support_precision@{k}": direct / selected_count if selected_count else 0.0,
    }


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


class GraniteBatchGenerator:
    """Small deterministic batched wrapper used for all cached Granite calls."""

    def __init__(self, model_id: str, *, device: str = "cuda") -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            local_files_only=True,
            dtype=torch.float16 if device.startswith("cuda") else "auto",
            device_map=device,
        )
        self.model.eval()

    def generate(
        self, prompts: Sequence[str], *, batch_size: int, max_new_tokens: int
    ) -> list[str]:
        outputs: list[str] = []
        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start : start + batch_size]
            rendered = [
                self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for prompt in batch_prompts
            ]
            encoded = self.tokenizer(
                rendered, return_tensors="pt", padding=True, truncation=True, max_length=1024
            ).to(self.model.device)
            with self.torch.inference_mode():
                generated = self.model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            for index in range(len(batch_prompts)):
                new_tokens = generated[index, encoded["input_ids"].shape[1] :]
                outputs.append(
                    self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                )
        return outputs


def _surface_alias(needle: str, aliases: Sequence[str]) -> str | None:
    matches = [alias for alias in aliases if alias and re.search(re.escape(alias), needle, re.I)]
    return max(matches, key=len) if matches else None


def _clean_entity(value: str) -> str:
    first_line = value.strip().splitlines()[0] if value.strip() else ""
    return first_line.strip(" \t\"'`.*:-")


def generate_pilot_material(
    *, base_pool: Path, out: Path, model_id: str, device: str, batch_size: int
) -> None:
    """Cache Query2Doc expansions and two controlled distractors per query."""

    rows = _read_jsonl(base_pool)
    generator = GraniteBatchGenerator(model_id, device=device)
    q2d_outputs = generator.generate(
        [Q2D_PROMPT.format(question=row["question"]) for row in rows],
        batch_size=batch_size,
        max_new_tokens=96,
    )
    aliases: list[str | None] = []
    needles: list[dict[str, object]] = []
    for row in rows:
        by_id = {candidate["candidate_id"]: candidate for candidate in row["candidates"]}
        needle = by_id[str(row["needle_doc_id"])]
        needles.append(needle)
        aliases.append(_surface_alias(str(needle["text"]), row["answers"]))
    wrong_prompts = [
        (
            SEALED_WRONG_ENTITY_PROMPT if row["split"] == "test" else WRONG_ENTITY_PROMPT
        ).format(answer=alias or row["answers"][0])
        for row, alias in zip(rows, aliases)
    ]
    wrong_outputs = generator.generate(
        wrong_prompts, batch_size=batch_size, max_new_tokens=24
    )
    non_answer_prompts = [
        (SEALED_NON_ANSWER_PROMPT if row["split"] == "test" else NON_ANSWER_PROMPT).format(
            question=row["question"], needle=str(needle["text"])[:700]
        )
        for row, needle in zip(rows, needles)
    ]
    non_answers = generator.generate(
        non_answer_prompts, batch_size=batch_size, max_new_tokens=128
    )

    enriched: list[dict[str, object]] = []
    for row, q2d, alias, wrong_raw, non_answer, needle in zip(
        rows, q2d_outputs, aliases, wrong_outputs, non_answers, needles
    ):
        item = dict(row)
        candidates = [dict(candidate) for candidate in row["candidates"]]
        wrong = _clean_entity(wrong_raw)
        counterfactual = None
        if alias and wrong and normalize_answer(wrong) != normalize_answer(alias):
            replaced = re.sub(re.escape(alias), wrong, str(needle["text"]), flags=re.I)
            if replaced != needle["text"]:
                counterfactual = {
                    "candidate_id": f"{row['query_id']}__counterfactual",
                    "text": replaced,
                    "title": needle["title"],
                    "source_parent_id": needle["source_parent_id"],
                    "source": "counterfactual",
                    "dpr_relevance": None,
                    "dpr_rank": len(candidates) + 1,
                    "utility_grade": 0,
                }
                candidates.append(counterfactual)
        non_answer_valid = bool(non_answer.strip()) and not any(
            alias_value.strip().casefold() in non_answer.casefold()
            for alias_value in row["answers"]
            if alias_value.strip()
        )
        if non_answer_valid:
            candidates.append(
                {
                    "candidate_id": f"{row['query_id']}__non_answer",
                    "text": non_answer.strip(),
                    "title": "synthetic topical background",
                    "source_parent_id": f"synthetic:{row['query_id']}",
                    "source": "generative_non_answer",
                    "dpr_relevance": None,
                    "dpr_rank": len(candidates) + 1,
                    "utility_grade": 2,
                }
            )
        item["query2doc"] = f"{row['question']} {q2d.strip()}".strip()
        item["counterfactual_valid"] = counterfactual is not None
        item["non_answer_valid"] = non_answer_valid
        item["candidates"] = candidates
        enriched.append(item)
    _write_jsonl(out, enriched)


def rank_pilot_pool(
    *, generated_pool: Path, out: Path, embedding_model: str, device: str, batch_size: int
) -> None:
    """Use Query2Doc + Granite embeddings to freeze one top-20 pool per query."""

    import numpy as np
    from sentence_transformers import SentenceTransformer

    rows = _read_jsonl(generated_pool)
    texts_by_id: dict[str, str] = {}
    for row in rows:
        for candidate in row["candidates"]:
            candidate_id = str(candidate["candidate_id"])
            text = str(candidate["text"])
            if candidate_id in texts_by_id and texts_by_id[candidate_id] != text:
                raise ValueError(f"Candidate ID {candidate_id} has inconsistent text")
            texts_by_id[candidate_id] = text
    candidate_ids = sorted(texts_by_id)
    model = SentenceTransformer(embedding_model, device=device)
    doc_vectors = model.encode(
        [texts_by_id[candidate_id] for candidate_id in candidate_ids],
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
    vector_by_id = {candidate_id: doc_vectors[index] for index, candidate_id in enumerate(candidate_ids)}
    ranked_rows: list[dict[str, object]] = []
    for row_index, row in enumerate(rows):
        scored: list[dict[str, object]] = []
        for candidate in row["candidates"]:
            item = dict(candidate)
            item["relevance_score"] = float(
                np.dot(query_vectors[row_index], vector_by_id[str(candidate["candidate_id"])] )
            )
            item["original_rank"] = int(candidate["dpr_rank"])
            scored.append(item)
        ranked = rank_candidates(scored, "relevance_score")[:20]
        for rank, candidate in enumerate(ranked, start=1):
            candidate["original_rank"] = rank
        output = {key: value for key, value in row.items() if key != "candidates"}
        output["candidates"] = ranked
        ranked_rows.append(output)
    _write_jsonl(out, ranked_rows)


def extract_pilot_features(
    *, ranked_pool: Path, out: Path, model_id: str, device: str, batch_size: int
) -> None:
    """Extract candidate answers once and derive the shared selector feature cache."""

    rows = _read_jsonl(ranked_pool)
    generator = GraniteBatchGenerator(model_id, device=device)
    prompts: list[str] = []
    positions: list[tuple[int, int]] = []
    for row_index, row in enumerate(rows):
        for candidate_index, candidate in enumerate(row["candidates"]):
            prompts.append(
                EXTRACT_PROMPT.format(
                    question=row["question"], passage=str(candidate["text"])[:900]
                )
            )
            positions.append((row_index, candidate_index))
    extracted = generator.generate(prompts, batch_size=batch_size, max_new_tokens=32)
    parametric = generator.generate(
        [PARAMETRIC_PROMPT.format(question=row["question"]) for row in rows],
        batch_size=batch_size,
        max_new_tokens=32,
    )
    mutable_candidates = [
        [dict(candidate) for candidate in row["candidates"]] for row in rows
    ]
    for answer, (row_index, candidate_index) in zip(extracted, positions):
        mutable_candidates[row_index][candidate_index]["extracted_answer"] = answer
    output_rows: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        output = {key: value for key, value in row.items() if key != "candidates"}
        output["parametric_answer"] = parametric[index]
        output["candidates"] = add_group_features(
            mutable_candidates[index], parametric_answer=parametric[index]
        )
        output_rows.append(output)
    _write_jsonl(out, output_rows)


def prepare_dpr_pool(*, split_manifest: Path, dataset_id: str, out: Path) -> None:
    """Materialize the 1,100-query DPR candidate pools used by the pilot."""

    import ir_datasets

    manifest = json.loads(split_manifest.read_text(encoding="utf-8"))["datasets"]["niah"]
    split_by_qid: dict[str, str] = {
        qid: "train" for qid in manifest["pilot_train_query_ids"]
    }
    needle_by_qid: dict[str, str] = {}
    parent_by_qid: dict[str, str] = {}
    for split in ("train", "dev", "test"):
        for row in manifest["splits"][split]:
            qid = str(row["query_id"])
            needle_by_qid[qid] = str(row["needle_doc_id"])
            parent_by_qid[qid] = str(row["parent_page_id"])
            if split != "train":
                split_by_qid[qid] = split
    selected = set(split_by_qid)
    dataset = ir_datasets.load(dataset_id)
    questions: dict[str, str] = {}
    aliases: dict[str, list[str]] = {}
    for query in dataset.queries_iter():
        qid = str(query.query_id)
        if qid in selected:
            questions[qid] = str(query.text)
            aliases[qid] = [str(answer) for answer in query.answers]
    qrels: dict[str, list[tuple[str, int]]] = {qid: [] for qid in selected}
    needed_docs: set[str] = set()
    for qrel in dataset.qrels_iter():
        qid = str(qrel.query_id)
        if qid not in selected:
            continue
        doc_id = str(qrel.doc_id)
        qrels[qid].append((doc_id, int(qrel.relevance)))
        needed_docs.add(doc_id)
    documents: dict[str, tuple[str, str]] = {}
    for document in dataset.docs_iter():
        doc_id = str(document.doc_id)
        if doc_id in needed_docs:
            documents[doc_id] = (str(document.title), str(document.text))
            if len(documents) == len(needed_docs):
                break
    missing_docs = needed_docs.difference(documents)
    if missing_docs:
        raise ValueError(f"DPR pool is missing {len(missing_docs)} candidate documents")
    rows: list[dict[str, object]] = []
    for qid in sorted(selected):
        candidates = [
            {
                "candidate_id": doc_id,
                "text": documents[doc_id][1],
                "title": documents[doc_id][0],
                "source_parent_id": re.sub(r"\s+", " ", documents[doc_id][0]).strip().casefold(),
                "source": "dpr",
                "dpr_relevance": relevance,
                "dpr_rank": rank,
                "utility_grade": utility_grade(source="dpr", relevance=relevance),
            }
            for rank, (doc_id, relevance) in enumerate(qrels[qid], start=1)
        ]
        rows.append(
            {
                "query_id": qid,
                "split": split_by_qid[qid],
                "question": questions[qid],
                "answers": aliases[qid],
                "needle_doc_id": needle_by_qid[qid],
                "needle_parent_id": parent_by_qid[qid],
                "candidates": candidates,
            }
        )
    _write_jsonl(out, rows)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--split-manifest", type=Path, required=True)
    prepare.add_argument("--dataset-id", default="dpr-w100/natural-questions/dev")
    prepare.add_argument("--out", type=Path, required=True)
    generate = subparsers.add_parser("generate")
    generate.add_argument("--base-pool", type=Path, required=True)
    generate.add_argument("--out", type=Path, required=True)
    generate.add_argument("--model-id", required=True)
    generate.add_argument("--device", default="cuda:0")
    generate.add_argument("--batch-size", type=int, default=16)
    rank = subparsers.add_parser("rank")
    rank.add_argument("--generated-pool", type=Path, required=True)
    rank.add_argument("--out", type=Path, required=True)
    rank.add_argument("--embedding-model", default="ibm-granite/granite-embedding-english-r2")
    rank.add_argument("--device", default="cuda:0")
    rank.add_argument("--batch-size", type=int, default=256)
    extract = subparsers.add_parser("extract")
    extract.add_argument("--ranked-pool", type=Path, required=True)
    extract.add_argument("--out", type=Path, required=True)
    extract.add_argument("--model-id", required=True)
    extract.add_argument("--device", default="cuda:0")
    extract.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.command == "prepare":
        prepare_dpr_pool(
            split_manifest=args.split_manifest, dataset_id=args.dataset_id, out=args.out
        )
    elif args.command == "generate":
        generate_pilot_material(
            base_pool=args.base_pool,
            out=args.out,
            model_id=args.model_id,
            device=args.device,
            batch_size=args.batch_size,
        )
    elif args.command == "rank":
        rank_pilot_pool(
            generated_pool=args.generated_pool,
            out=args.out,
            embedding_model=args.embedding_model,
            device=args.device,
            batch_size=args.batch_size,
        )
    elif args.command == "extract":
        extract_pilot_features(
            ranked_pool=args.ranked_pool,
            out=args.out,
            model_id=args.model_id,
            device=args.device,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()
