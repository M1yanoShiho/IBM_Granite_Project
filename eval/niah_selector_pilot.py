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
PASSAGE_CHARS = 600
VALIDATION_PASSAGE_CHARS = 900
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
    "judge_direct_support",
    "judge_condition_coverage",
    "judge_evidence_sufficiency",
    "judge_parse_failure",
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
    "Write two short, fluent sentences about how someone should research the topic "
    "in the question. Refer to the subject only in general terms. Do not give or imply "
    "the answer. Do not include names, dates, numbers, places, titles, organizations, "
    "quoted phrases, or causal facts that could answer the question.\n"
    "Question: {question}\nNon-answering research note:"
)
SEALED_NON_ANSWER_PROMPT = (
    "Draft two concise sentences explaining that the query needs source verification "
    "and what kind of record should be consulted. Keep the subject generic. Exclude "
    "every concrete person, date, number, location, work title, institution, and factual "
    "conclusion that might resolve the query.\n"
    "Query: {question}\nVerification note:"
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
RELIABILITY_PROMPT = (
    "Assess whether the passage itself is usable evidence for the question. Do not "
    "reward topical similarity or fluent writing. Score each field with an integer "
    "from 0 to 2: direct_support (0 none, 1 partial, 2 direct), condition_coverage "
    "(0 misses or violates stated conditions, 1 partial or no explicit condition, "
    "2 covers all stated conditions), and evidence_sufficiency (0 unusable, 1 needs "
    "other evidence, 2 sufficient by itself). Return only a JSON object with exactly "
    "these three keys.\n\n"
    "Example 1\nQuestion: Who wrote The Old Man and the Sea?\n"
    "Passage: Ernest Hemingway wrote The Old Man and the Sea.\n"
    "Candidate answer extracted from this passage: Ernest Hemingway\n"
    'JSON: {{"direct_support": 2, "condition_coverage": 1, '
    '"evidence_sufficiency": 2}}\n\n'
    "Example 2\nQuestion: Who wrote The Old Man and the Sea?\n"
    "Passage: To research the novel's authorship, consult catalogues and publication records.\n"
    "Candidate answer extracted from this passage: NONE\n"
    'JSON: {{"direct_support": 0, "condition_coverage": 1, '
    '"evidence_sufficiency": 0}}\n\n'
    "Example 3\nQuestion: Does the United Kingdom policy after 2025 allow the action?\n"
    "Passage: A United States policy from 2022 allowed the action.\n"
    "Candidate answer extracted from this passage: allowed\n"
    'JSON: {{"direct_support": 0, "condition_coverage": 0, '
    '"evidence_sufficiency": 0}}\n\n'
    "Question: {question}\nPassage: {passage}\n"
    "Candidate answer extracted from this passage: {candidate_answer}\nJSON:"
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


def parse_reliability_judgment(value: str) -> dict[str, float]:
    """Parse the frozen Granite judge schema and fail closed on malformed output."""

    failure = {
        "judge_direct_support": 0.0,
        "judge_condition_coverage": 0.0,
        "judge_evidence_sufficiency": 0.0,
        "judge_parse_failure": 1.0,
    }
    match = re.search(r"\{[^{}]*\}", value, flags=re.S)
    if not match:
        return failure
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return failure
    expected = ("direct_support", "condition_coverage", "evidence_sufficiency")
    if set(payload) != set(expected):
        return failure
    scores: list[int] = []
    for key in expected:
        score = payload[key]
        if isinstance(score, bool) or not isinstance(score, int) or score not in (0, 1, 2):
            return failure
        scores.append(score)
    return {
        "judge_direct_support": scores[0] / 2.0,
        "judge_condition_coverage": scores[1] / 2.0,
        "judge_evidence_sufficiency": scores[2] / 2.0,
        "judge_parse_failure": 0.0,
    }


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
    direct_mrr: list[float] = []
    conflict_exposures: list[float] = []
    harmful = direct = noise = selected_count = 0
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
        noise += sum(int(row["utility_grade"]) == 1 for row in selected)
        first_direct = next(
            (rank for rank, row in enumerate(selected, start=1) if int(row["utility_grade"]) == 4),
            None,
        )
        direct_mrr.append(1.0 / first_direct if first_direct else 0.0)
        conflict_exposures.append(
            float(any(grade == 0 for grade in grades) and any(grade >= 3 for grade in grades))
        )
        selected_count += len(selected)
    return {
        f"ndcg@{k}": sum(ndcgs) / len(ndcgs) if ndcgs else 0.0,
        f"required_evidence_recall@{k}": (
            sum(required_recalls) / len(required_recalls) if required_recalls else 0.0
        ),
        f"harmful_rate@{k}": harmful / selected_count if selected_count else 0.0,
        f"direct_support_precision@{k}": direct / selected_count if selected_count else 0.0,
        f"noise_rate@{k}": noise / selected_count if selected_count else 0.0,
        "mrr_direct_support": sum(direct_mrr) / len(direct_mrr) if direct_mrr else 0.0,
        f"conflict_exposure@{k}": (
            sum(conflict_exposures) / len(conflict_exposures) if conflict_exposures else 0.0
        ),
    }


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    """Return Holm-adjusted p-values while preserving the input key order."""

    ranked = sorted(p_values.items(), key=lambda pair: (pair[1], pair[0]))
    adjusted_ranked: dict[str, float] = {}
    running = 0.0
    total = len(ranked)
    for rank, (name, value) in enumerate(ranked):
        running = max(running, min(1.0, (total - rank) * float(value)))
        adjusted_ranked[name] = running
    return {name: adjusted_ranked[name] for name in p_values}


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


def non_answer_passes_validator(
    text: str, *, aliases: Sequence[str], validator_output: str
) -> bool:
    """Keep a synthetic non-answer only when literal and semantic checks agree."""

    if not text.strip():
        return False
    if any(
        alias.strip().casefold() in text.casefold() for alias in aliases if alias.strip()
    ):
        return False
    return not is_valid_answer(validator_output)


def replace_non_answer_candidate(
    row: Mapping[str, object], *, passage: str, validator_output: str
) -> dict[str, object]:
    """Replace any old synthetic non-answer with one independently validated copy."""

    item = dict(row)
    candidates = [
        dict(candidate)
        for candidate in row["candidates"]
        if candidate["source"] != "generative_non_answer"
    ]
    valid = non_answer_passes_validator(
        passage,
        aliases=[str(alias) for alias in row["answers"]],
        validator_output=validator_output,
    )
    if valid:
        candidates.append(
            {
                "candidate_id": f"{row['query_id']}__non_answer",
                "text": passage.strip(),
                "title": "synthetic research note",
                "source_parent_id": f"synthetic:{row['query_id']}",
                "source": "generative_non_answer",
                "dpr_relevance": None,
                "dpr_rank": len(candidates) + 1,
                "utility_grade": 2,
            }
        )
    item["non_answer_generation_prompt_version"] = "v2_research_note"
    item["non_answer_validation_applied"] = True
    item["non_answer_validator_output"] = validator_output
    item["non_answer_valid"] = valid
    item["candidates"] = candidates
    return item


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


def validate_generated_non_answers(
    *, generated_pool: Path, out: Path, model_id: str, device: str, batch_size: int
) -> None:
    """Remove synthetic passages that Granite can answer despite literal filtering."""

    rows = _read_jsonl(generated_pool)
    prompts: list[str] = []
    positions: list[tuple[int, str]] = []
    for row_index, row in enumerate(rows):
        for candidate in row["candidates"]:
            if candidate["source"] != "generative_non_answer":
                continue
            prompts.append(
                EXTRACT_PROMPT.format(
                    question=row["question"],
                    passage=str(candidate["text"])[:VALIDATION_PASSAGE_CHARS],
                )
            )
            positions.append((row_index, str(candidate["candidate_id"])))
    generator = GraniteBatchGenerator(model_id, device=device)
    judgments = generator.generate(prompts, batch_size=batch_size, max_new_tokens=32)
    judgment_by_position = dict(zip(positions, judgments))
    validated: list[dict[str, object]] = []
    for row_index, row in enumerate(rows):
        item = dict(row)
        candidates: list[dict[str, object]] = []
        validator_output = None
        for candidate_raw in row["candidates"]:
            candidate = dict(candidate_raw)
            if candidate["source"] != "generative_non_answer":
                candidates.append(candidate)
                continue
            validator_output = judgment_by_position[
                (row_index, str(candidate["candidate_id"]))
            ]
            if non_answer_passes_validator(
                str(candidate["text"]),
                aliases=[str(alias) for alias in row["answers"]],
                validator_output=validator_output,
            ):
                candidates.append(candidate)
        item["non_answer_validation_applied"] = True
        item["non_answer_validator_output"] = validator_output
        item["non_answer_valid"] = any(
            candidate["source"] == "generative_non_answer" for candidate in candidates
        )
        item["candidates"] = candidates
        validated.append(item)
    _write_jsonl(out, validated)


def regenerate_and_validate_non_answers(
    *, generated_pool: Path, out: Path, model_id: str, device: str, batch_size: int
) -> None:
    """Generate conservative topical non-answers and retain only unanswerable passages."""

    rows = _read_jsonl(generated_pool)
    generator = GraniteBatchGenerator(model_id, device=device)
    passages = generator.generate(
        [
            (
                SEALED_NON_ANSWER_PROMPT if row["split"] == "test" else NON_ANSWER_PROMPT
            ).format(question=row["question"])
            for row in rows
        ],
        batch_size=batch_size,
        max_new_tokens=96,
    )
    validator_outputs = generator.generate(
        [
            EXTRACT_PROMPT.format(
                question=row["question"], passage=passage[:VALIDATION_PASSAGE_CHARS]
            )
            for row, passage in zip(rows, passages)
        ],
        batch_size=batch_size,
        max_new_tokens=32,
    )
    validated = [
        replace_non_answer_candidate(
            row, passage=passage, validator_output=validator_output
        )
        for row, passage, validator_output in zip(rows, passages, validator_outputs)
    ]
    _write_jsonl(out, validated)


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
    for row in rows:
        for candidate in row["candidates"]:
            prompts.append(
                EXTRACT_PROMPT.format(
                    question=row["question"], passage=str(candidate["text"])[:PASSAGE_CHARS]
                )
            )
    extracted = generator.generate(prompts, batch_size=batch_size, max_new_tokens=32)
    parametric = generator.generate(
        [PARAMETRIC_PROMPT.format(question=row["question"]) for row in rows],
        batch_size=batch_size,
        max_new_tokens=32,
    )
    answer_rows = materialize_answer_rows(rows, extracted=extracted, parametric=parametric)
    reliability_prompts = []
    for row in answer_rows:
        for candidate in row["candidates"]:
            reliability_prompts.append(
                RELIABILITY_PROMPT.format(
                    question=row["question"],
                    passage=str(candidate["text"])[:PASSAGE_CHARS],
                    candidate_answer=candidate["extracted_answer"],
                )
            )
    reliability = generator.generate(
        reliability_prompts,
        batch_size=batch_size,
        max_new_tokens=64,
    )
    _write_jsonl(out, materialize_reliability_rows(answer_rows, judgments=reliability))


def materialize_answer_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    extracted: Sequence[str],
    parametric: Sequence[str],
) -> list[dict[str, object]]:
    """Attach cached candidate and parametric answers with strict length checks."""

    expected_candidates = sum(len(row["candidates"]) for row in rows)
    if len(extracted) != expected_candidates:
        raise ValueError(
            f"received {len(extracted)} candidate answers; expected {expected_candidates}"
        )
    if len(parametric) != len(rows):
        raise ValueError(f"received {len(parametric)} parametric answers; expected {len(rows)}")
    answer_index = 0
    output: list[dict[str, object]] = []
    for row_index, row in enumerate(rows):
        item = {key: value for key, value in row.items() if key != "candidates"}
        candidates = []
        for candidate_raw in row["candidates"]:
            candidate = dict(candidate_raw)
            candidate["extracted_answer"] = extracted[answer_index]
            answer_index += 1
            candidates.append(candidate)
        item["parametric_answer"] = parametric[row_index]
        item["candidates"] = candidates
        output.append(item)
    return output


def materialize_reliability_rows(
    rows: Sequence[Mapping[str, object]], *, judgments: Sequence[str]
) -> list[dict[str, object]]:
    """Attach reliability judgments and derive final shared group features."""

    expected = sum(len(row["candidates"]) for row in rows)
    if len(judgments) != expected:
        raise ValueError(f"received {len(judgments)} judgments; expected {expected}")
    judgment_index = 0
    output_rows: list[dict[str, object]] = []
    for row in rows:
        item = {key: value for key, value in row.items() if key != "candidates"}
        candidates = []
        for candidate_raw in row["candidates"]:
            candidate = dict(candidate_raw)
            judgment = judgments[judgment_index]
            judgment_index += 1
            candidate["reliability_judgment"] = judgment
            candidate.update(parse_reliability_judgment(judgment))
            candidates.append(candidate)
        item["candidates"] = add_group_features(
            candidates, parametric_answer=str(row["parametric_answer"])
        )
        output_rows.append(item)
    return output_rows


def extract_pilot_answers(
    *, ranked_pool: Path, out: Path, model_id: str, device: str, batch_size: int
) -> None:
    """Persist candidate and parametric answers as a restartable first stage."""

    rows = _read_jsonl(ranked_pool)
    generator = GraniteBatchGenerator(model_id, device=device)
    extracted = generator.generate(
        [
            EXTRACT_PROMPT.format(
                question=row["question"], passage=str(candidate["text"])[:PASSAGE_CHARS]
            )
            for row in rows
            for candidate in row["candidates"]
        ],
        batch_size=batch_size,
        max_new_tokens=32,
    )
    parametric = generator.generate(
        [PARAMETRIC_PROMPT.format(question=row["question"]) for row in rows],
        batch_size=batch_size,
        max_new_tokens=32,
    )
    _write_jsonl(out, materialize_answer_rows(rows, extracted=extracted, parametric=parametric))


def judge_pilot_reliability(
    *, answer_cache: Path, out: Path, model_id: str, device: str, batch_size: int
) -> None:
    """Resume from answer cache and persist final reliability-aware features."""

    rows = _read_jsonl(answer_cache)
    generator = GraniteBatchGenerator(model_id, device=device)
    judgments = generator.generate(
        [
            RELIABILITY_PROMPT.format(
                question=row["question"],
                passage=str(candidate["text"])[:PASSAGE_CHARS],
                candidate_answer=candidate["extracted_answer"],
            )
            for row in rows
            for candidate in row["candidates"]
        ],
        batch_size=batch_size,
        max_new_tokens=64,
    )
    _write_jsonl(out, materialize_reliability_rows(rows, judgments=judgments))


def _groups_by_split(rows: Sequence[Mapping[str, object]]) -> dict[str, dict[str, list[dict[str, object]]]]:
    output: dict[str, dict[str, list[dict[str, object]]]] = {
        "train": {},
        "dev": {},
        "test": {},
    }
    for row in rows:
        output[str(row["split"])][str(row["query_id"])] = [
            dict(candidate) for candidate in row["candidates"]
        ]
    return output


def _attach_blend_scores(
    groups: Mapping[str, list[dict[str, object]]], *, alpha: float
) -> None:
    for rows in groups.values():
        correlation = minmax([float(row["exact_vote_count"]) for row in rows])
        source_correlation = minmax(
            [float(row["source_dedup_vote_count"]) for row in rows]
        )
        for index, row in enumerate(rows):
            relevance = float(row["relevance_normalized"])
            row["q2d_score"] = float(row["relevance_score"])
            row["fixed_score"] = alpha * relevance + (1.0 - alpha) * correlation[index]
            row["source_dedup_fixed_score"] = (
                alpha * relevance + (1.0 - alpha) * source_correlation[index]
            )


def _flatten_training(
    groups: Mapping[str, Sequence[Mapping[str, object]]], features: Sequence[str]
):
    import numpy as np

    ordered = [(qid, groups[qid]) for qid in sorted(groups)]
    x = np.asarray(
        [[float(row[feature]) for feature in features] for _, rows in ordered for row in rows],
        dtype=np.float32,
    )
    y = np.asarray(
        [int(row["utility_grade"]) for _, rows in ordered for row in rows], dtype=np.int32
    )
    group = [len(rows) for _, rows in ordered]
    return x, y, group, ordered


def _fit_ranker(
    train_groups: Mapping[str, Sequence[Mapping[str, object]]],
    dev_groups: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    features: Sequence[str],
    seed: int,
    shuffled_labels: bool = False,
):
    import lightgbm as lgb
    import numpy as np

    x_train, y_train, train_sizes, _ = _flatten_training(train_groups, features)
    x_dev, y_dev, dev_sizes, _ = _flatten_training(dev_groups, features)
    if shuffled_labels:
        rng = np.random.default_rng(seed)
        offset = 0
        for size in train_sizes:
            rng.shuffle(y_train[offset : offset + size])
            offset += size
    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        label_gain=list(LABEL_GAINS),
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=15,
        max_depth=5,
        min_child_samples=20,
        feature_fraction=0.9,
        reg_lambda=1.0,
        random_state=seed,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )
    model.fit(
        x_train,
        y_train,
        group=train_sizes,
        eval_set=[(x_dev, y_dev)],
        eval_group=[dev_sizes],
        eval_at=[10],
        callbacks=[lgb.early_stopping(40, verbose=False)],
    )
    return model


def _predict_groups(model, groups: Mapping[str, list[dict[str, object]]], features, key: str) -> None:
    import numpy as np
    import warnings

    ordered = [(qid, groups[qid]) for qid in sorted(groups)]
    x = np.asarray(
        [[float(row[feature]) for feature in features] for _, rows in ordered for row in rows],
        dtype=np.float32,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names")
        predictions = model.predict(x, num_iteration=model.best_iteration_)
    offset = 0
    for _, rows in ordered:
        for row, prediction in zip(rows, predictions[offset : offset + len(rows)]):
            row[key] = float(prediction)
        offset += len(rows)


def _query_metric_vector(
    groups: Mapping[str, Sequence[Mapping[str, object]]], *, score_key: str, k: int
) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for qid, rows in groups.items():
        ranked = rank_candidates(rows, score_key)
        selected = ranked[:k]
        grades = [int(row["utility_grade"]) for row in selected]
        ideal = sorted((int(row["utility_grade"]) for row in rows), reverse=True)[:k]
        required_total = sum(int(row["utility_grade"]) >= 3 for row in rows)
        output[qid] = {
            "ndcg": _dcg(grades) / _dcg(ideal) if _dcg(ideal) else 0.0,
            "required_recall": (
                sum(int(row["utility_grade"]) >= 3 for row in selected) / required_total
                if required_total
                else 1.0
            ),
            "harmful_rate": sum(int(row["utility_grade"]) == 0 for row in selected)
            / max(1, len(selected)),
        }
    return output


def _paired_bootstrap(
    reference: Mapping[str, Mapping[str, float]],
    method: Mapping[str, Mapping[str, float]],
    *,
    metric: str,
    improvement_sign: float,
    seed: int = 42,
    samples: int = 5000,
) -> dict[str, float]:
    import numpy as np

    qids = sorted(set(reference).intersection(method))
    delta = np.asarray(
        [improvement_sign * (method[qid][metric] - reference[qid][metric]) for qid in qids]
    )
    rng = np.random.default_rng(seed)
    boot = np.empty(samples, dtype=float)
    for index in range(samples):
        boot[index] = delta[rng.integers(0, len(delta), len(delta))].mean()
    return {
        "delta": float(delta.mean()),
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
        "p_two_sided": float(
            min(1.0, 2.0 * min((boot <= 0).mean(), (boot >= 0).mean()))
        ),
    }


def train_and_evaluate(*, feature_cache: Path, out_dir: Path) -> None:
    """Train Core/Full LambdaRank models and run the sealed controlled test once."""

    import numpy as np
    from sklearn.linear_model import LogisticRegression

    all_groups = _groups_by_split(_read_jsonl(feature_cache))
    for groups in all_groups.values():
        _attach_blend_scores(groups, alpha=0.6)

    alpha_grid = [index / 20 for index in range(21)]
    alpha_scores: list[tuple[float, float]] = []
    for alpha in alpha_grid:
        for rows in all_groups["dev"].values():
            corr = minmax([float(row["exact_vote_count"]) for row in rows])
            for index, row in enumerate(rows):
                row["alpha_tune_score"] = alpha * float(row["relevance_normalized"]) + (
                    1.0 - alpha
                ) * corr[index]
        score = selector_metrics(all_groups["dev"], score_key="alpha_tune_score", k=10)[
            "ndcg@10"
        ]
        alpha_scores.append((alpha, score))
    alpha_star = max(alpha_scores, key=lambda pair: (pair[1], pair[0]))[0]
    for split_groups in all_groups.values():
        for rows in split_groups.values():
            corr = minmax([float(row["exact_vote_count"]) for row in rows])
            for index, row in enumerate(rows):
                row["alpha_star_score"] = alpha_star * float(row["relevance_normalized"]) + (
                    1.0 - alpha_star
                ) * corr[index]

    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = (13, 42, 73)
    models: dict[str, list[object]] = {"core": [], "full": []}
    for name, features in (("core", CORE_FEATURES), ("full", FULL_FEATURES)):
        for seed in seeds:
            model = _fit_ranker(
                all_groups["train"], all_groups["dev"], features=features, seed=seed
            )
            model.booster_.save_model(str(out_dir / f"{name}_seed{seed}.txt"))
            models[name].append(model)
        for split in ("dev", "test"):
            temporary_keys = []
            for seed, model in zip(seeds, models[name]):
                temporary_key = f"__{name}_seed_{seed}"
                _predict_groups(
                    model,
                    all_groups[split],
                    features,
                    temporary_key,
                )
                temporary_keys.append(temporary_key)
            for rows in all_groups[split].values():
                for row in rows:
                    row[f"ml_{name}_score"] = float(
                        np.mean([float(row.pop(key)) for key in temporary_keys])
                    )

    x_train, y_train_grade, _, train_order = _flatten_training(
        all_groups["train"], CORE_FEATURES
    )
    y_train = (y_train_grade >= 3).astype(int)
    logistic = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    logistic.fit(x_train, y_train)
    for split in ("dev", "test"):
        for rows in all_groups[split].values():
            x = np.asarray(
                [[float(row[feature]) for feature in CORE_FEATURES] for row in rows],
                dtype=np.float32,
            )
            for row, score in zip(rows, logistic.predict_proba(x)[:, 1]):
                row["logistic_score"] = float(score)

    shuffled = _fit_ranker(
        all_groups["train"],
        all_groups["dev"],
        features=FULL_FEATURES,
        seed=42,
        shuffled_labels=True,
    )
    _predict_groups(shuffled, all_groups["test"], FULL_FEATURES, "shuffled_label_score")

    methods = {
        "q2d": "q2d_score",
        "fixed_0.6": "fixed_score",
        "alpha_star": "alpha_star_score",
        "source_dedup_fixed": "source_dedup_fixed_score",
        "logistic": "logistic_score",
        "ml_core": "ml_core_score",
        "ml_full": "ml_full_score",
        "shuffled_label": "shuffled_label_score",
    }
    metrics = {
        split: {
            method: selector_metrics(all_groups[split], score_key=score_key, k=10)
            for method, score_key in methods.items()
            if all(score_key in row for rows in all_groups[split].values() for row in rows)
        }
        for split in ("dev", "test")
    }
    fixed_vector = _query_metric_vector(
        all_groups["test"], score_key="fixed_score", k=10
    )
    full_vector = _query_metric_vector(
        all_groups["test"], score_key="ml_full_score", k=10
    )
    statistics = {
        "ndcg_improvement": _paired_bootstrap(
            fixed_vector, full_vector, metric="ndcg", improvement_sign=1.0
        ),
        "harmful_rate_reduction": _paired_bootstrap(
            fixed_vector, full_vector, metric="harmful_rate", improvement_sign=-1.0
        ),
        "required_recall_change": _paired_bootstrap(
            fixed_vector, full_vector, metric="required_recall", improvement_sign=1.0
        ),
    }
    adjusted = holm_adjust(
        {name: values["p_two_sided"] for name, values in statistics.items()}
    )
    for name, value in adjusted.items():
        statistics[name]["holm_adjusted_p"] = value
    gate1 = {
        "ndcg_delta_at_least_0.02": statistics["ndcg_improvement"]["delta"] >= 0.02,
        "harmful_reduction_at_least_0.02": statistics["harmful_rate_reduction"]["delta"]
        >= 0.02,
        "ndcg_ci_lower_above_zero": statistics["ndcg_improvement"]["ci_low"] > 0,
        "harmful_ci_lower_above_zero": statistics["harmful_rate_reduction"]["ci_low"] > 0,
        "ndcg_holm_p_below_0.05": statistics["ndcg_improvement"]["holm_adjusted_p"]
        < 0.05,
        "harmful_holm_p_below_0.05": statistics["harmful_rate_reduction"][
            "holm_adjusted_p"
        ]
        < 0.05,
        "required_recall_noninferior": statistics["required_recall_change"]["ci_low"]
        > -0.01,
    }
    gate1["passed"] = all(gate1.values())
    result = {
        "protocol": {
            "candidate_pool": "DPR per-query hard pool reranked by Query2Doc + Granite",
            "top_n": 20,
            "context_k": 10,
            "passage_chars": PASSAGE_CHARS,
            "fixed_alpha": 0.6,
            "alpha_star": alpha_star,
            "seeds": list(seeds),
            "core_features": list(CORE_FEATURES),
            "full_features": list(FULL_FEATURES),
        },
        "metrics": metrics,
        "statistics": statistics,
        "gate1": gate1,
    }
    (out_dir / "results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (out_dir / "test_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        fieldnames = ["query_id", "candidate_id", "utility_grade", *methods]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for qid in sorted(all_groups["test"]):
            for row in all_groups["test"][qid]:
                writer.writerow(
                    {
                        "query_id": qid,
                        "candidate_id": row["candidate_id"],
                        "utility_grade": row["utility_grade"],
                        **{method: row[score_key] for method, score_key in methods.items()},
                    }
                )


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
    validate = subparsers.add_parser("validate-non-answers")
    validate.add_argument("--generated-pool", type=Path, required=True)
    validate.add_argument("--out", type=Path, required=True)
    validate.add_argument("--model-id", required=True)
    validate.add_argument("--device", default="cuda:0")
    validate.add_argument("--batch-size", type=int, default=32)
    regenerate = subparsers.add_parser("regenerate-non-answers")
    regenerate.add_argument("--generated-pool", type=Path, required=True)
    regenerate.add_argument("--out", type=Path, required=True)
    regenerate.add_argument("--model-id", required=True)
    regenerate.add_argument("--device", default="cuda:0")
    regenerate.add_argument("--batch-size", type=int, default=32)
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
    extract_answers = subparsers.add_parser("extract-answers")
    extract_answers.add_argument("--ranked-pool", type=Path, required=True)
    extract_answers.add_argument("--out", type=Path, required=True)
    extract_answers.add_argument("--model-id", required=True)
    extract_answers.add_argument("--device", default="cuda:0")
    extract_answers.add_argument("--batch-size", type=int, default=32)
    judge_reliability = subparsers.add_parser("judge-reliability")
    judge_reliability.add_argument("--answer-cache", type=Path, required=True)
    judge_reliability.add_argument("--out", type=Path, required=True)
    judge_reliability.add_argument("--model-id", required=True)
    judge_reliability.add_argument("--device", default="cuda:0")
    judge_reliability.add_argument("--batch-size", type=int, default=32)
    train = subparsers.add_parser("train")
    train.add_argument("--feature-cache", type=Path, required=True)
    train.add_argument("--out-dir", type=Path, required=True)
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
    elif args.command == "validate-non-answers":
        validate_generated_non_answers(
            generated_pool=args.generated_pool,
            out=args.out,
            model_id=args.model_id,
            device=args.device,
            batch_size=args.batch_size,
        )
    elif args.command == "regenerate-non-answers":
        regenerate_and_validate_non_answers(
            generated_pool=args.generated_pool,
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
    elif args.command == "extract-answers":
        extract_pilot_answers(
            ranked_pool=args.ranked_pool,
            out=args.out,
            model_id=args.model_id,
            device=args.device,
            batch_size=args.batch_size,
        )
    elif args.command == "judge-reliability":
        judge_pilot_reliability(
            answer_cache=args.answer_cache,
            out=args.out,
            model_id=args.model_id,
            device=args.device,
            batch_size=args.batch_size,
        )
    elif args.command == "train":
        train_and_evaluate(feature_cache=args.feature_cache, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
