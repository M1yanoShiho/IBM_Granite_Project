"""Materialize the 2026-08-18 G200 v2 generator data pre-audit bundle.

This script does not train a model and does not read sealed/system-heldout data.
It writes runtime-sized train/model-val case files plus small manifests that can
be archived in git.  TRUE, citation, minimal-support, manual, and length audits
belong to G210/G300 and must pass before training starts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from full_flow_g200 import (
    QA2D_MODEL_ID,
    QA2D_REVISION,
    _contains_normalised,
    _is_unknown_reference,
    _sha256_text,
)

from evidence_rag.generator.draft import DRAFT_PROMPT

TRAIN_ROLE = "train-fit"
VALIDATION_ROLE = "train-modelval"
UNKNOWN_ANSWER = "I don't know."
SCHEMA_CASE = "full-flow-g200-case-v2"
SCHEMA_MANIFEST = "full-flow-g200-v2-data-manifest-v1"
SCHEMA_NIAH_PREPARE = "full-flow-g200-v2-niah-modelval-prepare-v1"


def _jsonl(path: Path) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, Mapping):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        rows.append(value)
    return rows


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_empty(path: Path, label: str) -> None:
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"{label} output directory must be absent or empty")


def _normalise(text: str) -> str:
    value = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.findall(r"\w+", value))


def _key_string(axis: str, value: str) -> str:
    return f"{axis}:{value}"


def _component_id(query_ids: Sequence[str]) -> str:
    if not query_ids:
        raise ValueError("component query IDs must be non-empty")
    payload = "".join(f"{query_id}\n" for query_id in sorted(query_ids))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class _Dsu:
    def __init__(self, members: Iterable[str]) -> None:
        self.parent = {member: member for member in members}

    def find(self, member: str) -> str:
        parent = self.parent[member]
        if parent != member:
            self.parent[member] = self.find(parent)
        return self.parent[member]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        low, high = sorted((left_root, right_root))
        self.parent[high] = low


def _load_map(path: Path, key: str) -> dict[str, Mapping[str, Any]]:
    output: dict[str, Mapping[str, Any]] = {}
    for row in _jsonl(path):
        row_key = str(row.get(key, ""))
        if not row_key:
            raise ValueError(f"{path} has a row without {key}")
        if row_key in output:
            raise ValueError(f"duplicate {key}: {row_key}")
        output[row_key] = row
    return output


def _load_source_parent(path: Path) -> dict[str, str]:
    output: dict[str, str] = {}
    for row in _jsonl(path):
        document_id = str(row.get("document_id", ""))
        parent = str(row.get("source_parent_id", ""))
        if not document_id or not parent:
            raise ValueError(f"invalid source-parent row in {path}")
        if document_id in output:
            raise ValueError(f"duplicate source-parent document: {document_id}")
        output[document_id] = parent
    return output


def _topk(candidates: Sequence[Mapping[str, Any]], *, k: int = 10) -> list[Mapping[str, Any]]:
    ordered = sorted(
        candidates,
        key=lambda item: (int(item["retrieval_rank"]), str(item["evidence_id"])),
    )[:k]
    expected = list(range(1, k + 1))
    ranks = [int(item["retrieval_rank"]) for item in ordered]
    if ranks != expected:
        raise ValueError(f"candidate set does not have exact TopK{k}: {ranks}")
    return ordered


def _context(candidates: Sequence[Mapping[str, Any]]) -> str:
    return "\n".join(
        f"[{index}] ({item['evidence_id']}) {item['text']}"
        for index, item in enumerate(candidates, start=1)
    )


def _prompt(question: str, candidates: Sequence[Mapping[str, Any]]) -> str:
    return DRAFT_PROMPT.format(context=_context(candidates), question=question)


def _target_sentence(sentence: str, citation_index: int) -> str:
    text = sentence.strip()
    punctuation = text[-1] if text[-1:] in {".", "!", "?"} else "."
    if text[-1:] in {".", "!", "?"}:
        text = text[:-1].rstrip()
    return f"{text} [{citation_index}]{punctuation}"


def _target_chain(sentences: Sequence[str], evidence_ids: Sequence[str], context: Sequence[Mapping[str, Any]]) -> str:
    index_by_evidence = {
        str(item["evidence_id"]): index for index, item in enumerate(context, start=1)
    }
    parts: list[str] = []
    for sentence, evidence_id in zip(sentences, evidence_ids, strict=True):
        if evidence_id not in index_by_evidence:
            raise ValueError(f"context lacks support evidence {evidence_id}")
        parts.append(_target_sentence(sentence, index_by_evidence[evidence_id]))
    return " ".join(parts)


def _variant(
    *,
    question: str,
    semantic_sentences: Sequence[str],
    support_evidence_ids: Sequence[str],
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    return {
        "evidence_ids": [str(item["evidence_id"]) for item in candidates],
        "support_evidence_ids": list(support_evidence_ids),
        "prompt": _prompt(question, candidates),
        "target": _target_chain(semantic_sentences, support_evidence_ids, candidates),
    }


def _support_context_variants(
    *,
    question: str,
    semantic_sentences: Sequence[str],
    support_candidates: Sequence[Mapping[str, Any]],
    support_evidence_ids: Sequence[str],
    topk: Sequence[Mapping[str, Any]],
    include_niah_noise: bool,
) -> dict[str, dict[str, object]]:
    support_ids = {str(item["evidence_id"]) for item in support_candidates}
    distractors = [item for item in topk if str(item["evidence_id"]) not in support_ids]
    middle = len(distractors) // 2
    contexts: dict[str, list[Mapping[str, Any]]] = {
        "support_only": list(support_candidates),
        "topk": list(topk),
        "support_first": [*support_candidates, *distractors],
        "support_middle": [*distractors[:middle], *support_candidates, *distractors[middle:]],
        "support_last": [*distractors, *support_candidates],
    }
    if include_niah_noise:
        harmful = [
            item
            for item in distractors
            if _is_harmful_candidate(item)
        ]
        benign = [
            item
            for item in distractors
            if not _is_harmful_candidate(item)
        ]
        if harmful:
            contexts["support_harmful"] = sorted(
                [*support_candidates, *harmful],
                key=lambda item: (int(item["retrieval_rank"]), str(item["evidence_id"])),
            )
        if benign:
            contexts["support_benign"] = sorted(
                [*support_candidates, *benign],
                key=lambda item: (int(item["retrieval_rank"]), str(item["evidence_id"])),
            )
    return {
        name: _variant(
            question=question,
            semantic_sentences=semantic_sentences,
            support_evidence_ids=support_evidence_ids,
            candidates=context,
        )
        for name, context in contexts.items()
    }


def _is_harmful_candidate(item: Mapping[str, Any]) -> bool:
    return str(item.get("document_id", "")).startswith("cf::") and str(
        item.get("source_uri", "")
    ).startswith("synthetic://cf/")


def _old_parent_keys(component_map_path: Path) -> set[str]:
    parents: set[str] = set()
    for row in _jsonl(component_map_path):
        raw_keys = row.get("allowed_keys")
        if not isinstance(raw_keys, list):
            raise ValueError("old component map has no allowed_keys list")
        for item in raw_keys:
            if not isinstance(item, Mapping):
                raise ValueError("old component allowed_keys must be objects")
            if item.get("axis") == "parent":
                parents.add(str(item.get("value", "")))
    return parents


def _old_query_ids(role_assignments_path: Path) -> set[str]:
    return {str(row["query_id"]) for row in _jsonl(role_assignments_path)}


def _prepare_components(keys_by_query: Mapping[str, Sequence[tuple[str, str]]]) -> dict[str, str]:
    dsu = _Dsu(keys_by_query)
    owner_by_key: dict[tuple[str, str], str] = {}
    for query_id in sorted(keys_by_query):
        for key in keys_by_query[query_id]:
            owner = owner_by_key.setdefault(key, query_id)
            dsu.union(query_id, owner)
    members: dict[str, list[str]] = {}
    for query_id in sorted(keys_by_query):
        members.setdefault(dsu.find(query_id), []).append(query_id)
    component_by_query: dict[str, str] = {}
    for query_ids in members.values():
        component_id = _component_id(query_ids)
        for query_id in query_ids:
            component_by_query[query_id] = component_id
    return component_by_query


def prepare_niah_modelval(
    *,
    queries_path: Path,
    candidate_pool_path: Path,
    gold_path: Path,
    source_parent_path: Path,
    old_role_assignments_path: Path,
    old_component_map_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    _require_empty(output_dir, "G200 v2 NIAH model-val prepare")
    queries = _load_map(queries_path, "query_id")
    pools = _load_map(candidate_pool_path, "query_id")
    gold = _load_map(gold_path, "query_id")
    parents = _load_source_parent(source_parent_path)
    old_qids = _old_query_ids(old_role_assignments_path)
    denied_parents = _old_parent_keys(old_component_map_path)

    counts: Counter[str] = Counter()
    selected: list[str] = []
    keys_by_query: dict[str, tuple[tuple[str, str], ...]] = {}
    for query_id in sorted(queries):
        if query_id in old_qids:
            continue
        counts["candidate_old_excluded_remaining"] += 1
        gold_row = gold.get(query_id)
        pool_row = pools.get(query_id)
        if gold_row is None or pool_row is None:
            counts["excluded_missing_gold_or_pool"] += 1
            continue
        references = gold_row.get("reference_answers")
        relevant = gold_row.get("relevant_document_ids")
        if not isinstance(references, list) or len(references) != 1:
            counts["excluded_bad_reference"] += 1
            continue
        answer = str(references[0]).strip()
        if not answer or _is_unknown_reference(answer):
            counts["excluded_bad_reference"] += 1
            continue
        if not isinstance(relevant, list) or not relevant:
            counts["excluded_no_relevant_documents"] += 1
            continue
        relevant_ids = {str(item) for item in relevant}
        if any(document_id not in parents for document_id in relevant_ids):
            counts["excluded_missing_source_parent"] += 1
            continue
        parent_values = {parents[document_id] for document_id in relevant_ids}
        if parent_values & denied_parents:
            counts["excluded_old_parent_overlap"] += 1
            continue
        candidates = pool_row.get("candidates")
        if not isinstance(candidates, list):
            counts["excluded_bad_candidate_set"] += 1
            continue
        topk = _topk(candidates)
        support = [item for item in topk if str(item.get("document_id", "")) in relevant_ids]
        if not support:
            counts["excluded_no_top10_support"] += 1
            continue
        if not any(_contains_normalised(answer, str(item.get("text", ""))) for item in support):
            counts["excluded_support_lacks_answer"] += 1
            continue
        distractors = [
            item for item in topk if str(item.get("document_id", "")) not in relevant_ids
        ]
        if not any(_is_harmful_candidate(item) for item in distractors):
            counts["excluded_no_harmful"] += 1
            continue
        if not any(not _is_harmful_candidate(item) for item in distractors):
            counts["excluded_no_benign"] += 1
            continue
        selected.append(query_id)
        keys_by_query[query_id] = tuple(
            [("query", query_id)]
            + [("parent", parent) for parent in sorted(parent_values)]
        )

    if not selected:
        raise ValueError("no NIAH model-val candidates survived G200 v2 prepare")
    component_by_query = _prepare_components(keys_by_query)
    role_rows: list[dict[str, object]] = []
    component_rows: list[dict[str, object]] = []
    component_sizes = Counter(component_by_query.values())
    for query_id in selected:
        component_id = component_by_query[query_id]
        key_rows = [
            {"axis": axis, "value": value, "key": _key_string(axis, value)}
            for axis, value in keys_by_query[query_id]
        ]
        role_rows.append(
            {
                "schema_version": "1.0",
                "dataset_id": "niah/dpr-w100-nq",
                "query_id": query_id,
                "component_id": component_id,
                "fold": 0,
                "role": VALIDATION_ROLE,
                "chain_eligible_topk10": True,
                "g200_v2_split_source": "old-1023-excluded-parent-disjoint",
            }
        )
        component_rows.append(
            {
                "schema_version": "1.0",
                "dataset_id": "niah/dpr-w100-nq",
                "query_id": query_id,
                "component_id": component_id,
                "component_size": component_sizes[component_id],
                "component_root": min(item["key"] for item in key_rows),
                "allowed_keys": key_rows,
            }
        )
    _write_jsonl(output_dir / "role_assignments.jsonl", role_rows)
    _write_jsonl(output_dir / "component_map.jsonl", component_rows)
    summary: dict[str, object] = {
        "schema_version": SCHEMA_NIAH_PREPARE,
        "status": "COMPLETE",
        "selected_queries": len(selected),
        "selected_components": len(set(component_by_query.values())),
        "old_query_ids_excluded": len(old_qids),
        "old_parent_keys_denied": len(denied_parents),
        "counts": dict(counts),
        "input_sha256": {
            "queries": _sha256(queries_path),
            "candidate_pool": _sha256(candidate_pool_path),
            "gold": _sha256(gold_path),
            "source_parent": _sha256(source_parent_path),
            "old_role_assignments": _sha256(old_role_assignments_path),
            "old_component_map": _sha256(old_component_map_path),
        },
        "role_assignments_sha256": _sha256(output_dir / "role_assignments.jsonl"),
        "component_map_sha256": _sha256(output_dir / "component_map.jsonl"),
        "ordered_query_ids": selected,
        "ordered_query_ids_sha256": _sha256_text("\n".join(selected) + "\n"),
    }
    _write_json(output_dir / "summary.json", summary)
    return summary


def _convert_old_niah_train(row: Mapping[str, Any]) -> dict[str, object]:
    query_id = str(row.get("query_id", ""))
    variants = row.get("variants")
    if not query_id or row.get("schema_version") != "full-flow-g200-case-v1":
        raise ValueError(f"invalid old NIAH G200 row: {query_id!r}")
    if row.get("role") != TRAIN_ROLE or not isinstance(variants, Mapping):
        raise ValueError(f"old NIAH reuse accepts train-fit cases only: {query_id}")
    converted_variants: dict[str, dict[str, object]] = {}
    for name, variant in variants.items():
        if not isinstance(variant, Mapping):
            raise ValueError(f"invalid old NIAH variant {query_id}/{name}")
        copy = dict(variant)
        support = copy.get("support_evidence_id")
        if support is not None and "support_evidence_ids" not in copy:
            copy["support_evidence_ids"] = [str(support)]
        converted_variants[str(name)] = copy
    return {
        "schema_version": SCHEMA_CASE,
        "case_id": f"niah-old::{query_id}",
        "dataset": "niah",
        "source": "G200-v1-reviewed-train-fit-reuse",
        "group_id": query_id,
        "query_id": query_id,
        "role": TRAIN_ROLE,
        "component_id": str(row.get("component_id", "")),
        "answerable": True,
        "target_kind": "niah_qa2d_single_claim",
        "question": str(row.get("question", "")),
        "answer": str(row.get("answer", "")),
        "semantic_target": str(row.get("semantic_target", "")),
        "semantic_target_sha256": str(row.get("semantic_target_sha256", "")),
        "variants": converted_variants,
    }


def _load_qa2d_targets(path: Path) -> dict[str, Mapping[str, Any]]:
    targets: dict[str, Mapping[str, Any]] = {}
    for row in _jsonl(path):
        query_id = str(row.get("query_id", ""))
        if row.get("model") != QA2D_MODEL_ID or row.get("revision") != QA2D_REVISION:
            raise ValueError(f"QA2D identity mismatch for {query_id}")
        if query_id in targets:
            raise ValueError(f"duplicate QA2D target: {query_id}")
        targets[query_id] = row
    return targets


def _materialize_new_niah_modelval(
    *,
    queries_path: Path,
    candidate_pool_path: Path,
    gold_path: Path,
    role_assignments_path: Path,
    component_map_path: Path,
    qa2d_targets_path: Path,
) -> tuple[list[dict[str, object]], Counter[str]]:
    queries = _load_map(queries_path, "query_id")
    pools = _load_map(candidate_pool_path, "query_id")
    gold = _load_map(gold_path, "query_id")
    roles = _load_map(role_assignments_path, "query_id")
    components = _load_map(component_map_path, "query_id")
    targets = _load_qa2d_targets(qa2d_targets_path)
    counts: Counter[str] = Counter()
    rows: list[dict[str, object]] = []
    for query_id in sorted(roles):
        counts["role_assigned"] += 1
        role = roles[query_id]
        if role.get("role") != VALIDATION_ROLE:
            raise ValueError("new NIAH model-val roles must all be train-modelval")
        target = targets.get(query_id)
        if target is None:
            counts["excluded_missing_qa2d"] += 1
            continue
        if not bool(target.get("answer_preserved")):
            counts["excluded_qa2d_answer_not_preserved"] += 1
            continue
        answer = str(target.get("answer", ""))
        declarative = str(target.get("declarative", "")).strip()
        if not answer or not declarative or _is_unknown_reference(answer):
            counts["excluded_bad_target"] += 1
            continue
        gold_row = gold[query_id]
        relevant = {str(item) for item in gold_row.get("relevant_document_ids", [])}
        topk = _topk(pools[query_id]["candidates"])
        support = [
            item
            for item in topk
            if str(item.get("document_id", "")) in relevant
            and _contains_normalised(answer, str(item.get("text", "")))
        ]
        if not support:
            counts["excluded_no_supported_answer_carrier"] += 1
            continue
        carrier = support[0]
        variants = _support_context_variants(
            question=str(queries[query_id]["text"]),
            semantic_sentences=[declarative],
            support_candidates=[carrier],
            support_evidence_ids=[str(carrier["evidence_id"])],
            topk=topk,
            include_niah_noise=True,
        )
        rows.append(
            {
                "schema_version": SCHEMA_CASE,
                "case_id": f"niah-new-modelval::{query_id}",
                "dataset": "niah",
                "source": "G200-v2-new-parent-disjoint-modelval",
                "group_id": query_id,
                "query_id": query_id,
                "role": VALIDATION_ROLE,
                "component_id": str(components[query_id]["component_id"]),
                "answerable": True,
                "target_kind": "niah_qa2d_single_claim",
                "question": str(queries[query_id]["text"]),
                "answer": answer,
                "semantic_target": declarative,
                "semantic_target_sha256": _sha256_text(declarative),
                "variants": variants,
            }
        )
        counts["materialized"] += 1
    return rows, counts


def _as_json(value: object) -> Any:
    return json.loads(value) if isinstance(value, (str, bytes)) else value


def _load_twowiki_rows(path: Path) -> dict[str, Mapping[str, Any]]:
    if path.suffix == ".jsonl":
        return {str(row["_id"]): row for row in _jsonl(path)}
    pandas = __import__("pandas")
    frame = pandas.read_parquet(path)
    return {str(row["_id"]): row for row in frame.to_dict("records")}


def _triple_sentence(triple: Sequence[object]) -> str:
    if len(triple) != 3:
        raise ValueError(f"2Wiki evidence triple must have length 3: {triple!r}")
    subject = str(triple[0]).strip()
    relation = str(triple[1]).strip().replace("_", " ")
    obj = str(triple[2]).strip()
    if not subject or not relation or not obj:
        raise ValueError(f"2Wiki evidence triple contains a blank value: {triple!r}")
    relation = " ".join(relation.split())
    return f"{subject}'s {relation} is {obj}."


def _twowiki_case(
    *,
    query_id: str,
    role: str,
    component_id: str,
    role_row: Mapping[str, Any],
    query_text: str,
    answer: str,
    topk: Sequence[Mapping[str, Any]],
    official_row: Mapping[str, Any],
) -> dict[str, object] | None:
    supporting_facts = _as_json(official_row.get("supporting_facts"))
    evidences = _as_json(official_row.get("evidences"))
    context = _as_json(official_row.get("context"))
    if (
        not isinstance(supporting_facts, list)
        or not isinstance(evidences, list)
        or not isinstance(context, list)
        or len(supporting_facts) != len(evidences)
        or not evidences
    ):
        return None
    context_by_title: dict[str, Sequence[object]] = {}
    for item in context:
        if not isinstance(item, Sequence) or len(item) < 2:
            return None
        context_by_title[str(item[0])] = item[1]  # type: ignore[assignment]
    candidate_by_document = {str(item["document_id"]): item for item in topk}
    support_candidates: list[Mapping[str, Any]] = []
    semantic_sentences: list[str] = []
    supporting_fact_rows: list[dict[str, object]] = []
    for fact, triple in zip(supporting_facts, evidences, strict=True):
        if not isinstance(fact, Sequence) or len(fact) < 2:
            return None
        title = str(fact[0])
        sentence_index = int(fact[1])
        sentences = context_by_title.get(title)
        if (
            sentences is None
            or isinstance(sentences, str)
            or sentence_index < 0
            or sentence_index >= len(sentences)
            or title not in candidate_by_document
        ):
            return None
        support_candidates.append(candidate_by_document[title])
        semantic_sentences.append(_triple_sentence(triple))
        supporting_fact_rows.append(
            {
                "title": title,
                "sentence_index": sentence_index,
                "sentence": str(sentences[sentence_index]),
            }
        )
    dedup_support: list[Mapping[str, Any]] = []
    seen_support: set[str] = set()
    for candidate in support_candidates:
        evidence_id = str(candidate["evidence_id"])
        if evidence_id not in seen_support:
            dedup_support.append(candidate)
            seen_support.add(evidence_id)
    support_evidence_ids = [str(item["evidence_id"]) for item in support_candidates]
    variants = _support_context_variants(
        question=query_text,
        semantic_sentences=semantic_sentences,
        support_candidates=dedup_support,
        support_evidence_ids=support_evidence_ids,
        topk=topk,
        include_niah_noise=False,
    )
    semantic_target = " ".join(semantic_sentences)
    return {
        "schema_version": SCHEMA_CASE,
        "case_id": f"2wiki::{query_id}",
        "dataset": "2wiki",
        "source": "official-train-evidences-supporting-facts-context",
        "group_id": query_id,
        "query_id": query_id,
        "role": role,
        "component_id": component_id,
        "answerable": True,
        "target_kind": "twowiki_evidence_chain",
        "question": query_text,
        "answer": answer,
        "semantic_target": semantic_target,
        "semantic_target_sha256": _sha256_text(semantic_target),
        "official_answer": answer,
        "official_evidences": evidences,
        "official_supporting_facts": supporting_fact_rows,
        "chain_eligible_topk10": bool(role_row.get("chain_eligible_topk10")),
        "variants": variants,
    }


def _materialize_twowiki(
    *,
    queries_path: Path,
    candidate_pool_path: Path,
    gold_path: Path,
    role_assignments_path: Path,
    component_map_path: Path,
    official_rows_path: Path,
) -> tuple[list[dict[str, object]], Counter[str]]:
    queries = _load_map(queries_path, "query_id")
    pools = _load_map(candidate_pool_path, "query_id")
    gold = _load_map(gold_path, "query_id")
    roles = _load_map(role_assignments_path, "query_id")
    components = _load_map(component_map_path, "query_id")
    official_rows = _load_twowiki_rows(official_rows_path)
    counts: Counter[str] = Counter()
    output: list[dict[str, object]] = []
    for query_id in sorted(roles):
        role = str(roles[query_id].get("role", ""))
        if role not in {TRAIN_ROLE, VALIDATION_ROLE}:
            continue
        counts[f"role_{role}"] += 1
        official = official_rows.get(query_id)
        if official is None:
            counts["excluded_missing_official_row"] += 1
            continue
        relevant = {str(item) for item in gold[query_id].get("relevant_document_ids", [])}
        topk = _topk(pools[query_id]["candidates"])
        if not relevant <= {str(item.get("document_id", "")) for item in topk}:
            counts[f"excluded_{role}_support_not_fully_in_top10"] += 1
            continue
        references = gold[query_id].get("reference_answers")
        if not isinstance(references, list) or len(references) != 1:
            counts["excluded_bad_reference"] += 1
            continue
        case = _twowiki_case(
            query_id=query_id,
            role=role,
            component_id=str(components[query_id]["component_id"]),
            role_row=roles[query_id],
            query_text=str(queries[query_id]["text"]),
            answer=str(references[0]),
            topk=topk,
            official_row=official,
        )
        if case is None:
            counts[f"excluded_{role}_bad_official_chain"] += 1
            continue
        output.append(case)
        counts[f"materialized_{role}"] += 1
    return output, counts


def _unsupported_case(row: Mapping[str, Any]) -> dict[str, object] | None:
    if row.get("dataset") != "2wiki" or row.get("role") != TRAIN_ROLE or not row.get("answerable"):
        return None
    variants = row.get("variants")
    if not isinstance(variants, Mapping):
        return None
    topk = variants.get("topk")
    if not isinstance(topk, Mapping):
        return None
    evidence_ids = topk.get("evidence_ids")
    support_ids = topk.get("support_evidence_ids")
    prompt = str(topk.get("prompt", ""))
    if not isinstance(evidence_ids, list) or not isinstance(support_ids, list) or not prompt:
        return None
    support_set = {str(item) for item in support_ids}
    kept_ids = [str(item) for item in evidence_ids if str(item) not in support_set]
    if not kept_ids or len(kept_ids) == len(evidence_ids):
        return None
    # Rebuild a prompt by filtering the serialized evidence lines.  The source answerable
    # prompt never appears at runtime; this is an offline unsupported training target.
    kept_lines: list[str] = []
    evidence_index = 1
    for line in prompt.splitlines():
        if not line.startswith("["):
            kept_lines.append(line)
            continue
        keep = any(f"({evidence_id})" in line for evidence_id in kept_ids)
        if keep:
            kept_lines.append(re.sub(r"^\[\d+\]", f"[{evidence_index}]", line))
            evidence_index += 1
    unsupported_prompt = "\n".join(kept_lines)
    query_id = str(row["query_id"])
    return {
        "schema_version": SCHEMA_CASE,
        "case_id": f"unsupported::2wiki::{query_id}",
        "dataset": "2wiki",
        "source": "support-removed-from-train-split",
        "group_id": query_id,
        "query_id": query_id,
        "role": TRAIN_ROLE,
        "component_id": str(row.get("component_id", "")),
        "answerable": False,
        "target_kind": "unsupported_support_removed",
        "question": str(row.get("question", "")),
        "answer": UNKNOWN_ANSWER,
        "semantic_target": UNKNOWN_ANSWER,
        "semantic_target_sha256": _sha256_text(UNKNOWN_ANSWER),
        "removed_support_evidence_ids": sorted(support_set),
        "variants": {
            "support_removed": {
                "evidence_ids": kept_ids,
                "support_evidence_ids": [],
                "prompt": unsupported_prompt,
                "target": UNKNOWN_ANSWER,
            }
        },
    }


def _example_updates(rows: Sequence[Mapping[str, Any]]) -> int:
    total = 0
    for row in rows:
        variants = row.get("variants")
        if not isinstance(variants, Mapping):
            raise ValueError(f"case lacks variants: {row.get('case_id')}")
        total += len(variants)
    return total


def _counts_by_dataset_role(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    seen: dict[tuple[str, str, str, bool], set[str]] = {}
    for row in rows:
        dataset = str(row.get("dataset", ""))
        role = str(row.get("role", ""))
        answerable = bool(row.get("answerable"))
        key = (dataset, role, "answerable" if answerable else "unsupported", answerable)
        seen.setdefault(key, set()).add(str(row.get("group_id", "")))
    for (dataset, role, kind, _answerable), groups in seen.items():
        counts.setdefault(dataset, {})[f"{role}_{kind}_groups"] = len(groups)
    return counts


def _leakage_report(train_rows: Sequence[Mapping[str, Any]], validation_rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    train_groups = {(str(row.get("dataset")), str(row.get("group_id"))) for row in train_rows}
    validation_groups = {
        (str(row.get("dataset")), str(row.get("group_id"))) for row in validation_rows
    }
    train_components = {
        (str(row.get("dataset")), str(row.get("component_id"))) for row in train_rows
    }
    validation_components = {
        (str(row.get("dataset")), str(row.get("component_id"))) for row in validation_rows
    }
    return {
        "group_overlap": len(train_groups & validation_groups),
        "component_overlap": len(train_components & validation_components),
    }


def materialize(
    *,
    old_niah_train_cases_path: Path,
    niah_queries_path: Path,
    niah_candidate_pool_path: Path,
    niah_gold_path: Path,
    niah_modelval_roles_path: Path,
    niah_modelval_components_path: Path,
    niah_modelval_qa2d_targets_path: Path,
    twowiki_queries_path: Path,
    twowiki_candidate_pool_path: Path,
    twowiki_gold_path: Path,
    twowiki_roles_path: Path,
    twowiki_components_path: Path,
    twowiki_official_rows_path: Path,
    output_dir: Path,
    min_niah_train_groups: int = 400,
    min_niah_modelval_groups: int = 100,
    min_twowiki_train_groups: int = 400,
    min_twowiki_modelval_groups: int = 100,
    unsupported_ratio_min: float = 0.10,
    unsupported_ratio_max: float = 0.15,
) -> dict[str, object]:
    _require_empty(output_dir, "G200 v2 data")
    old_niah_train = [_convert_old_niah_train(row) for row in _jsonl(old_niah_train_cases_path)]
    new_niah_validation, niah_counts = _materialize_new_niah_modelval(
        queries_path=niah_queries_path,
        candidate_pool_path=niah_candidate_pool_path,
        gold_path=niah_gold_path,
        role_assignments_path=niah_modelval_roles_path,
        component_map_path=niah_modelval_components_path,
        qa2d_targets_path=niah_modelval_qa2d_targets_path,
    )
    twowiki_rows, twowiki_counts = _materialize_twowiki(
        queries_path=twowiki_queries_path,
        candidate_pool_path=twowiki_candidate_pool_path,
        gold_path=twowiki_gold_path,
        role_assignments_path=twowiki_roles_path,
        component_map_path=twowiki_components_path,
        official_rows_path=twowiki_official_rows_path,
    )
    twowiki_train = [row for row in twowiki_rows if row["role"] == TRAIN_ROLE]
    twowiki_validation = [row for row in twowiki_rows if row["role"] == VALIDATION_ROLE]
    answerable_train = [*old_niah_train, *twowiki_train]
    answerable_validation = [*new_niah_validation, *twowiki_validation]
    answerable_updates = _example_updates(answerable_train)
    minimum_unsupported = math.ceil(
        (unsupported_ratio_min * answerable_updates) / (1.0 - unsupported_ratio_min)
    )
    desired_unsupported = round((0.12 * answerable_updates) / 0.88)
    maximum_unsupported = math.floor(
        (unsupported_ratio_max * answerable_updates) / (1.0 - unsupported_ratio_max)
    )
    unsupported_candidates = [
        item for item in (_unsupported_case(row) for row in twowiki_train) if item is not None
    ]
    unsupported_target = min(max(desired_unsupported, minimum_unsupported), maximum_unsupported)
    if len(unsupported_candidates) < unsupported_target:
        unsupported_target = len(unsupported_candidates)
    unsupported_rows = unsupported_candidates[:unsupported_target]
    all_train = [*answerable_train, *unsupported_rows]
    leakage = _leakage_report(all_train, answerable_validation)
    counts = _counts_by_dataset_role([*all_train, *answerable_validation])
    unsupported_updates = _example_updates(unsupported_rows)
    unsupported_ratio = unsupported_updates / (answerable_updates + unsupported_updates)
    gates = {
        "niah_train_groups": len({str(row["group_id"]) for row in old_niah_train})
        >= min_niah_train_groups,
        "niah_new_modelval_groups": len(
            {str(row["group_id"]) for row in new_niah_validation}
        )
        >= min_niah_modelval_groups,
        "twowiki_train_groups": len({str(row["group_id"]) for row in twowiki_train})
        >= min_twowiki_train_groups,
        "twowiki_modelval_groups": len({str(row["group_id"]) for row in twowiki_validation})
        >= min_twowiki_modelval_groups,
        "unsupported_update_ratio": unsupported_ratio_min
        <= unsupported_ratio
        <= unsupported_ratio_max,
        "split_group_overlap_zero": leakage["group_overlap"] == 0,
        "split_component_overlap_zero": leakage["component_overlap"] == 0,
    }
    if not all(gates.values()):
        failed = {name: value for name, value in gates.items() if not value}
        raise ValueError(f"G200 v2 materialization gates failed: {failed}")

    train_path = output_dir / "train_cases.jsonl"
    validation_path = output_dir / "validation_cases.jsonl"
    _write_jsonl(train_path, all_train)
    _write_jsonl(validation_path, answerable_validation)
    ordered_ids = {
        "train_case_ids": [str(row["case_id"]) for row in all_train],
        "validation_case_ids": [str(row["case_id"]) for row in answerable_validation],
        "answerable_train_group_ids": [
            f"{row['dataset']}::{row['group_id']}" for row in answerable_train
        ],
        "answerable_validation_group_ids": [
            f"{row['dataset']}::{row['group_id']}" for row in answerable_validation
        ],
        "unsupported_train_case_ids": [str(row["case_id"]) for row in unsupported_rows],
    }
    ordered_path = output_dir / "ordered_ids.json"
    _write_json(ordered_path, ordered_ids)
    manifest: dict[str, object] = {
        "schema_version": SCHEMA_MANIFEST,
        "status": "PRE_AUDIT",
        "stage": "G200",
        "sealed_or_heldout_read": False,
        "dev_read": False,
        "gold_use": "offline target construction only; runtime prompts contain evidence text and question only",
        "training_started": False,
        "true_audit_pending": True,
        "qa2d": {
            "model": QA2D_MODEL_ID,
            "revision": QA2D_REVISION,
            "runtime_component": False,
        },
        "counts": counts,
        "niah_modelval_counts": dict(niah_counts),
        "twowiki_counts": dict(twowiki_counts),
        "train_cases": len(all_train),
        "validation_cases": len(answerable_validation),
        "answerable_train_updates": answerable_updates,
        "unsupported_updates": unsupported_updates,
        "unsupported_update_ratio": unsupported_ratio,
        "split_leakage": leakage,
        "gates": gates,
        "train_cases_sha256": _sha256(train_path),
        "validation_cases_sha256": _sha256(validation_path),
        "ordered_ids_sha256": _sha256(ordered_path),
        "input_sha256": {
            "old_niah_train_cases": _sha256(old_niah_train_cases_path),
            "niah_queries": _sha256(niah_queries_path),
            "niah_candidate_pool": _sha256(niah_candidate_pool_path),
            "niah_gold": _sha256(niah_gold_path),
            "niah_modelval_roles": _sha256(niah_modelval_roles_path),
            "niah_modelval_components": _sha256(niah_modelval_components_path),
            "niah_modelval_qa2d_targets": _sha256(niah_modelval_qa2d_targets_path),
            "twowiki_queries": _sha256(twowiki_queries_path),
            "twowiki_candidate_pool": _sha256(twowiki_candidate_pool_path),
            "twowiki_gold": _sha256(twowiki_gold_path),
            "twowiki_roles": _sha256(twowiki_roles_path),
            "twowiki_components": _sha256(twowiki_components_path),
            "twowiki_official_rows": _sha256(twowiki_official_rows_path),
        },
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)

    prepare = subcommands.add_parser("prepare-niah-modelval")
    prepare.add_argument("--queries", required=True, type=Path)
    prepare.add_argument("--candidate-pool", required=True, type=Path)
    prepare.add_argument("--gold", required=True, type=Path)
    prepare.add_argument("--source-parent", required=True, type=Path)
    prepare.add_argument("--old-role-assignments", required=True, type=Path)
    prepare.add_argument("--old-component-map", required=True, type=Path)
    prepare.add_argument("--output-dir", required=True, type=Path)

    data = subcommands.add_parser("materialize")
    data.add_argument("--old-niah-train-cases", required=True, type=Path)
    data.add_argument("--niah-queries", required=True, type=Path)
    data.add_argument("--niah-candidate-pool", required=True, type=Path)
    data.add_argument("--niah-gold", required=True, type=Path)
    data.add_argument("--niah-modelval-roles", required=True, type=Path)
    data.add_argument("--niah-modelval-components", required=True, type=Path)
    data.add_argument("--niah-modelval-qa2d-targets", required=True, type=Path)
    data.add_argument("--twowiki-queries", required=True, type=Path)
    data.add_argument("--twowiki-candidate-pool", required=True, type=Path)
    data.add_argument("--twowiki-gold", required=True, type=Path)
    data.add_argument("--twowiki-roles", required=True, type=Path)
    data.add_argument("--twowiki-components", required=True, type=Path)
    data.add_argument("--twowiki-official-rows", required=True, type=Path)
    data.add_argument("--output-dir", required=True, type=Path)
    data.add_argument("--min-niah-train-groups", type=int, default=400)
    data.add_argument("--min-niah-modelval-groups", type=int, default=100)
    data.add_argument("--min-twowiki-train-groups", type=int, default=400)
    data.add_argument("--min-twowiki-modelval-groups", type=int, default=100)
    data.add_argument("--unsupported-ratio-min", type=float, default=0.10)
    data.add_argument("--unsupported-ratio-max", type=float, default=0.15)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare-niah-modelval":
        report = prepare_niah_modelval(
            queries_path=args.queries,
            candidate_pool_path=args.candidate_pool,
            gold_path=args.gold,
            source_parent_path=args.source_parent,
            old_role_assignments_path=args.old_role_assignments,
            old_component_map_path=args.old_component_map,
            output_dir=args.output_dir.resolve(),
        )
    else:
        report = materialize(
            old_niah_train_cases_path=args.old_niah_train_cases,
            niah_queries_path=args.niah_queries,
            niah_candidate_pool_path=args.niah_candidate_pool,
            niah_gold_path=args.niah_gold,
            niah_modelval_roles_path=args.niah_modelval_roles,
            niah_modelval_components_path=args.niah_modelval_components,
            niah_modelval_qa2d_targets_path=args.niah_modelval_qa2d_targets,
            twowiki_queries_path=args.twowiki_queries,
            twowiki_candidate_pool_path=args.twowiki_candidate_pool,
            twowiki_gold_path=args.twowiki_gold,
            twowiki_roles_path=args.twowiki_roles,
            twowiki_components_path=args.twowiki_components,
            twowiki_official_rows_path=args.twowiki_official_rows,
            output_dir=args.output_dir.resolve(),
            min_niah_train_groups=args.min_niah_train_groups,
            min_niah_modelval_groups=args.min_niah_modelval_groups,
            min_twowiki_train_groups=args.min_twowiki_train_groups,
            min_twowiki_modelval_groups=args.min_twowiki_modelval_groups,
            unsupported_ratio_min=args.unsupported_ratio_min,
            unsupported_ratio_max=args.unsupported_ratio_max,
        )
    print(json.dumps(report, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
