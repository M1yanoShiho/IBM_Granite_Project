"""Build a parent-page-safe NQ/DPR split for the ML evidence selector."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable, Mapping, Sequence

from src.retrieval.selector_data import (
    NIAHMetadataRecord,
    assign_niah_grouped_splits,
    sha256_file,
    validate_unique_ids,
)


@dataclass(frozen=True)
class NQGoldPassage:
    doc_id: str
    title: str
    text: str
    relevance: int


@dataclass(frozen=True)
class NQNeedleCandidate:
    query_id: str
    needle_doc_id: str
    parent_page_id: str
    synthetic_family_id: str


def normalize_parent_page(title: str) -> str:
    """Normalize a Wikipedia title for group-level leakage checks."""

    if not isinstance(title, str) or not title.strip():
        raise ValueError("Wikipedia parent title must be a non-empty string")
    return re.sub(r"\s+", " ", title).strip().casefold()


def _contains_alias(text: str, aliases: Sequence[str]) -> bool:
    folded = text.casefold()
    return any(alias.strip() and alias.strip().casefold() in folded for alias in aliases)


def choose_designated_needle(
    passages: Sequence[NQGoldPassage], aliases: Sequence[str]
) -> NQGoldPassage:
    """Choose one deterministic gold passage, preferring answer-bearing passages."""

    if not passages:
        raise ValueError("A query needs at least one relevant DPR passage")
    answer_bearing = [passage for passage in passages if _contains_alias(passage.text, aliases)]
    pool = answer_bearing or list(passages)
    return min(pool, key=lambda passage: (-passage.relevance, passage.doc_id))


def load_legacy_query_ids(path: str | Path) -> set[str]:
    """Load the old NQ-300 query IDs from a per-query result CSV."""

    source = Path(path)
    try:
        with source.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if "qid" not in (reader.fieldnames or []):
                raise ValueError(f"{source} must contain a qid column")
            values = [str(row["qid"]).strip() for row in reader]
    except OSError as exc:
        raise ValueError(f"Unable to read legacy query CSV {source}: {exc}") from exc
    if any(not value for value in values):
        raise ValueError(f"{source} contains an empty legacy query ID")
    if len(set(values)) != len(values):
        raise ValueError(f"duplicate legacy query ID in {source}")
    return set(values)


def _stable_order(candidates: Iterable[NQNeedleCandidate], seed: int) -> list[NQNeedleCandidate]:
    return sorted(
        candidates,
        key=lambda item: (
            hashlib.sha256(f"{seed}\x1f{item.query_id}".encode("utf-8")).hexdigest(),
            item.query_id,
        ),
    )


def build_niah_split_payload(
    candidates: Sequence[NQNeedleCandidate],
    *,
    legacy_parent_pages: set[str],
    targets: Mapping[str, int],
    pilot_train_target: int,
    seed: int,
    dataset_fingerprint: str,
    legacy_query_fingerprint: str,
) -> dict[str, object]:
    """Select and assign exact train/dev/test counts without parent/family leakage."""

    validate_unique_ids((candidate.query_id for candidate in candidates), id_kind="query")
    if isinstance(pilot_train_target, bool) or not isinstance(pilot_train_target, int):
        raise ValueError("pilot_train_target must be an integer")
    required = sum(targets.values())
    normalized_legacy = {normalize_parent_page(page) for page in legacy_parent_pages}
    excluded = [
        candidate
        for candidate in candidates
        if normalize_parent_page(candidate.parent_page_id) in normalized_legacy
    ]
    eligible = [candidate for candidate in candidates if candidate not in excluded]
    if len(eligible) < required:
        raise ValueError(
            f"Only {len(eligible)} eligible NQ queries remain for {required} requested rows"
        )
    if pilot_train_target < 0 or pilot_train_target > targets.get("train", -1):
        raise ValueError("pilot_train_target must be between zero and the train target")

    selected: list[NQNeedleCandidate] | None = None
    assignments: dict[str, tuple[str, ...]] | None = None
    last_error: ValueError | None = None
    for attempt in range(256):
        ordered = _stable_order(eligible, seed + attempt)
        trial = ordered[:required]
        metadata = [
            NIAHMetadataRecord(
                query_id=item.query_id,
                parent_page_id=normalize_parent_page(item.parent_page_id),
                synthetic_family_id=item.synthetic_family_id,
            )
            for item in trial
        ]
        try:
            trial_assignments = assign_niah_grouped_splits(
                metadata, targets=targets, seed=seed
            )
        except ValueError as exc:
            last_error = exc
            continue
        selected = trial
        assignments = trial_assignments
        break
    if selected is None or assignments is None:
        raise ValueError(
            "Unable to select an exact leakage-safe NIAH split after 256 deterministic attempts"
        ) from last_error

    by_query = {candidate.query_id: candidate for candidate in selected}
    rows_by_split: dict[str, list[dict[str, str]]] = {}
    for split in ("train", "dev", "test"):
        rows_by_split[split] = [
            {
                "query_id": query_id,
                "needle_doc_id": by_query[query_id].needle_doc_id,
                "parent_page_id": normalize_parent_page(by_query[query_id].parent_page_id),
                "synthetic_family_id": by_query[query_id].synthetic_family_id,
            }
            for query_id in sorted(assignments[split])
        ]
    pilot_ids = [
        item.query_id
        for item in _stable_order(
            (by_query[row["query_id"]] for row in rows_by_split["train"]), seed
        )[:pilot_train_target]
    ]
    return {
        "status": "ready",
        "strategy": "dpr_nq_parent_page_and_synthetic_family_grouped",
        "seed": seed,
        "dataset_fingerprint": dataset_fingerprint,
        "legacy_query_fingerprint": legacy_query_fingerprint,
        "candidate_query_count": len(candidates),
        "eligible_query_count": len(eligible),
        "excluded_legacy_parent_query_count": len(excluded),
        "counts": {split: len(rows_by_split[split]) for split in ("train", "dev", "test")},
        "splits": rows_by_split,
        "pilot_train_query_ids": pilot_ids,
        "selection_attempt_limit": 256,
    }


def collect_nq_candidates(dataset, legacy_query_ids: set[str]) -> tuple[
    list[NQNeedleCandidate], set[str], str, dict[str, int]
]:
    """Collect designated needles and parent pages from an ir_datasets DPR split."""

    queries: dict[str, tuple[str, ...]] = {}
    digest = hashlib.sha256()
    for query in dataset.queries_iter():
        qid = str(query.query_id)
        aliases = tuple(str(answer) for answer in getattr(query, "answers", ()) if str(answer))
        if aliases:
            queries[qid] = aliases
        digest.update(json.dumps([qid, query.text, aliases], ensure_ascii=False).encode("utf-8"))

    missing_legacy_ids = legacy_query_ids.difference(queries)
    if missing_legacy_ids:
        preview = ", ".join(sorted(missing_legacy_ids)[:5])
        raise ValueError(f"Legacy query IDs are absent from the DPR split: {preview}")

    relevant: dict[str, dict[str, int]] = {}
    for qrel in dataset.qrels_iter():
        qid = str(qrel.query_id)
        relevance = int(qrel.relevance)
        if qid in queries and relevance > 0:
            relevant.setdefault(qid, {})[str(qrel.doc_id)] = relevance
            digest.update(f"\x1e{qid}\x1f{qrel.doc_id}\x1f{relevance}".encode("utf-8"))
    required_doc_ids = {doc_id for rows in relevant.values() for doc_id in rows}
    passages: dict[str, NQGoldPassage] = {}
    for document in dataset.docs_iter():
        doc_id = str(document.doc_id)
        if doc_id not in required_doc_ids:
            continue
        passage = NQGoldPassage(
            doc_id=doc_id,
            title=str(document.title),
            text=str(document.text),
            relevance=0,
        )
        passages[doc_id] = passage
        digest.update(
            json.dumps([doc_id, passage.title, passage.text], ensure_ascii=False).encode("utf-8")
        )
        if len(passages) == len(required_doc_ids):
            break

    candidates: list[NQNeedleCandidate] = []
    legacy_parent_pages: set[str] = set()
    missing_gold_queries = 0
    for qid in sorted(queries):
        gold = [
            NQGoldPassage(
                doc_id=doc_id,
                title=passages[doc_id].title,
                text=passages[doc_id].text,
                relevance=relevance,
            )
            for doc_id, relevance in relevant.get(qid, {}).items()
            if doc_id in passages
        ]
        if not gold:
            missing_gold_queries += 1
            continue
        if qid in legacy_query_ids:
            legacy_parent_pages.update(normalize_parent_page(passage.title) for passage in gold)
            continue
        needle = choose_designated_needle(gold, queries[qid])
        parent = normalize_parent_page(needle.title)
        candidates.append(
            NQNeedleCandidate(
                query_id=qid,
                needle_doc_id=needle.doc_id,
                parent_page_id=parent,
                synthetic_family_id=f"nq:{qid}:{needle.doc_id}",
            )
        )
    audit = {
        "answer_bearing_query_count": len(queries),
        "relevant_document_count": len(required_doc_ids),
        "relevant_document_found_count": len(passages),
        "missing_gold_query_count": missing_gold_queries,
        "legacy_query_count": len(legacy_query_ids),
        "legacy_parent_page_count": len(legacy_parent_pages),
    }
    return candidates, legacy_parent_pages, f"sha256:{digest.hexdigest()}", audit


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default="dpr-w100/natural-questions/dev")
    parser.add_argument("--legacy-qids-csv", type=Path, required=True)
    parser.add_argument("--base-split-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train", type=int, default=2000)
    parser.add_argument("--dev", type=int, default=300)
    parser.add_argument("--test", type=int, default=300)
    parser.add_argument("--pilot-train", type=int, default=500, dest="pilot_train")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    import ir_datasets

    legacy_ids = load_legacy_query_ids(args.legacy_qids_csv)
    dataset = ir_datasets.load(args.dataset_id)
    candidates, legacy_pages, fingerprint, audit = collect_nq_candidates(dataset, legacy_ids)
    payload = build_niah_split_payload(
        candidates,
        legacy_parent_pages=legacy_pages,
        targets={"train": args.train, "dev": args.dev, "test": args.test},
        pilot_train_target=args.pilot_train,
        seed=args.seed,
        dataset_fingerprint=fingerprint,
        legacy_query_fingerprint=f"sha256:{sha256_file(args.legacy_qids_csv)}",
    )
    payload["dataset_id"] = args.dataset_id
    payload["source_audit"] = audit
    try:
        base = json.loads(args.base_split_manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to load base split manifest: {exc}") from exc
    base["datasets"]["niah"] = payload
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(base, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
