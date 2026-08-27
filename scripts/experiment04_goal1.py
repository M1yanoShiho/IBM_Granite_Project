"""Experiment 04 Goal 1: sealed data materialization and scorer validation.

This command is intentionally incapable of running a Retriever, Selector,
Generator, TRUE, or MiniCheck model.  Its held-out log surface is restricted to
counts, schemas, ordered-ID hashes, file hashes, and boolean audit results.
Five-metric behavior is validated only on the synthetic fixture in this file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from heldout_data import (  # noqa: E402
    HOTPOT_URL,
    MUSIQUE_URL,
    RGB_URL,
    _download,
)

from evidence_rag.evaluation.sealed_runtime import (  # noqa: E402
    RUNTIME_SCHEMA_VERSION,
    file_sha256,
    ordered_id_sha256,
    read_runtime_bundle,
    runtime_audit,
)
from evidence_rag.evaluation.system_scorer import (  # noqa: E402
    METRIC_KEYS,
    SCORER_SCHEMA_VERSION,
    SIDECAR_KEYS,
    score_bundle,
    validate_sidecar_record,
)

EXPERIMENT_DIR = (
    REPO_ROOT
    / "docs/full-flow/experiments/04_frozen_three_module_system_evaluation_2026-08-21"
)
DEFAULT_SAMPLE_MANIFEST = REPO_ROOT / "configs/heldout-sample.json"
DEFAULT_SEALED_ROOT = REPO_ROOT / "artifacts/experiment04_goal1/sealed"
DEFAULT_SOURCE_CACHE = REPO_ROOT / "artifacts/experiment04_goal1/source_cache"
DEFAULT_DATA_MANIFEST = EXPERIMENT_DIR / "artifacts/goal1_data_manifest.json"
DEFAULT_SCORER_VALIDATION = EXPERIMENT_DIR / "artifacts/goal1_scorer_validation.json"

DATASET_ORDER = ("hotpotqa", "musique-answerable", "rgb-noise")
EXPECTED_COUNTS = {"hotpotqa": 400, "musique-answerable": 400, "rgb-noise": 300}


class Goal1Error(RuntimeError):
    """A hard Goal 1 contract failure."""


def _clean(text: object) -> str:
    return " ".join(str(text).split())


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _component_id(dataset: str, raw_id: str) -> str:
    digest = hashlib.sha256(f"{dataset}\0{raw_id}".encode()).hexdigest()
    return f"cmp-{digest}"


def _support_unit(unit: Mapping[str, Any]) -> dict[str, str]:
    text = str(unit["text"])
    return {
        "unit_id": str(unit["unit_id"]),
        "source_id": str(unit["source_id"]),
        "text": text,
        "text_sha256": _text_sha256(text),
    }


def _write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        for record in records:
            handle.write(_json_bytes(record))
            handle.write(b"\n")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_sample_manifest(path: Path) -> tuple[dict[str, list[str]], dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if set(raw) != {"datasets", "seed"} or raw["seed"] != 13:
        raise Goal1Error("frozen sample manifest schema/seed differs from pre-registration")
    datasets = raw["datasets"]
    if not isinstance(datasets, dict):
        raise Goal1Error("frozen sample manifest has no datasets object")

    integrity: dict[str, Any] = {}
    for name in ("hotpotqa", "musique-full", "rgb"):
        entry = datasets.get(name)
        if not isinstance(entry, dict) or not isinstance(entry.get("query_ids"), list):
            raise Goal1Error(f"frozen sample manifest has no ordered IDs for {name}")
        ids = [str(value) for value in entry["query_ids"]]
        observed_hash = ordered_id_sha256(ids)
        expected_hash = str(entry.get("sha256", ""))
        if observed_hash != expected_hash:
            raise Goal1Error(f"frozen sample manifest hash mismatch for {name}")
        integrity[name] = {
            "record_count": len(ids),
            "declared_ordered_ids_sha256": expected_hash,
            "recomputed_ordered_ids_sha256": observed_hash,
            "match": True,
        }

    hotpot_ids = [str(value) for value in datasets["hotpotqa"]["query_ids"]]
    musique_pairs = [str(value) for value in datasets["musique-full"]["query_ids"]]
    musique_answerable = [query_id for query_id in musique_pairs if query_id.endswith("#ans")]
    rgb_ids = [str(value) for value in datasets["rgb"]["query_ids"]]
    expected = {
        "hotpotqa": hotpot_ids,
        "musique-answerable": musique_answerable,
        "rgb-noise": rgb_ids,
    }
    for dataset, query_ids in expected.items():
        if len(query_ids) != EXPECTED_COUNTS[dataset] or len(query_ids) != len(set(query_ids)):
            raise Goal1Error(f"{dataset} does not contain the frozen unique count")
    if len(musique_pairs) != 800:
        raise Goal1Error("MuSiQue paired source manifest is not the frozen 800-record draw")
    paired_bases: dict[str, set[str]] = {}
    for query_id in musique_pairs:
        base, suffix = query_id.rsplit("#", 1)
        paired_bases.setdefault(base, set()).add(suffix)
    if len(paired_bases) != 400 or any(suffixes != {"ans", "unans"} for suffixes in paired_bases.values()):
        raise Goal1Error("MuSiQue source manifest no longer contains 400 complete pairs")
    return expected, integrity


def _candidate(source_id: str, title: object, unit_texts: Iterable[tuple[int, object]]) -> dict[str, Any] | None:
    units = [
        {"unit_id": f"{source_id}:u{unit_index:03d}", "text": cleaned}
        for unit_index, raw_text in unit_texts
        if (cleaned := _clean(raw_text))
    ]
    if not units:
        return None
    return {
        "source_id": source_id,
        "title": _clean(title),
        "text": _clean(" ".join(str(unit["text"]) for unit in units)),
        "units": units,
    }


def _runtime_record(dataset: str, query_id: str, question: object, candidates: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": RUNTIME_SCHEMA_VERSION,
        "dataset": dataset,
        "query_id": query_id,
        "question": _clean(question),
        "candidates": candidates,
    }


def _sidecar_record(
    dataset: str,
    query_id: str,
    raw_component_id: str,
    aliases: Iterable[object],
    support_units: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    clean_aliases = list(dict.fromkeys(_clean(alias) for alias in aliases if _clean(alias)))
    clean_supports: list[dict[str, str]] = []
    seen: set[str] = set()
    for unit in support_units:
        unit_id = str(unit["unit_id"])
        if unit_id not in seen:
            clean_supports.append(_support_unit(unit))
            seen.add(unit_id)
    if not clean_aliases or not clean_supports:
        raise Goal1Error(f"{dataset} record lacks answer aliases or support units")
    return {
        "schema_version": SCORER_SCHEMA_VERSION,
        "dataset": dataset,
        "query_id": query_id,
        "gold_answer_aliases": clean_aliases,
        "support_units": clean_supports,
        "component_id": _component_id(dataset, raw_component_id),
    }


def _hotpot_records(path: Path, expected_ids: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - dependency error is environment-specific
        raise Goal1Error("HotpotQA materialization requires the data-prep optional dependency") from exc

    frame = pd.read_parquet(path)
    expected_set = set(expected_ids)
    by_id: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for _, row in frame.iterrows():
        query_id = str(row["id"])
        if query_id not in expected_set:
            continue
        context = row["context"]
        candidates: list[dict[str, Any]] = []
        unit_by_fact: dict[tuple[str, int], dict[str, Any]] = {}
        source_by_title: dict[str, str] = {}
        for paragraph_index, (raw_title, raw_sentences) in enumerate(
            zip(list(context["title"]), list(context["sentences"]), strict=True)
        ):
            source_id = f"p{paragraph_index:03d}"
            raw_title_key = str(raw_title)
            if raw_title_key in source_by_title:
                raise Goal1Error("HotpotQA candidate titles are not unique within a query")
            candidate = _candidate(source_id, raw_title, enumerate(list(raw_sentences)))
            if candidate is None:
                continue
            source_by_title[raw_title_key] = source_id
            candidates.append(candidate)
            for unit in candidate["units"]:
                raw_index = int(str(unit["unit_id"]).rsplit("u", 1)[1])
                unit_by_fact[(raw_title_key, raw_index)] = {
                    **unit,
                    "source_id": source_id,
                }
        supporting = row["supporting_facts"]
        supports: list[dict[str, Any]] = []
        for raw_title, raw_sentence_id in zip(
            list(supporting["title"]), list(supporting["sent_id"]), strict=True
        ):
            key = (str(raw_title), int(raw_sentence_id))
            if key not in unit_by_fact:
                raise Goal1Error("HotpotQA support fact cannot be aligned to its local candidate")
            supports.append(unit_by_fact[key])
        if query_id in by_id:
            raise Goal1Error("HotpotQA source contains a duplicate frozen query ID")
        by_id[query_id] = (
            _runtime_record("hotpotqa", query_id, row["question"], candidates),
            _sidecar_record("hotpotqa", query_id, query_id, [row["answer"]], supports),
        )
    return _ordered_records("hotpotqa", expected_ids, by_id)


def _musique_records(path: Path, expected_ids: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    expected_set = set(expected_ids)
    by_id: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            answerable = bool(row.get("answerable", True))
            query_id = f"{row['id']}#{'ans' if answerable else 'unans'}"
            if query_id not in expected_set:
                continue
            candidates: list[dict[str, Any]] = []
            supports: list[dict[str, Any]] = []
            for paragraph_index, paragraph in enumerate(row.get("paragraphs", [])):
                source_id = f"p{paragraph_index:03d}"
                candidate = _candidate(
                    source_id,
                    paragraph.get("title", ""),
                    [(0, paragraph.get("paragraph_text", ""))],
                )
                if candidate is None:
                    continue
                candidates.append(candidate)
                if bool(paragraph.get("is_supporting", False)):
                    supports.append({**candidate["units"][0], "source_id": source_id})
            aliases = [row.get("answer", ""), *(row.get("answer_aliases") or [])]
            raw_id = str(row["id"])
            if query_id in by_id:
                raise Goal1Error("MuSiQue source contains a duplicate frozen answerable query ID")
            by_id[query_id] = (
                _runtime_record("musique-answerable", query_id, row["question"], candidates),
                _sidecar_record("musique-answerable", query_id, raw_id, aliases, supports),
            )
    return _ordered_records("musique-answerable", expected_ids, by_id)


def _rgb_text(value: object) -> str:
    if isinstance(value, list):
        return _clean(" ".join(str(part) for part in value))
    return _clean(value)


def _rgb_aliases(raw: object) -> list[str]:
    values = [raw] if isinstance(raw, str) else list(raw) if isinstance(raw, list) else []
    aliases: list[str] = []
    for value in values:
        if isinstance(value, list):
            aliases.extend(_clean(alias) for alias in value if _clean(alias))
        elif _clean(value):
            aliases.append(_clean(value))
    return list(dict.fromkeys(aliases))


def _rgb_records(path: Path, expected_ids: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    text = path.read_text(encoding="utf-8")
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        raw = [json.loads(line) for line in text.splitlines() if line.strip()]
    expected_set = set(expected_ids)
    by_id: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for row_index, row in enumerate(raw):
        query_id = str(row.get("id", row_index))
        if query_id not in expected_set:
            continue
        labelled_documents = [
            *((document, True) for document in (row.get("positive") or [])),
            *((document, False) for document in (row.get("negative") or [])),
        ]
        candidates: list[dict[str, Any]] = []
        supports: list[dict[str, Any]] = []
        for document_index, (document, is_supporting) in enumerate(labelled_documents):
            source_id = f"p{document_index:03d}"
            candidate = _candidate(source_id, "", [(0, _rgb_text(document))])
            if candidate is None:
                continue
            candidates.append(candidate)
            if is_supporting:
                supports.append({**candidate["units"][0], "source_id": source_id})
        if query_id in by_id:
            raise Goal1Error("RGB source contains a duplicate frozen query ID")
        by_id[query_id] = (
            _runtime_record("rgb-noise", query_id, row.get("query", ""), candidates),
            _sidecar_record(
                "rgb-noise",
                query_id,
                query_id,
                _rgb_aliases(row.get("answer", [])),
                supports,
            ),
        )
    return _ordered_records("rgb-noise", expected_ids, by_id)


def _ordered_records(
    dataset: str,
    expected_ids: list[str],
    by_id: Mapping[str, tuple[dict[str, Any], dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    missing = [query_id for query_id in expected_ids if query_id not in by_id]
    if missing:
        raise Goal1Error(f"{dataset} source is missing {len(missing)} frozen query IDs")
    return (
        [by_id[query_id][0] for query_id in expected_ids],
        [by_id[query_id][1] for query_id in expected_ids],
    )


def _sidecar_audit(
    sidecars: list[dict[str, Any]],
    runtimes: list[dict[str, Any]],
    *,
    dataset: str,
    expected_ids: list[str],
) -> dict[str, Any]:
    if [str(row["query_id"]) for row in sidecars] != expected_ids:
        raise Goal1Error("sidecar ordered IDs differ from the frozen manifest")
    runtime_by_id = {str(row["query_id"]): row for row in runtimes}
    support_counts: list[int] = []
    for sidecar in sidecars:
        validate_sidecar_record(sidecar, dataset=dataset)
        runtime = runtime_by_id[str(sidecar["query_id"])]
        runtime_units = {
            str(unit["unit_id"]): (str(candidate["source_id"]), str(unit["text"]))
            for candidate in runtime["candidates"]
            for unit in candidate["units"]
        }
        for support in sidecar["support_units"]:
            observed = runtime_units.get(str(support["unit_id"]))
            expected = (str(support["source_id"]), str(support["text"]))
            if observed != expected or _text_sha256(expected[1]) != support["text_sha256"]:
                raise Goal1Error("sidecar support unit is not aligned to the same query's runtime pool")
        support_counts.append(len(sidecar["support_units"]))
    return {
        "count": len(sidecars),
        "ordered_ids_sha256": ordered_id_sha256(expected_ids),
        "schema_version": SCORER_SCHEMA_VERSION,
        "top_level_fields": sorted(SIDECAR_KEYS),
        "support_unit_count_min": min(support_counts),
        "support_unit_count_max": max(support_counts),
        "contains_gold_answer_aliases": True,
        "contains_minimal_support_units": True,
        "contains_component_id": True,
        "support_units_resolve_within_same_query_only": True,
    }


def _download_sources(source_cache: Path) -> dict[str, Path]:
    source_cache.mkdir(parents=True, exist_ok=True)
    return {
        "hotpotqa": _download(HOTPOT_URL, source_cache / "hotpot_dev_distractor.parquet"),
        "musique-answerable": _download(
            MUSIQUE_URL, source_cache / "musique_full_v1.0_dev.jsonl"
        ),
        "rgb-noise": _download(RGB_URL, source_cache / "rgb_en.json"),
    }


def materialize(
    *,
    sample_manifest_path: Path,
    source_cache: Path,
    sealed_root: Path,
    output_manifest_path: Path,
) -> dict[str, Any]:
    expected_ids, sample_integrity = _load_sample_manifest(sample_manifest_path)
    source_paths = _download_sources(source_cache)
    builders = {
        "hotpotqa": _hotpot_records,
        "musique-answerable": _musique_records,
        "rgb-noise": _rgb_records,
    }
    dataset_audits: dict[str, Any] = {}
    for dataset in DATASET_ORDER:
        runtime_records, sidecar_records = builders[dataset](
            source_paths[dataset], expected_ids[dataset]
        )
        runtime_path = sealed_root / "runtime" / f"{dataset}.jsonl"
        sidecar_path = sealed_root / "scorer_only" / f"{dataset}.jsonl"
        _write_jsonl(runtime_path, runtime_records)
        _write_jsonl(sidecar_path, sidecar_records)
        loaded_runtime = list(read_runtime_bundle(runtime_path, dataset=dataset))
        runtime_result = runtime_audit(
            loaded_runtime,
            dataset=dataset,
            expected_query_ids=expected_ids[dataset],
        )
        sidecar_result = _sidecar_audit(
            sidecar_records,
            runtime_records,
            dataset=dataset,
            expected_ids=expected_ids[dataset],
        )
        expected_hash = ordered_id_sha256(expected_ids[dataset])
        runtime_result["file_sha256"] = file_sha256(runtime_path)
        sidecar_result["file_sha256"] = file_sha256(sidecar_path)
        dataset_audits[dataset] = {
            "frozen_ids": {
                "expected_count": EXPECTED_COUNTS[dataset],
                "observed_count": len(runtime_records),
                "expected_ordered_ids_sha256": expected_hash,
                "observed_ordered_ids_sha256": runtime_result["ordered_ids_sha256"],
                "count_match": len(runtime_records) == EXPECTED_COUNTS[dataset],
                "ordered_ids_match": runtime_result["ordered_ids_sha256"] == expected_hash,
                "pre_registration_relation": (
                    "direct"
                    if dataset != "musique-answerable"
                    else "ordered #ans projection of the frozen 400-ID/800-record paired manifest"
                ),
            },
            "source_file": {
                "sha256": file_sha256(source_paths[dataset]),
                "bytes": source_paths[dataset].stat().st_size,
            },
            "runtime": runtime_result,
            "scorer_only": sidecar_result,
            "cross_checks": {
                "runtime_sidecar_ordered_ids_equal": (
                    runtime_result["ordered_ids_sha256"] == sidecar_result["ordered_ids_sha256"]
                ),
                "alignment_key": "query_id",
                "physical_parent_directories_distinct": (
                    runtime_path.resolve().parent != sidecar_path.resolve().parent
                ),
                "system_reader_accepts_runtime_path_only": True,
            },
        }
        print(
            f"{dataset}: count={len(runtime_records)} "
            f"ordered_ids_match=true runtime_gold_fields=0 pool_isolated=true",
            flush=True,
        )

    manifest = {
        "manifest_version": "experiment04.goal1.data_manifest.v1",
        "goal": 1,
        "sample_seed": 13,
        "frozen_sample_manifest": {
            "path": str(sample_manifest_path.relative_to(REPO_ROOT)),
            "file_sha256": file_sha256(sample_manifest_path),
            "integrity": sample_integrity,
        },
        "sealed_bundle_root": str(sealed_root.relative_to(REPO_ROOT)),
        "datasets": dataset_audits,
        "isolation": {
            "runtime_root": str((sealed_root / "runtime").relative_to(REPO_ROOT)),
            "scorer_only_root": str((sealed_root / "scorer_only").relative_to(REPO_ROOT)),
            "runtime_and_sidecar_physically_separate": True,
            "runtime_contains_gold_fields": False,
            "candidate_pools_are_nested_per_query": True,
            "only_alignment_key": "query_id",
        },
        "scope_attestation": {
            "heldout_content_emitted_to_log": False,
            "retriever_run_on_heldout": False,
            "selector_run_on_heldout": False,
            "generator_run_on_heldout": False,
            "minicheck_run_on_heldout": False,
            "heldout_scored": False,
            "baseline_connected": False,
            "goal2_started": False,
        },
        "overall_pass": all(
            audit["frozen_ids"]["count_match"]
            and audit["frozen_ids"]["ordered_ids_match"]
            and audit["runtime"]["forbidden_gold_field_count"] == 0
            and audit["runtime"]["candidate_pool_isolated"]
            and audit["cross_checks"]["runtime_sidecar_ordered_ids_equal"]
            and audit["cross_checks"]["physical_parent_directories_distinct"]
            for audit in dataset_audits.values()
        ),
    }
    _write_json(output_manifest_path, manifest)
    print(f"data_manifest_pass={str(manifest['overall_pass']).lower()}", flush=True)
    return manifest


def _fixture_gold(
    query_id: str,
    dataset: str,
    aliases: list[str],
    support_count: int = 1,
) -> dict[str, Any]:
    supports = []
    for index in range(support_count):
        text = f"Revealed synthetic support unit {query_id}-{index}."
        supports.append(
            {
                "unit_id": f"p{index:03d}:u000",
                "source_id": f"p{index:03d}",
                "text": text,
                "text_sha256": _text_sha256(text),
            }
        )
    return {
        "schema_version": SCORER_SCHEMA_VERSION,
        "dataset": dataset,
        "query_id": query_id,
        "gold_answer_aliases": aliases,
        "support_units": supports,
        "component_id": _component_id(dataset, query_id),
    }


def validate_scorer(output_path: Path) -> dict[str, Any]:
    sidecars = [
        _fixture_gold("q-perfect", "hotpotqa", ["Nile"], 2),
        _fixture_gold("q-partial", "musique-answerable", ["red green"], 2),
        _fixture_gold("q-rgb", "rgb-noise", ["yes"]),
        _fixture_gold("q-empty", "hotpotqa", ["answer"]),
        _fixture_gold("q-invalid", "hotpotqa", ["answer"]),
        _fixture_gold("q-generation", "hotpotqa", ["answer"], 2),
        _fixture_gold("q-scoring", "hotpotqa", ["answer"]),
        _fixture_gold("q-missing", "hotpotqa", ["answer"]),
    ]
    outcomes = [
        {
            "query_id": "q-perfect",
            "retrieved_unit_ids": ["p000:u000", "p001:u000"],
            "selected_unit_ids": ["p000:u000", "p001:u000"],
            "selected_source_ids": ["p000", "p001"],
            "answer": "The Nile [1]",
            "citation_indices": [1],
            "minicheck": {"precision": 1.0, "recall": 1.0},
        },
        {
            "query_id": "q-partial",
            "retrieved_unit_ids": ["p000:u000"],
            "selected_unit_ids": ["p000:u000"],
            "selected_source_ids": ["p000"],
            "answer": "red blue [1]",
            "citation_indices": [1],
            "minicheck": {"precision": 1.0, "recall": 0.5},
        },
        {
            "query_id": "q-rgb",
            "retrieved_unit_ids": ["p000:u000"],
            "selected_unit_ids": ["p000:u000"],
            "selected_source_ids": ["p000"],
            "answer": "Yes [1]",
            "citation_indices": [1],
            "minicheck": {"precision": 1.0, "recall": 1.0},
        },
        {
            "query_id": "q-empty",
            "retrieved_unit_ids": ["p000:u000"],
            "selected_unit_ids": ["p000:u000"],
            "selected_source_ids": ["p000"],
            "answer": "",
            "citation_indices": [],
        },
        {
            "query_id": "q-invalid",
            "retrieved_unit_ids": ["p000:u000"],
            "selected_unit_ids": ["p000:u000"],
            "selected_source_ids": ["p000"],
            "answer": "answer [2]",
            "citation_indices": [2],
            "minicheck": {"precision": 1.0, "recall": 1.0},
        },
        {
            "query_id": "q-generation",
            "retrieved_unit_ids": ["p000:u000", "p001:u000"],
            "selected_unit_ids": ["p000:u000"],
            "failure_stage": "generation",
        },
        {
            "query_id": "q-scoring",
            "retrieved_unit_ids": ["p000:u000"],
            "selected_unit_ids": ["p000:u000"],
            "selected_source_ids": ["p000"],
            "answer": "answer [1]",
            "citation_indices": [1],
            "failure_stage": "scoring",
        },
    ]
    scored = score_bundle(sidecars, outcomes)
    expected_aggregate = {
        "ret": 0.8125,
        "sel": 0.75,
        "ans": 0.4375,
        "cit": 1.0 / 3.0,
        "rar": 0.25,
    }
    by_id = {str(row["query_id"]): row for row in scored["per_query"]}
    expected_failures = {
        "empty_answer": 1,
        "generation_failure": 1,
        "invalid_citation": 1,
        "missing_output": 1,
        "scoring_failure": 1,
    }
    checks = {
        "ret_known_answer": by_id["q-partial"]["metrics"]["ret"] == 0.5,
        "sel_same_gold_denominator": by_id["q-partial"]["metrics"]["sel"] == 0.5,
        "ans_hotpot_musique_token_f1": by_id["q-partial"]["metrics"]["ans"] == 0.5,
        "ans_rgb_accuracy": by_id["q-rgb"]["metrics"]["ans"] == 1.0,
        "cit_harmonic_f1": abs(by_id["q-partial"]["metrics"]["cit"] - (2.0 / 3.0)) < 1e-12,
        "rar_exact_valid_fully_supported": by_id["q-perfect"]["metrics"]["rar"] == 1.0,
        "rar_rejects_non_exact_or_partial_support": by_id["q-partial"]["metrics"]["rar"] == 0.0,
        "empty_answer_zeroed": all(
            by_id["q-empty"]["metrics"][key] == 0.0 for key in ("ans", "cit", "rar")
        ),
        "invalid_citation_zeroes_cit_rar": (
            by_id["q-invalid"]["metrics"]["ans"] == 1.0
            and by_id["q-invalid"]["metrics"]["cit"] == 0.0
            and by_id["q-invalid"]["metrics"]["rar"] == 0.0
        ),
        "generation_failure_zeroed_not_dropped": all(
            by_id["q-generation"]["metrics"][key] == 0.0
            for key in ("ans", "cit", "rar")
        ),
        "scoring_failure_zeroed_not_dropped": all(
            by_id["q-scoring"]["metrics"][key] == 0.0 for key in ("ans", "cit", "rar")
        ),
        "missing_output_zeroed_not_dropped": all(
            by_id["q-missing"]["metrics"][key] == 0.0 for key in METRIC_KEYS
        ),
        "all_metrics_keep_common_denominator": scored["denominators"] == {
            key: 8 for key in METRIC_KEYS
        },
        "failure_reasons_machine_readable": scored["failure_counts"] == expected_failures,
        "aggregate_known_answers": all(
            abs(scored["aggregate"][key] - value) < 1e-12
            for key, value in expected_aggregate.items()
        ),
    }
    validation = {
        "validation_version": "experiment04.goal1.scorer_validation.v1",
        "fixture": "synthetic_revealed_only",
        "minicheck_model_run": False,
        "n_fixture_queries": 8,
        "metric_definitions": {
            "ret": "macro support-unit recall in Retriever final Top10",
            "sel": "macro support-unit recall in final Generator context, using Ret. denominator",
            "ans": "token F1 for HotpotQA/MuSiQue; normalized exact accuracy for RGB",
            "cit": "macro per-query F1 from frozen MiniCheck precision/recall",
            "rar": "normalized alias exact match + legal citation + MiniCheck precision=recall=1",
        },
        "expected_aggregate": expected_aggregate,
        "observed_aggregate": scored["aggregate"],
        "denominators": scored["denominators"],
        "failure_counts": scored["failure_counts"],
        "checks": checks,
        "overall_pass": all(checks.values()),
    }
    _write_json(output_path, validation)
    print(
        f"scorer_validation_pass={str(validation['overall_pass']).lower()} "
        f"fixture_queries=8 common_denominator=true",
        flush=True,
    )
    return validation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("materialize", "validate-scorer", "all"))
    parser.add_argument("--sample-manifest", type=Path, default=DEFAULT_SAMPLE_MANIFEST)
    parser.add_argument("--source-cache", type=Path, default=DEFAULT_SOURCE_CACHE)
    parser.add_argument("--sealed-root", type=Path, default=DEFAULT_SEALED_ROOT)
    parser.add_argument("--data-manifest-out", type=Path, default=DEFAULT_DATA_MANIFEST)
    parser.add_argument("--scorer-validation-out", type=Path, default=DEFAULT_SCORER_VALIDATION)
    args = parser.parse_args()

    passed = True
    if args.command in {"materialize", "all"}:
        manifest = materialize(
            sample_manifest_path=args.sample_manifest,
            source_cache=args.source_cache,
            sealed_root=args.sealed_root,
            output_manifest_path=args.data_manifest_out,
        )
        passed = passed and bool(manifest["overall_pass"])
    if args.command in {"validate-scorer", "all"}:
        validation = validate_scorer(args.scorer_validation_out)
        passed = passed and bool(validation["overall_pass"])
    print(f"goal1_command_pass={str(passed).lower()}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
