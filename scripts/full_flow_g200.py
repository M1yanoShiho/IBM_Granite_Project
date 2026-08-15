"""Build provenance-isolated evidence-to-draft data for G200.

The workflow has three explicit trust boundaries:

* ``select`` runs the frozen Legacy Selector on NIAH-train runtime inputs. It
  cannot read gold, provenance labels, or reference answers.
* ``qa2d`` converts the NIAH-train question/reference pairs into frozen,
  self-contained answer sentences. This is offline target construction.
* ``materialize`` joins those frozen targets to labelled train evidence and
  emits the clean and mixed draft contexts. It never invokes a model.

No command accepts sealed600 or system-held-out inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from evidence_rag.cli.export_qa2d import (
    MAX_NEW_TOKENS as QA2D_MAX_NEW_TOKENS,
)
from evidence_rag.cli.export_qa2d import (
    input_template,
    normalise_terminal_space,
    qa2d_input,
)
from evidence_rag.contracts.models import CandidateSet, EvidenceCandidate, Query
from evidence_rag.generator.draft import DRAFT_PROMPT
from evidence_rag.selector.dual_head import load_dual_head_checkpoint
from evidence_rag.selector.models import CandidateRiskScore
from evidence_rag.selector.nli_dual_head import load_nli_dual_head_model
from evidence_rag.selector.risk_controlled import RiskControlledSelector

SELECTOR_MODEL_ID = "cross-encoder/nli-deberta-v3-base"
SELECTOR_REVISION = "6c749ce3425cd33b46d187e45b92bbf96ee12ec7"
SELECTOR_THRESHOLD = 0.9212157130241394
SELECTOR_CAP = 2
QA2D_MODEL_ID = "MarkS/bart-base-qa2d"
QA2D_REVISION = "94f286a9102dd7e178778d531d4aefcfbcbe5a07"
ALLOWED_ROLES = frozenset({"train-fit", "train-modelval"})
TRAIN_ROLE = "train-fit"
VALIDATION_ROLE = "train-modelval"
VARIANT_NAMES = (
    "support_only",
    "topk",
    "legacy_selected",
    "support_benign",
    "support_harmful",
    "support_first",
    "support_middle",
    "support_last",
)


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
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _normalise(text: str) -> str:
    value = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.findall(r"\w+", value))


def _require_empty(output_dir: Path, label: str) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"{label} output directory must be absent or empty")


def _load_roles(path: Path) -> dict[str, tuple[str, str]]:
    roles: dict[str, tuple[str, str]] = {}
    component_roles: dict[str, set[str]] = {}
    for row in _jsonl(path):
        query_id = str(row.get("query_id", ""))
        role = str(row.get("role", ""))
        component_id = str(row.get("component_id", ""))
        if not query_id or role not in ALLOWED_ROLES or not component_id:
            raise ValueError(f"invalid G200 role assignment for query {query_id!r}")
        if query_id in roles:
            raise ValueError(f"duplicate G200 role assignment: {query_id}")
        roles[query_id] = (role, component_id)
        component_roles.setdefault(component_id, set()).add(role)
    crossed = sorted(key for key, values in component_roles.items() if len(values) != 1)
    if crossed:
        raise ValueError(f"provenance components cross train/model-val: {crossed[:5]}")
    return roles


def _load_queries(path: Path, wanted: set[str]) -> dict[str, Query]:
    queries: dict[str, Query] = {}
    for row in _jsonl(path):
        query = Query.model_validate(row)
        if query.query_id not in wanted:
            continue
        if query.query_id in queries:
            raise ValueError(f"duplicate query: {query.query_id}")
        queries[query.query_id] = query
    if set(queries) != wanted:
        raise ValueError("queries do not exactly cover G200 role assignments")
    return queries


def _load_topk10(path: Path, wanted: set[str]) -> dict[str, tuple[EvidenceCandidate, ...]]:
    pools: dict[str, tuple[EvidenceCandidate, ...]] = {}
    for row in _jsonl(path):
        candidate_set = CandidateSet.model_validate(row)
        if candidate_set.query_id not in wanted:
            continue
        ordered = tuple(
            sorted(
                candidate_set.candidates,
                key=lambda item: (item.retrieval_rank, item.evidence_id),
            )[:10]
        )
        if tuple(item.retrieval_rank for item in ordered) != tuple(range(1, 11)):
            raise ValueError(f"query {candidate_set.query_id} lacks exact TopK10")
        if candidate_set.query_id in pools:
            raise ValueError(f"duplicate candidate set: {candidate_set.query_id}")
        pools[candidate_set.query_id] = ordered
    if set(pools) != wanted:
        raise ValueError("candidate pool does not exactly cover G200 role assignments")
    return pools


def select(
    *,
    queries_path: Path,
    candidate_pool_path: Path,
    role_assignments_path: Path,
    model_snapshot: Path,
    checkpoint: Path,
    output_dir: Path,
    device: str,
    batch_size: int,
) -> dict[str, object]:
    """Run the frozen Legacy Selector without accepting any gold input."""

    _require_empty(output_dir, "G200 selection")
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    roles = _load_roles(role_assignments_path)
    wanted = set(roles)
    queries = _load_queries(queries_path, wanted)
    pools = _load_topk10(candidate_pool_path, wanted)

    model = load_nli_dual_head_model(
        str(model_snapshot.resolve()),
        revision=SELECTOR_REVISION,
        identity_model_id=SELECTOR_MODEL_ID,
        local_files_only=True,
        device=device,
    )
    fingerprint = load_dual_head_checkpoint(model, checkpoint)
    model.eval()
    torch = importlib.import_module("torch")
    flat = [
        (queries[query_id], candidate)
        for query_id in sorted(wanted)
        for candidate in pools[query_id]
    ]
    scores: dict[str, dict[str, CandidateRiskScore]] = {}
    with torch.inference_mode():
        for start in range(0, len(flat), batch_size):
            batch = flat[start : start + batch_size]
            output = model(
                question=[query.text for query, _ in batch],
                candidate_text=[candidate.text for _, candidate in batch],
            )
            protects = output.protect_scores.detach().float().cpu().tolist()
            harms = output.harm_scores.detach().float().cpu().tolist()
            for (query, candidate), protect, harm in zip(
                batch, protects, harms, strict=True
            ):
                scores.setdefault(query.query_id, {})[candidate.evidence_id] = (
                    CandidateRiskScore(
                        protect_score=float(protect), harm_score=float(harm)
                    )
                )
            done = min(start + len(batch), len(flat))
            if done % 1000 == 0 or done == len(flat):
                print(f"[G200 select] {done}/{len(flat)} candidates", flush=True)

    selector = RiskControlledSelector(
        scores_by_query=scores,
        safe_threshold=SELECTOR_THRESHOLD,
        max_delete=SELECTOR_CAP,
    )
    rows: list[dict[str, object]] = []
    changed = 0
    dropped = 0
    for query_id in sorted(wanted):
        candidates = CandidateSet(query_id=query_id, candidates=pools[query_id])
        _result, trace = selector.select_with_trace(queries[query_id], candidates, 10)
        changed += int(bool(trace.dropped_evidence_ids))
        dropped += len(trace.dropped_evidence_ids)
        role, component_id = roles[query_id]
        rows.append(
            {
                "schema_version": "full-flow-g200-selection-row-v1",
                "query_id": query_id,
                "role": role,
                "component_id": component_id,
                "selected_evidence_ids": list(trace.selected_evidence_ids),
                "dropped_evidence_ids": list(trace.dropped_evidence_ids),
                "trace": trace.model_dump(mode="json"),
            }
        )
    trace_path = output_dir / "selection_trace.jsonl"
    _write_jsonl(trace_path, rows)
    manifest: dict[str, object] = {
        "schema_version": "full-flow-g200-selection-manifest-v1",
        "status": "COMPLETE",
        "source_role": "NIAH train role assignments only",
        "queries": len(rows),
        "changed_queries": changed,
        "dropped_candidates": dropped,
        "gold_loaded_at_runtime": False,
        "reference_answers_loaded_at_runtime": False,
        "selector_model_id": SELECTOR_MODEL_ID,
        "selector_revision": SELECTOR_REVISION,
        "selector_checkpoint_sha256": _sha256(checkpoint),
        "selector_state_sha256": fingerprint.weights_sha256,
        "safe_threshold": SELECTOR_THRESHOLD,
        "cap": SELECTOR_CAP,
        "input_sha256": {
            "queries": _sha256(queries_path),
            "candidate_pool": _sha256(candidate_pool_path),
            "role_assignments": _sha256(role_assignments_path),
        },
        "selection_trace_sha256": _sha256(trace_path),
    }
    _write_json(output_dir / "selection_manifest.json", manifest)
    return manifest


Qa2dGenerator = Callable[[Sequence[tuple[str, str]]], Sequence[str]]


def _load_qa2d_generator(
    *, model_id: str, revision: str, batch_size: int
) -> Qa2dGenerator:
    if model_id != QA2D_MODEL_ID or revision != QA2D_REVISION:
        raise ValueError("G200 QA2D model/revision differs from the frozen configuration")
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    template = input_template(model_id)
    cache_dir = os.getenv("MODEL_CACHE_DIR") or None
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id, revision=revision, cache_dir=cache_dir
    )
    model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
        model_id, revision=revision, cache_dir=cache_dir
    )
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"[G200 qa2d] {model_id}@{revision} on {device}", flush=True)

    def generate(pairs: Sequence[tuple[str, str]]) -> Sequence[str]:
        sentences: list[str] = []
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]
            encoded = tokenizer(
                [qa2d_input(question, answer, template) for question, answer in batch],
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.no_grad():
                generated = model.generate(
                    **encoded,
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=QA2D_MAX_NEW_TOKENS,
                )
            sentences.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
            print(f"[G200 qa2d] {start + len(batch)}/{len(pairs)}", flush=True)
        return sentences

    return generate


def export_qa2d_targets(
    *,
    queries_path: Path,
    gold_path: Path,
    role_assignments_path: Path,
    output_dir: Path,
    model_id: str,
    revision: str,
    batch_size: int,
    generator: Qa2dGenerator | None = None,
) -> dict[str, object]:
    """Freeze one self-contained semantic target for each allowed train query."""

    _require_empty(output_dir, "G200 QA2D")
    roles = _load_roles(role_assignments_path)
    wanted = set(roles)
    queries = _load_queries(queries_path, wanted)
    gold: dict[str, str] = {}
    for row in _jsonl(gold_path):
        query_id = str(row.get("query_id", ""))
        if query_id not in wanted:
            continue
        references = row.get("reference_answers")
        if not isinstance(references, list) or len(references) != 1:
            raise ValueError(f"G200 requires one NIAH reference answer for {query_id}")
        answer = str(references[0]).strip()
        if not answer:
            raise ValueError(f"blank G200 reference answer for {query_id}")
        if query_id in gold:
            raise ValueError(f"duplicate G200 gold row: {query_id}")
        gold[query_id] = answer
    if set(gold) != wanted:
        raise ValueError("gold does not exactly cover G200 role assignments")

    pairs = [(queries[query_id].text, gold[query_id]) for query_id in sorted(wanted)]
    generate = generator or _load_qa2d_generator(
        model_id=model_id, revision=revision, batch_size=batch_size
    )
    raw_sentences = list(generate(pairs))
    if len(raw_sentences) != len(pairs):
        raise ValueError("QA2D output count differs from input pair count")
    rows: list[dict[str, object]] = []
    preserved = 0
    for query_id, raw_sentence in zip(sorted(wanted), raw_sentences, strict=True):
        sentence = normalise_terminal_space(str(raw_sentence).strip())
        if not sentence:
            raise ValueError(f"empty QA2D target for {query_id}")
        answer_preserved = _normalise(gold[query_id]) in _normalise(sentence)
        preserved += int(answer_preserved)
        role, component_id = roles[query_id]
        rows.append(
            {
                "schema_version": "full-flow-g200-qa2d-target-v1",
                "query_id": query_id,
                "role": role,
                "component_id": component_id,
                "question": queries[query_id].text,
                "answer": gold[query_id],
                "declarative": sentence,
                "answer_preserved": answer_preserved,
                "model": model_id,
                "revision": revision,
                "input_template": input_template(model_id),
            }
        )
    target_path = output_dir / "qa2d_targets.jsonl"
    _write_jsonl(target_path, rows)
    manifest: dict[str, object] = {
        "schema_version": "full-flow-g200-qa2d-manifest-v1",
        "status": "COMPLETE",
        "role": "offline target construction only; not a runtime system component",
        "model": model_id,
        "revision": revision,
        "input_template": input_template(model_id),
        "decode": {
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens": QA2D_MAX_NEW_TOKENS,
        },
        "queries": len(rows),
        "answer_preserved": preserved,
        "answer_not_preserved": len(rows) - preserved,
        "input_sha256": {
            "queries": _sha256(queries_path),
            "gold": _sha256(gold_path),
            "role_assignments": _sha256(role_assignments_path),
        },
        "qa2d_targets_sha256": _sha256(target_path),
    }
    _write_json(output_dir / "qa2d_manifest.json", manifest)
    return manifest


def _is_harmful(item: EvidenceCandidate) -> bool:
    by_document = item.document_id.startswith("cf::")
    by_source = item.source_uri.startswith("synthetic://cf/")
    if by_document != by_source:
        raise ValueError(f"counterfactual provenance disagrees for {item.evidence_id}")
    return by_document


def _context(candidates: Sequence[EvidenceCandidate]) -> str:
    return "\n".join(
        f"[{index}] ({item.evidence_id}) {item.text}"
        for index, item in enumerate(candidates, start=1)
    )


def _prompt(question: str, candidates: Sequence[EvidenceCandidate]) -> str:
    return DRAFT_PROMPT.format(context=_context(candidates), question=question)


def _target(declarative: str, citation_index: int) -> str:
    sentence = declarative.strip()
    punctuation = sentence[-1] if sentence[-1:] in {".", "!", "?"} else "."
    if sentence[-1:] in {".", "!", "?"}:
        sentence = sentence[:-1].rstrip()
    return f"{sentence} [{citation_index}]{punctuation}"


def _load_gold(path: Path, wanted: set[str]) -> dict[str, tuple[str, frozenset[str]]]:
    output: dict[str, tuple[str, frozenset[str]]] = {}
    for row in _jsonl(path):
        query_id = str(row.get("query_id", ""))
        if query_id not in wanted:
            continue
        references = row.get("reference_answers")
        relevant = row.get("relevant_document_ids")
        if not isinstance(references, list) or len(references) != 1:
            raise ValueError(f"G200 requires one NIAH reference answer for {query_id}")
        if not isinstance(relevant, list) or not relevant:
            raise ValueError(f"G200 requires relevant documents for {query_id}")
        if query_id in output:
            raise ValueError(f"duplicate G200 gold row: {query_id}")
        output[query_id] = (str(references[0]), frozenset(str(item) for item in relevant))
    if set(output) != wanted:
        raise ValueError("gold does not exactly cover G200 role assignments")
    return output


def _load_component_map(path: Path, roles: Mapping[str, tuple[str, str]]) -> None:
    seen: dict[str, str] = {}
    for row in _jsonl(path):
        query_id = str(row.get("query_id", ""))
        if query_id not in roles:
            continue
        component_id = str(row.get("component_id", ""))
        if component_id != roles[query_id][1]:
            raise ValueError(f"component map disagrees for {query_id}")
        if query_id in seen:
            raise ValueError(f"duplicate component map row: {query_id}")
        seen[query_id] = component_id
    if set(seen) != set(roles):
        raise ValueError("component map does not exactly cover G200 role assignments")


def _load_selection(
    path: Path,
    *,
    roles: Mapping[str, tuple[str, str]],
    pools: Mapping[str, tuple[EvidenceCandidate, ...]],
) -> dict[str, tuple[EvidenceCandidate, ...]]:
    selected: dict[str, tuple[EvidenceCandidate, ...]] = {}
    for row in _jsonl(path):
        query_id = str(row.get("query_id", ""))
        if query_id not in roles:
            raise ValueError(f"selection contains non-G200 query: {query_id}")
        raw_ids = row.get("selected_evidence_ids")
        if not isinstance(raw_ids, list) or not raw_ids:
            raise ValueError(f"invalid G200 selection for {query_id}")
        if str(row.get("role", "")) != roles[query_id][0]:
            raise ValueError(f"selection role disagrees for {query_id}")
        if str(row.get("component_id", "")) != roles[query_id][1]:
            raise ValueError(f"selection component disagrees for {query_id}")
        wanted_ids = tuple(str(item) for item in raw_ids)
        topk = pools[query_id]
        topk_ids = tuple(item.evidence_id for item in topk)
        if not set(wanted_ids) <= set(topk_ids):
            raise ValueError(f"selection contains unknown evidence for {query_id}")
        ordered = tuple(item for item in topk if item.evidence_id in set(wanted_ids))
        if tuple(item.evidence_id for item in ordered) != wanted_ids:
            raise ValueError(f"selection order differs from TopK10 for {query_id}")
        if query_id in selected:
            raise ValueError(f"duplicate G200 selection: {query_id}")
        selected[query_id] = ordered
    if set(selected) != set(roles):
        raise ValueError("selection does not exactly cover G200 role assignments")
    return selected


def _load_targets(
    path: Path, roles: Mapping[str, tuple[str, str]]
) -> dict[str, Mapping[str, Any]]:
    targets: dict[str, Mapping[str, Any]] = {}
    for row in _jsonl(path):
        query_id = str(row.get("query_id", ""))
        if query_id not in roles:
            raise ValueError(f"QA2D targets contain non-G200 query: {query_id}")
        if str(row.get("model", "")) != QA2D_MODEL_ID:
            raise ValueError(f"QA2D model disagrees for {query_id}")
        if str(row.get("revision", "")) != QA2D_REVISION:
            raise ValueError(f"QA2D revision disagrees for {query_id}")
        if query_id in targets:
            raise ValueError(f"duplicate QA2D target: {query_id}")
        targets[query_id] = row
    if set(targets) != set(roles):
        raise ValueError("QA2D targets do not exactly cover G200 role assignments")
    return targets


def _ordered_by_rank(candidates: Iterable[EvidenceCandidate]) -> tuple[EvidenceCandidate, ...]:
    return tuple(sorted(candidates, key=lambda item: (item.retrieval_rank, item.evidence_id)))


def _variant_row(
    *,
    question: str,
    answer: str,
    declarative: str,
    candidates: Sequence[EvidenceCandidate],
    relevant_document_ids: frozenset[str],
) -> dict[str, object] | None:
    carriers = [
        (index, item)
        for index, item in enumerate(candidates, start=1)
        if item.document_id in relevant_document_ids
        and _normalise(answer) in _normalise(item.text)
    ]
    if not carriers:
        return None
    citation_index, carrier = carriers[0]
    target = _target(declarative, citation_index)
    return {
        "evidence_ids": [item.evidence_id for item in candidates],
        "support_evidence_id": carrier.evidence_id,
        "support_citation_index": citation_index,
        "prompt": _prompt(question, candidates),
        "target": target,
    }


def materialize(
    *,
    queries_path: Path,
    candidate_pool_path: Path,
    gold_path: Path,
    role_assignments_path: Path,
    component_map_path: Path,
    selection_trace_path: Path,
    qa2d_targets_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Join frozen offline labels to contexts and emit no model calls."""

    _require_empty(output_dir, "G200 materialization")
    roles = _load_roles(role_assignments_path)
    _load_component_map(component_map_path, roles)
    wanted = set(roles)
    queries = _load_queries(queries_path, wanted)
    pools = _load_topk10(candidate_pool_path, wanted)
    gold = _load_gold(gold_path, wanted)
    selected = _load_selection(selection_trace_path, roles=roles, pools=pools)
    targets = _load_targets(qa2d_targets_path, roles)

    counts: Counter[str] = Counter()
    rows: list[dict[str, object]] = []
    for query_id in sorted(wanted):
        counts["role_assigned_queries"] += 1
        role, component_id = roles[query_id]
        question = queries[query_id].text
        answer, relevant_document_ids = gold[query_id]
        target_row = targets[query_id]
        if str(target_row.get("question", "")) != question:
            raise ValueError(f"QA2D question disagrees for {query_id}")
        if str(target_row.get("answer", "")) != answer:
            raise ValueError(f"QA2D answer disagrees for {query_id}")
        declarative = str(target_row.get("declarative", "")).strip()
        if not bool(target_row.get("answer_preserved")) or not declarative:
            counts["excluded_qa2d_answer_not_preserved"] += 1
            continue

        topk = pools[query_id]
        support = tuple(item for item in topk if item.document_id in relevant_document_ids)
        harmful = tuple(
            item
            for item in topk
            if item.document_id not in relevant_document_ids and _is_harmful(item)
        )
        benign = tuple(
            item
            for item in topk
            if item.document_id not in relevant_document_ids and not _is_harmful(item)
        )
        if len(support) + len(harmful) + len(benign) != len(topk):
            raise AssertionError("G200 evidence partition is incomplete")
        if not support:
            counts["excluded_no_support"] += 1
            continue
        if not harmful:
            counts["excluded_no_harmful"] += 1
            continue
        if not benign:
            counts["excluded_no_benign"] += 1
            continue
        if not any(_normalise(answer) in _normalise(item.text) for item in support):
            counts["excluded_no_exact_supported_answer"] += 1
            continue

        distractors = _ordered_by_rank((*benign, *harmful))
        middle = len(distractors) // 2
        contexts: dict[str, tuple[EvidenceCandidate, ...]] = {
            "support_only": _ordered_by_rank(support),
            "topk": topk,
            "legacy_selected": selected[query_id],
            "support_benign": _ordered_by_rank((*support, *benign)),
            "support_harmful": _ordered_by_rank((*support, *harmful)),
            "support_first": (*_ordered_by_rank(support), *distractors),
            "support_middle": (
                *distractors[:middle],
                *_ordered_by_rank(support),
                *distractors[middle:],
            ),
            "support_last": (*distractors, *_ordered_by_rank(support)),
        }
        if tuple(contexts) != VARIANT_NAMES:
            raise AssertionError("G200 variant order differs from the frozen order")
        topk_set = {item.evidence_id for item in topk}
        for name in ("support_first", "support_middle", "support_last"):
            if {item.evidence_id for item in contexts[name]} != topk_set:
                raise AssertionError(f"{name} does not preserve the TopK evidence set")

        variants: dict[str, dict[str, object]] = {}
        for name, context in contexts.items():
            variant = _variant_row(
                question=question,
                answer=answer,
                declarative=declarative,
                candidates=context,
                relevant_document_ids=relevant_document_ids,
            )
            if variant is None:
                counts[f"excluded_target_unsupported_in_{name}"] += 1
                break
            variants[name] = variant
        if len(variants) != len(VARIANT_NAMES):
            continue

        row = {
            "schema_version": "full-flow-g200-case-v1",
            "query_id": query_id,
            "role": role,
            "component_id": component_id,
            "question": question,
            "answer": answer,
            "semantic_target": declarative,
            "semantic_target_sha256": _sha256_text(declarative),
            "variants": variants,
        }
        rows.append(row)
        counts["eligible"] += 1
        counts[f"eligible_{role}"] += 1

    training = [row for row in rows if row["role"] == TRAIN_ROLE]
    validation = [row for row in rows if row["role"] == VALIDATION_ROLE]
    train_components = {str(row["component_id"]) for row in training}
    validation_components = {str(row["component_id"]) for row in validation}
    if train_components & validation_components:
        raise AssertionError("G200 train/model-val provenance components overlap")
    if not training or not validation:
        raise ValueError("G200 requires non-empty train and model-val materializations")

    train_path = output_dir / "train_cases.jsonl"
    validation_path = output_dir / "validation_cases.jsonl"
    _write_jsonl(train_path, training)
    _write_jsonl(validation_path, validation)
    variants_per_query = len(VARIANT_NAMES)
    manifest: dict[str, object] = {
        "schema_version": "full-flow-g200-data-manifest-v1",
        "status": "COMPLETE",
        "source_role": "NIAH train only",
        "sealed_or_heldout_read": False,
        "dev_read": False,
        "gold_use": "offline target construction and support audit only",
        "selection_runtime_gold_loaded": False,
        "split_rule": "pre-existing component-grouped train-fit/train-modelval roles",
        "provenance_component_overlap": 0,
        "semantic_target": (
            "frozen QA2D(question, single NIAH reference); answer-preserving rows only"
        ),
        "qa2d": {
            "model": QA2D_MODEL_ID,
            "revision": QA2D_REVISION,
            "runtime_component": False,
        },
        "draft_model": "ibm-granite/granite-4.1-3b",
        "draft_prompt_sha256": _sha256_text(DRAFT_PROMPT),
        "variant_order": list(VARIANT_NAMES),
        "counts": dict(counts),
        "train_queries": len(training),
        "validation_queries": len(validation),
        "variants_per_query": variants_per_query,
        "gc_examples": len(training) * variants_per_query,
        "gm_examples": len(training) * variants_per_query,
        "gc_definition": "support_only duplicated once per frozen mixed variant",
        "gm_definition": "one example from each frozen context variant",
        "train_cases_sha256": _sha256(train_path),
        "validation_cases_sha256": _sha256(validation_path),
        "input_sha256": {
            "queries": _sha256(queries_path),
            "candidate_pool": _sha256(candidate_pool_path),
            "gold": _sha256(gold_path),
            "role_assignments": _sha256(role_assignments_path),
            "component_map": _sha256(component_map_path),
            "selection_trace": _sha256(selection_trace_path),
            "qa2d_targets": _sha256(qa2d_targets_path),
        },
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    choose = commands.add_parser("select")
    choose.add_argument("--queries", required=True, type=Path)
    choose.add_argument("--candidate-pool", required=True, type=Path)
    choose.add_argument("--role-assignments", required=True, type=Path)
    choose.add_argument("--model-snapshot", required=True, type=Path)
    choose.add_argument("--checkpoint", required=True, type=Path)
    choose.add_argument("--output-dir", required=True, type=Path)
    choose.add_argument("--device", default="cuda:0")
    choose.add_argument("--batch-size", type=int, default=32)

    qa2d = commands.add_parser("qa2d")
    qa2d.add_argument("--queries", required=True, type=Path)
    qa2d.add_argument("--gold", required=True, type=Path)
    qa2d.add_argument("--role-assignments", required=True, type=Path)
    qa2d.add_argument("--output-dir", required=True, type=Path)
    qa2d.add_argument("--model", default=QA2D_MODEL_ID)
    qa2d.add_argument("--revision", default=QA2D_REVISION)
    qa2d.add_argument("--batch-size", type=int, default=32)

    data = commands.add_parser("materialize")
    data.add_argument("--queries", required=True, type=Path)
    data.add_argument("--candidate-pool", required=True, type=Path)
    data.add_argument("--gold", required=True, type=Path)
    data.add_argument("--role-assignments", required=True, type=Path)
    data.add_argument("--component-map", required=True, type=Path)
    data.add_argument("--selection-trace", required=True, type=Path)
    data.add_argument("--qa2d-targets", required=True, type=Path)
    data.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "select":
        report = select(
            queries_path=args.queries,
            candidate_pool_path=args.candidate_pool,
            role_assignments_path=args.role_assignments,
            model_snapshot=args.model_snapshot,
            checkpoint=args.checkpoint,
            output_dir=args.output_dir.resolve(),
            device=args.device,
            batch_size=args.batch_size,
        )
    elif args.command == "qa2d":
        report = export_qa2d_targets(
            queries_path=args.queries,
            gold_path=args.gold,
            role_assignments_path=args.role_assignments,
            output_dir=args.output_dir.resolve(),
            model_id=args.model,
            revision=args.revision,
            batch_size=args.batch_size,
        )
    else:
        report = materialize(
            queries_path=args.queries,
            candidate_pool_path=args.candidate_pool,
            gold_path=args.gold,
            role_assignments_path=args.role_assignments,
            component_map_path=args.component_map,
            selection_trace_path=args.selection_trace,
            qa2d_targets_path=args.qa2d_targets,
            output_dir=args.output_dir.resolve(),
        )
    print(json.dumps(report, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
