"""Run the frozen F005 confirmation on the independent sealed600 set.

The workflow is intentionally split in three commands:

``select`` scores sealed TopK10 with the frozen Lean Seed-13 Selector and writes
an immutable decision trace without reading gold. ``run`` consumes those frozen
decisions and compares the default system, a Generator-only Mixed-LoRA arm, and
the full Selector+Mixed-LoRA system, again without gold. ``score`` is the only
command allowed to read reference answers and provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import traceback
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from full_flow_f004 import _routing_rows

from evidence_rag.contracts.models import (
    CandidateSet,
    EvidenceCandidate,
    GenerationResult,
    Query,
    QueryChecklist,
    SelectedEvidenceSet,
    strip_annotations,
)
from evidence_rag.evaluation.paired_metric import compare_paired
from evidence_rag.evaluation.scoring import answer_match
from evidence_rag.generator.draft import DraftAnswerGenerator, KeyFactDraftAnswerGenerator
from evidence_rag.generator.granite import (
    GraniteGenerationConfig,
    NamedAdapterTextGenerator,
    PeftGraniteLLMClient,
)
from evidence_rag.generator.key_facts import notes_manifest
from evidence_rag.generator.nli import TrueNLIModel
from evidence_rag.generator.verify_annotate import VerifyAnnotateGenerator
from evidence_rag.infrastructure.datasets import GoldCase
from evidence_rag.selector.dual_head import load_dual_head_checkpoint
from evidence_rag.selector.models import CandidateRiskScore
from evidence_rag.selector.nli_dual_head import load_nli_dual_head_model
from evidence_rag.selector.risk_controlled import RiskControlledSelector

ARMS = ("A_TopK_Base", "B_TopK_Mixed", "C_Selector_Mixed")
EXPECTED_CASES = 600
SELECTOR_MODEL_ID = "cross-encoder/nli-deberta-v3-base"
SELECTOR_REVISION = "6c749ce3425cd33b46d187e45b92bbf96ee12ec7"


@dataclass(frozen=True, slots=True)
class ConfirmationCase:
    query_id: str
    question: str
    topk10: tuple[EvidenceCandidate, ...]
    selected: tuple[EvidenceCandidate, ...]

    @property
    def selector_changed(self) -> bool:
        return tuple(item.evidence_id for item in self.topk10) != tuple(
            item.evidence_id for item in self.selected
        )


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonl(path: Path) -> Iterable[Mapping[str, Any]]:
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, Mapping):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        yield value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_queries(path: Path) -> dict[str, Query]:
    output: dict[str, Query] = {}
    for value in _jsonl(path):
        query = Query.model_validate(value)
        if query.query_id in output:
            raise ValueError(f"duplicate query: {query.query_id}")
        output[query.query_id] = query
    return output


def _topk10(path: Path) -> dict[str, tuple[EvidenceCandidate, ...]]:
    output: dict[str, tuple[EvidenceCandidate, ...]] = {}
    for value in _jsonl(path):
        candidate_set = CandidateSet.model_validate(value)
        ordered = tuple(
            sorted(
                candidate_set.candidates,
                key=lambda item: (item.retrieval_rank, item.evidence_id),
            )[:10]
        )
        if tuple(item.retrieval_rank for item in ordered) != tuple(range(1, 11)):
            raise ValueError(f"query {candidate_set.query_id} lacks exact TopK10")
        if candidate_set.query_id in output:
            raise ValueError(f"duplicate candidate set: {candidate_set.query_id}")
        output[candidate_set.query_id] = ordered
    return output


def _require_same_600(queries: Mapping[str, object], pools: Mapping[str, object]) -> None:
    if set(queries) != set(pools):
        raise ValueError("sealed query and candidate IDs differ")
    if len(queries) != EXPECTED_CASES:
        raise ValueError(f"sealed set has {len(queries)} cases, expected {EXPECTED_CASES}")


def select(
    *,
    queries_path: Path,
    candidate_pool_path: Path,
    model_snapshot: Path,
    checkpoint: Path,
    safe_threshold: float,
    cap: int,
    output_dir: Path,
    device: str,
    batch_size: int,
) -> dict[str, object]:
    """Apply the already-frozen Lean policy without reading gold or provenance."""

    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("F005 selection output must be absent or empty")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    queries = _load_queries(queries_path)
    pools = _topk10(candidate_pool_path)
    _require_same_600(queries, pools)

    model = load_nli_dual_head_model(
        str(model_snapshot.resolve()),
        revision=SELECTOR_REVISION,
        identity_model_id=SELECTOR_MODEL_ID,
        local_files_only=True,
        device=device,
    )
    fingerprint = load_dual_head_checkpoint(model, checkpoint)
    model.eval()
    torch = __import__("torch")
    flat = [
        (queries[query_id], candidate)
        for query_id in sorted(queries)
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
                scores.setdefault(query.query_id, {})[candidate.evidence_id] = CandidateRiskScore(
                    protect_score=float(protect), harm_score=float(harm)
                )
            done = min(start + len(batch), len(flat))
            if done % 1000 == 0 or done == len(flat):
                print(f"[F005 select] {done}/{len(flat)} candidates", flush=True)

    selector = RiskControlledSelector(
        scores_by_query=scores,
        safe_threshold=safe_threshold,
        max_delete=cap,
    )
    rows: list[dict[str, object]] = []
    changed = 0
    dropped = 0
    for query_id in sorted(queries):
        candidates = CandidateSet(query_id=query_id, candidates=pools[query_id])
        _result, trace = selector.select_with_trace(queries[query_id], candidates, 10)
        changed += int(bool(trace.dropped_evidence_ids))
        dropped += len(trace.dropped_evidence_ids)
        rows.append(
            {
                "schema_version": "full-flow-f005-selection-row-v1",
                "query_id": query_id,
                "selected_evidence_ids": list(trace.selected_evidence_ids),
                "dropped_evidence_ids": list(trace.dropped_evidence_ids),
                "trace": trace.model_dump(mode="json"),
            }
        )
    _write_jsonl(output_dir / "selection_trace.jsonl", rows)
    manifest: dict[str, object] = {
        "schema_version": "full-flow-f005-selection-manifest-v1",
        "status": "COMPLETE",
        "queries": len(rows),
        "changed_queries": changed,
        "dropped_candidates": dropped,
        "gold_loaded_at_runtime": False,
        "selector_model_id": SELECTOR_MODEL_ID,
        "selector_revision": SELECTOR_REVISION,
        "selector_checkpoint_sha256": _sha256(checkpoint),
        "selector_state_sha256": fingerprint.weights_sha256,
        "safe_threshold": safe_threshold,
        "cap": cap,
        "queries_sha256": _sha256(queries_path),
        "candidate_pool_sha256": _sha256(candidate_pool_path),
        "selection_trace_sha256": _sha256(output_dir / "selection_trace.jsonl"),
    }
    _write_json(output_dir / "selection_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=True, sort_keys=True))
    return manifest


def load_cases(
    queries_path: Path,
    candidate_pool_path: Path,
    selection_trace_path: Path,
    *,
    require_expected_count: bool = True,
) -> tuple[ConfirmationCase, ...]:
    queries = _load_queries(queries_path)
    pools = _topk10(candidate_pool_path)
    decisions: dict[str, tuple[str, ...]] = {}
    for value in _jsonl(selection_trace_path):
        query_id = str(value.get("query_id", ""))
        raw = value.get("selected_evidence_ids")
        if not query_id or not isinstance(raw, list):
            raise ValueError("invalid F005 selection row")
        if query_id in decisions:
            raise ValueError(f"duplicate F005 selection: {query_id}")
        decisions[query_id] = tuple(str(item) for item in raw)
    if set(queries) != set(pools) or set(queries) != set(decisions):
        raise ValueError("F005 query, pool and selection IDs differ")
    if require_expected_count and len(queries) != EXPECTED_CASES:
        raise ValueError(f"F005 has {len(queries)} cases, expected {EXPECTED_CASES}")
    cases: list[ConfirmationCase] = []
    for query_id in sorted(queries):
        topk = pools[query_id]
        by_id = {item.evidence_id: item for item in topk}
        selected_ids = decisions[query_id]
        if not selected_ids or set(selected_ids) - set(by_id):
            raise ValueError(f"invalid selected IDs for {query_id}")
        selected = tuple(item for item in topk if item.evidence_id in set(selected_ids))
        if tuple(item.evidence_id for item in selected) != selected_ids:
            raise ValueError(f"selected order differs from TopK10 for {query_id}")
        cases.append(
            ConfirmationCase(
                query_id=query_id,
                question=queries[query_id].text,
                topk10=topk,
                selected=selected,
            )
        )
    return tuple(cases)


def _run_one(
    generator: VerifyAnnotateGenerator,
    query: Query,
    checklist: QueryChecklist,
    selected: SelectedEvidenceSet,
    draft: KeyFactDraftAnswerGenerator | None = None,
) -> dict[str, object]:
    started = time.perf_counter()
    try:
        generation = generator.generate(query, checklist, selected)
    except Exception as error:  # noqa: BLE001 -- failures are measured outcomes
        traceback.print_exc()
        return {
            "generation": GenerationResult(
                query_id=query.query_id, answer="", cited_evidence_ids=()
            ).model_dump(mode="json"),
            "routing": [],
            "notes": [],
            "error": f"{type(error).__name__}: {error}",
            "seconds": time.perf_counter() - started,
        }
    return {
        "generation": generation.model_dump(mode="json"),
        "routing": _routing_rows(generator),
        "notes": [] if draft is None else json.loads(notes_manifest(draft.last_notes)),
        "error": None,
        "seconds": time.perf_counter() - started,
    }


def run(
    *,
    queries_path: Path,
    candidate_pool_path: Path,
    selection_trace_path: Path,
    model_snapshot: Path,
    true_model_id: str,
    mixed_adapter: Path,
    output_dir: Path,
    limit: int | None = None,
) -> dict[str, object]:
    """Run the three frozen system arms without reading gold."""

    cases = load_cases(queries_path, candidate_pool_path, selection_trace_path)
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        cases = cases[:limit]
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("F005 run output must be absent or empty")

    client = PeftGraniteLLMClient(
        model_id=str(model_snapshot.resolve()),
        adapters={"mixed": str(mixed_adapter.resolve())},
        config=GraniteGenerationConfig(max_new_tokens=256, temperature=0.0, top_p=1.0),
    )
    nli = TrueNLIModel(model_id=true_model_id)
    base = VerifyAnnotateGenerator(
        draft_generator=DraftAnswerGenerator(llm=client),
        nli=nli,
        entity_gate="observe",
        abstain_when_unverified=False,
    )
    mixed_draft = KeyFactDraftAnswerGenerator(
        client,
        note_llm=NamedAdapterTextGenerator(client, "mixed"),
        guided=False,
    )
    mixed = VerifyAnnotateGenerator(
        draft_generator=mixed_draft,
        nli=nli,
        entity_gate="observe",
        abstain_when_unverified=False,
    )

    rows: list[dict[str, object]] = []
    started = time.perf_counter()
    for index, case in enumerate(cases, 1):
        query = Query(query_id=case.query_id, text=case.question)
        checklist = QueryChecklist(query_id=case.query_id, focus=case.question, required_facts=())
        topk = SelectedEvidenceSet(query_id=case.query_id, evidence=case.topk10)
        selected = SelectedEvidenceSet(query_id=case.query_id, evidence=case.selected)
        arm_a = _run_one(base, query, checklist, topk)
        arm_b = _run_one(mixed, query, checklist, topk, mixed_draft)
        if case.selector_changed:
            arm_c = _run_one(mixed, query, checklist, selected, mixed_draft)
            arm_c["reused_from"] = None
        else:
            arm_c = {**arm_b, "reused_from": ARMS[1]}
        arm_a["reused_from"] = None
        arm_b["reused_from"] = None
        rows.append(
            {
                "schema_version": "full-flow-f005-generation-row-v1",
                "query_id": case.query_id,
                "question": case.question,
                "selector_changed": case.selector_changed,
                "topk10": [item.model_dump(mode="json") for item in case.topk10],
                "selected_evidence_ids": [item.evidence_id for item in case.selected],
                "arms": {ARMS[0]: arm_a, ARMS[1]: arm_b, ARMS[2]: arm_c},
            }
        )
        if index % 10 == 0 or index == len(cases):
            print(
                f"[F005 run] {index}/{len(cases)} "
                f"{(time.perf_counter()-started)/index:.2f}s/case",
                flush=True,
            )
    _write_jsonl(output_dir / "generations.jsonl", rows)
    manifest: dict[str, object] = {
        "schema_version": "full-flow-f005-run-manifest-v1",
        "status": "COMPLETE",
        "queries": len(rows),
        "changed_queries": sum(bool(row["selector_changed"]) for row in rows),
        "gold_loaded_at_runtime": False,
        "same_process": True,
        "shared_granite_base": True,
        "shared_true": True,
        "mixed_adapter_scope": "key-fact extraction only",
        "draft_and_claim_splitter_scope": "frozen Granite base with adapters disabled",
        "selection_trace_sha256": _sha256(selection_trace_path),
        "mixed_adapter_config_sha256": _sha256(mixed_adapter / "adapter_config.json"),
        "generations_sha256": _sha256(output_dir / "generations.jsonl"),
    }
    _write_json(output_dir / "run_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=True, sort_keys=True))
    return manifest


def _load_gold(path: Path, wanted: set[str]) -> dict[str, GoldCase]:
    output: dict[str, GoldCase] = {}
    for value in _jsonl(path):
        gold = GoldCase.model_validate(value)
        if gold.query_id in wanted:
            output[gold.query_id] = gold
    if set(output) != wanted:
        raise ValueError("gold does not exactly cover F005 generations")
    return output


def _parent_map(path: Path) -> dict[str, str]:
    output: dict[str, str] = {}
    for value in _jsonl(path):
        document_id = str(value.get("document_id", ""))
        parent = str(value.get("source_parent_id", ""))
        if not document_id or not parent:
            raise ValueError("invalid source-parent row")
        output[document_id] = parent
    return output


def _component_ids(
    gold: Mapping[str, GoldCase], parents: Mapping[str, str]
) -> dict[str, str]:
    """Join queries sharing any relevant parent page into one bootstrap component."""

    parent_by_query: dict[str, set[str]] = {}
    for query_id, case in gold.items():
        values = {parents[document_id] for document_id in case.relevant_document_ids or ()}
        if not values:
            raise ValueError(f"no parent-page component for {query_id}")
        parent_by_query[query_id] = values
    parent_owner: dict[str, str] = {}
    adjacency: dict[str, set[str]] = {query_id: set() for query_id in gold}
    for query_id in sorted(gold):
        for parent in parent_by_query[query_id]:
            owner = parent_owner.setdefault(parent, query_id)
            adjacency[query_id].add(owner)
            adjacency[owner].add(query_id)
    output: dict[str, str] = {}
    unseen = set(gold)
    while unseen:
        start = min(unseen)
        stack = [start]
        members: set[str] = set()
        while stack:
            query_id = stack.pop()
            if query_id in members:
                continue
            members.add(query_id)
            stack.extend(adjacency[query_id] - members)
        unseen -= members
        component = f"sealed-parent:{min(members)}"
        output.update({query_id: component for query_id in members})
    return output


def _provenance(path: Path) -> dict[str, str]:
    output: dict[str, str] = {}
    for value in _jsonl(path):
        output[str(value["query_id"])] = str(value["counterfactual_document_id"])
    return output


def score(
    rows: Sequence[Mapping[str, Any]],
    gold: Mapping[str, GoldCase],
    parents: Mapping[str, str],
    counterfactuals: Mapping[str, str],
) -> dict[str, object]:
    wanted = {str(row["query_id"]) for row in rows}
    if wanted != set(gold) or wanted != set(counterfactuals):
        raise ValueError("F005 score inputs have different query IDs")
    components = _component_ids(gold, parents)
    values: dict[str, dict[str, dict[str, float]]] = {arm: {} for arm in ARMS}
    errors = dict.fromkeys(ARMS, 0)
    notes = dict.fromkeys(ARMS, 0)
    changed: set[str] = set()
    total_dropped = 0
    harmful_pool_hits = 0
    harmful_dropped = 0
    gold_document_drops = 0
    recall_loss_queries = 0
    for row in rows:
        query_id = str(row["query_id"])
        if bool(row["selector_changed"]):
            changed.add(query_id)
        topk = tuple(EvidenceCandidate.model_validate(item) for item in row["topk10"])
        topk_by_id = {item.evidence_id: item for item in topk}
        selected_ids = {str(item) for item in row["selected_evidence_ids"]}
        dropped = tuple(item for item in topk if item.evidence_id not in selected_ids)
        total_dropped += len(dropped)
        harmful_document = counterfactuals[query_id]
        pool_hit = any(item.document_id == harmful_document for item in topk)
        harmful_pool_hits += int(pool_hit)
        harmful_dropped += int(any(item.document_id == harmful_document for item in dropped))
        relevant = set(gold[query_id].relevant_document_ids or ())
        dropped_gold = {item.document_id for item in dropped} & relevant
        gold_document_drops += len(dropped_gold)
        topk_relevant = {item.document_id for item in topk} & relevant
        selected_relevant = {
            topk_by_id[evidence_id].document_id
            for evidence_id in selected_ids
            if evidence_id in topk_by_id
        } & relevant
        recall_loss_queries += int(len(selected_relevant) < len(topk_relevant))
        arms = cast(Mapping[str, Mapping[str, Any]], row["arms"])
        for arm in ARMS:
            arm_row = arms[arm]
            generation = GenerationResult.model_validate(arm_row["generation"])
            result = answer_match(
                strip_annotations(generation.answer), gold[query_id].reference_answers
            ).value
            if result is None:
                raise ValueError(f"no scorable answer for {query_id}")
            values[arm][query_id] = {
                "answer_match": result,
                "coverage": float(bool(generation.answer.strip())),
            }
            errors[arm] += int(bool(arm_row.get("error")))
            notes[arm] += int(bool(arm_row.get("notes")))

    aggregate = {
        arm: {
            metric: sum(item[metric] for item in values[arm].values()) / len(rows)
            for metric in ("answer_match", "coverage")
        }
        for arm in ARMS
    }

    def comparison(on_arm: str, off_arm: str, subset: set[str]) -> dict[str, object]:
        output: dict[str, object] = {}
        for metric in ("answer_match", "coverage"):
            output[metric] = asdict(
                compare_paired(
                    {query_id: values[on_arm][query_id][metric] for query_id in subset},
                    {query_id: values[off_arm][query_id][metric] for query_id in subset},
                    component_ids={query_id: components[query_id] for query_id in subset},
                )
            )
        output["transitions"] = {
            "wrong_to_right": sum(
                values[off_arm][query_id]["answer_match"] == 0.0
                and values[on_arm][query_id]["answer_match"] == 1.0
                for query_id in subset
            ),
            "right_to_wrong": sum(
                values[off_arm][query_id]["answer_match"] == 1.0
                and values[on_arm][query_id]["answer_match"] == 0.0
                for query_id in subset
            ),
        }
        transition = cast(dict[str, int], output["transitions"])
        transition["net"] = transition["wrong_to_right"] - transition["right_to_wrong"]
        return output

    comparisons = {
        "B_minus_A_generator_only": comparison(ARMS[1], ARMS[0], wanted),
        "C_minus_B_selector_only": comparison(ARMS[2], ARMS[1], wanted),
        "C_minus_A_full_system": comparison(ARMS[2], ARMS[0], wanted),
    }
    changed_comparisons: dict[str, object] = {}
    if changed:
        changed_comparisons = {
            "C_minus_B_selector_only": comparison(ARMS[2], ARMS[1], changed),
            "C_minus_A_full_system": comparison(ARMS[2], ARMS[0], changed),
        }
    full = cast(Mapping[str, Any], comparisons["C_minus_A_full_system"])["answer_match"]
    selector = cast(Mapping[str, Any], comparisons["C_minus_B_selector_only"])["answer_match"]
    point_success = full["delta"] > 0.0 and selector["delta"] > 0.0
    statistical_confirmation = (
        point_success and full["ci_low"] > 0.0 and selector["ci_low"] > 0.0
    )
    return {
        "schema_version": "full-flow-f005-report-v1",
        "status": "COMPLETE",
        "queries": len(rows),
        "components": len(set(components.values())),
        "changed_queries": len(changed),
        "gold_loaded_at_runtime": False,
        "aggregate": aggregate,
        "comparisons": comparisons,
        "changed_subset_comparisons": changed_comparisons,
        "errors": errors,
        "notes_nonempty_queries": notes,
        "selector_evidence": {
            "total_dropped": total_dropped,
            "harmful_pool_hits": harmful_pool_hits,
            "harmful_dropped": harmful_dropped,
            "harmful_reduction": (
                harmful_dropped / harmful_pool_hits if harmful_pool_hits else None
            ),
            "deletion_precision": harmful_dropped / total_dropped if total_dropped else None,
            "gold_document_drops": gold_document_drops,
            "queries_with_recall_loss": recall_loss_queries,
        },
        "confirmation": {
            "point_estimate_success": point_success,
            "statistically_confirmed": statistical_confirmation,
            "rule": "C>A and C>B; confirmed only if both 95% CI lower bounds are >0",
        },
    }


def _percent(value: float) -> str:
    return f"{100 * value:.2f}%"


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# F005 独立最终确认结果",
        "",
        "| 组 | Answer match | Coverage |",
        "|---|---:|---:|",
    ]
    for arm in ARMS:
        item = report["aggregate"][arm]
        lines.append(f"| {arm} | {_percent(item['answer_match'])} | {_percent(item['coverage'])} |")
    lines.extend(
        [
            "",
            "| 比较 | Answer delta | 95% CI | 错→对 | 对→错 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for label, comparison in report["comparisons"].items():
        answer = comparison["answer_match"]
        transition = comparison["transitions"]
        lines.append(
            f"| {label} | {100*answer['delta']:+.2f} pp | "
            f"[{100*answer['ci_low']:+.2f}, {100*answer['ci_high']:+.2f}] | "
            f"{transition['wrong_to_right']} | {transition['right_to_wrong']} |"
        )
    evidence = report["selector_evidence"]
    lines.extend(
        [
            "",
            f"- Selector 改变问题：{report['changed_queries']} / {report['queries']}",
            f"- 删除证据：{evidence['total_dropped']} 条",
            f"- 删除已知有害证据：{evidence['harmful_dropped']} / {evidence['harmful_pool_hits']}",
            f"- 删除 gold 相关文档：{evidence['gold_document_drops']} 条",
            "",
            f"**点估计目标：** {'PASS' if report['confirmation']['point_estimate_success'] else 'FAIL'}",
            f"**统计确认：** {'PASS' if report['confirmation']['statistically_confirmed'] else 'NOT CONFIRMED'}",
            "",
        ]
    )
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    choose = commands.add_parser("select")
    choose.add_argument("--queries", required=True, type=Path)
    choose.add_argument("--candidate-pool", required=True, type=Path)
    choose.add_argument("--model-snapshot", required=True, type=Path)
    choose.add_argument("--checkpoint", required=True, type=Path)
    choose.add_argument("--safe-threshold", required=True, type=float)
    choose.add_argument("--cap", required=True, type=int, choices=(1, 2, 3))
    choose.add_argument("--output-dir", required=True, type=Path)
    choose.add_argument("--device", default="cuda:0")
    choose.add_argument("--batch-size", type=int, default=32)
    generate = commands.add_parser("run")
    generate.add_argument("--queries", required=True, type=Path)
    generate.add_argument("--candidate-pool", required=True, type=Path)
    generate.add_argument("--selection-trace", required=True, type=Path)
    generate.add_argument("--model-snapshot", required=True, type=Path)
    generate.add_argument("--true-model-id", required=True)
    generate.add_argument("--mixed-adapter", required=True, type=Path)
    generate.add_argument("--output-dir", required=True, type=Path)
    generate.add_argument("--limit", type=int)
    evaluate = commands.add_parser("score")
    evaluate.add_argument("--generations", required=True, type=Path)
    evaluate.add_argument("--gold", required=True, type=Path)
    evaluate.add_argument("--source-parent", required=True, type=Path)
    evaluate.add_argument("--provenance", required=True, type=Path)
    evaluate.add_argument("--output-json", required=True, type=Path)
    evaluate.add_argument("--output-report", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "select":
        select(
            queries_path=args.queries,
            candidate_pool_path=args.candidate_pool,
            model_snapshot=args.model_snapshot,
            checkpoint=args.checkpoint,
            safe_threshold=args.safe_threshold,
            cap=args.cap,
            output_dir=args.output_dir.resolve(),
            device=args.device,
            batch_size=args.batch_size,
        )
        return 0
    if args.command == "run":
        run(
            queries_path=args.queries,
            candidate_pool_path=args.candidate_pool,
            selection_trace_path=args.selection_trace,
            model_snapshot=args.model_snapshot,
            true_model_id=args.true_model_id,
            mixed_adapter=args.mixed_adapter,
            output_dir=args.output_dir.resolve(),
            limit=args.limit,
        )
        return 0
    rows = tuple(_jsonl(args.generations))
    wanted = {str(row["query_id"]) for row in rows}
    report = score(
        rows,
        _load_gold(args.gold, wanted),
        _parent_map(args.source_parent),
        _provenance(args.provenance),
    )
    _write_json(args.output_json, report)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
