"""B100 controlled-context diagnostic for the frozen Granite Generator.

The command boundary is deliberate:

* ``prepare`` may read gold document labels to build offline oracle contexts;
* ``run`` accepts only the stripped runtime-context artifact and never gold;
* ``score`` joins completed generations to gold and the construction audit.

The six arms are parallel diagnostic calls, not a deployable multi-answer flow.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
import time
import traceback
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import full_flow_joint as joint

from evidence_rag.contracts.models import (
    EvidenceCandidate,
    GenerationResult,
    Query,
    QueryChecklist,
    SelectedEvidenceSet,
    strip_annotations,
)
from evidence_rag.evaluation.paired_metric import compare_paired
from evidence_rag.evaluation.scoring import answer_match
from evidence_rag.generator.claim_splitter import FAITHFULNESS_PROMPT, SPLIT_PROMPT
from evidence_rag.generator.draft import DRAFT_PROMPT
from evidence_rag.generator.granite import GraniteGenerationConfig, GraniteLLMClient
from evidence_rag.generator.nli import TrueNLIModel
from evidence_rag.generator.trace import GeneratorTrace
from evidence_rag.generator.verify_annotate import VerifyAnnotateGenerator
from evidence_rag.infrastructure.datasets import GoldCase

ARMS = (
    "K_topk",
    "S_legacy_selected",
    "O_support_only",
    "OB_support_benign",
    "OH_support_harmful",
    "OP_support_last",
)
PAIR_SPECS = (
    ("S_minus_K", "S_legacy_selected", "K_topk"),
    ("O_minus_K", "O_support_only", "K_topk"),
    ("OB_minus_O", "OB_support_benign", "O_support_only"),
    ("OH_minus_O", "OH_support_harmful", "O_support_only"),
    ("OH_minus_OB", "OH_support_harmful", "OB_support_benign"),
    ("OP_minus_K", "OP_support_last", "K_topk"),
)
SEED = 13
EXPECTED_CHANGED = 109
EXPECTED_MATCHED_UNCHANGED = 109
DEFAULT_MAX_ERROR_RATE = 0.05
REPO_ROOT = Path(__file__).resolve().parents[1]


class TracedGenerator(Protocol):
    last_trace: GeneratorTrace | None

    def generate(
        self,
        query: Query,
        checklist: QueryChecklist,
        selected: SelectedEvidenceSet,
    ) -> GenerationResult: ...


@dataclass(frozen=True, slots=True)
class CaseProfile:
    case: joint.JointCase
    relevant_document_ids: frozenset[str]
    question_type: str
    reference_shape: str
    chain_eligible_topk10: bool
    support: tuple[EvidenceCandidate, ...]
    harmful: tuple[EvidenceCandidate, ...]
    benign: tuple[EvidenceCandidate, ...]

    @property
    def first_support_rank(self) -> int:
        return min((item.retrieval_rank for item in self.support), default=11)

    @property
    def support_bucket(self) -> str:
        if not self.support:
            return "absent"
        if len(self.support) == 1:
            return "single"
        return "multi"

    def matching_row(self) -> dict[str, object]:
        return {
            "question_type": self.question_type,
            "reference_shape": self.reference_shape,
            "chain_eligible_topk10": self.chain_eligible_topk10,
            "support_count": len(self.support),
            "harmful_count": len(self.harmful),
            "benign_count": len(self.benign),
            "support_bucket": self.support_bucket,
            "first_support_rank": self.first_support_rank,
            "harmful_visible": bool(self.harmful),
            "benign_visible": bool(self.benign),
        }


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _jsonl(path: Path) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, Mapping):
            raise ValueError(f"{path}:{line_number} is not an object")
        rows.append(value)
    return rows


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_gold(path: Path, wanted: set[str]) -> dict[str, GoldCase]:
    output: dict[str, GoldCase] = {}
    for value in _jsonl(path):
        gold = GoldCase.model_validate(value)
        if gold.query_id not in wanted:
            continue
        if gold.query_id in output:
            raise ValueError(f"duplicate gold query: {gold.query_id}")
        if not gold.reference_answers or not gold.relevant_document_ids:
            raise ValueError(f"query {gold.query_id} lacks required B100 gold fields")
        output[gold.query_id] = gold
    if set(output) != wanted:
        raise ValueError("gold does not exactly cover B100 source cases")
    return output


def _load_chain_eligibility(path: Path, wanted: set[str]) -> dict[str, bool]:
    output: dict[str, bool] = {}
    for value in _jsonl(path):
        if value.get("role") != "decision-dev":
            continue
        query_id = str(value.get("query_id", ""))
        if query_id not in wanted:
            continue
        flag = value.get("chain_eligible_topk10")
        if not isinstance(flag, bool):
            raise ValueError(f"query {query_id} lacks chain_eligible_topk10")
        output[query_id] = flag
    if set(output) != wanted:
        raise ValueError("role file does not exactly cover B100 source cases")
    return output


def _reference_shape(references: tuple[str, ...]) -> str:
    text = " ".join(references)
    if re.search(r"\b(?:1[0-9]{3}|20[0-9]{2})\b", text):
        return "date_or_year"
    if re.search(r"\d", text):
        return "number"
    months = (
        "january",
        "february",
        "march",
        "april",
        "may ",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
    )
    if any(token in text.casefold() for token in months):
        return "date_or_year"
    if len(text.split()) >= 2:
        return "named_entity_or_phrase"
    return "short_entity"


def _question_type(question: str) -> str:
    normalized = re.sub(r"[^a-z0-9 ]+", " ", question.casefold()).strip()
    if re.search(r"\bhow (?:many|much|long|old|far)\b", normalized):
        return "quantity"
    first = normalized.split(maxsplit=1)[0] if normalized else ""
    if first in {"who", "whom", "whose"}:
        return "who"
    if first == "when":
        return "when"
    if first == "where":
        return "where"
    if first == "which":
        return "which"
    if first in {"why", "how"}:
        return "why_or_how"
    if first == "what":
        return "what"
    return "other"


def _is_harmful(item: EvidenceCandidate) -> bool:
    by_document = item.document_id.startswith("cf::")
    by_source = item.source_uri.startswith("synthetic://cf/")
    if by_document != by_source:
        raise ValueError(f"counterfactual provenance disagrees for {item.evidence_id}")
    return by_document


def build_profile(
    case: joint.JointCase,
    gold: GoldCase,
    *,
    chain_eligible_topk10: bool,
) -> CaseProfile:
    relevant = frozenset(gold.relevant_document_ids or ())
    support: list[EvidenceCandidate] = []
    harmful: list[EvidenceCandidate] = []
    benign: list[EvidenceCandidate] = []
    for item in case.topk10:
        if item.document_id in relevant:
            support.append(item)
        elif _is_harmful(item):
            harmful.append(item)
        else:
            benign.append(item)
    if len(support) + len(harmful) + len(benign) != len(case.topk10):
        raise AssertionError("B100 evidence partition is incomplete")
    return CaseProfile(
        case=case,
        relevant_document_ids=relevant,
        question_type=_question_type(case.question),
        reference_shape=_reference_shape(cast(tuple[str, ...], gold.reference_answers)),
        chain_eligible_topk10=chain_eligible_topk10,
        support=tuple(support),
        harmful=tuple(harmful),
        benign=tuple(benign),
    )


def _matching_cost(changed: CaseProfile, control: CaseProfile) -> int:
    cost = 0
    cost += 1_000_000 * (changed.question_type != control.question_type)
    cost += 300_000 * (changed.reference_shape != control.reference_shape)
    cost += 100_000 * (
        changed.chain_eligible_topk10 != control.chain_eligible_topk10
    )
    cost += 500_000 * (bool(changed.support) != bool(control.support))
    cost += 100_000 * (changed.support_bucket != control.support_bucket)
    cost += 250_000 * (bool(changed.harmful) != bool(control.harmful))
    cost += 100_000 * (bool(changed.benign) != bool(control.benign))
    cost += 1_000 * abs(len(changed.support) - len(control.support))
    cost += 1_000 * abs(len(changed.harmful) - len(control.harmful))
    cost += 200 * abs(len(changed.benign) - len(control.benign))
    cost += 50 * abs(changed.first_support_rank - control.first_support_rank)
    return cost


def _minimum_assignment(costs: Sequence[Sequence[int]]) -> tuple[int, ...]:
    """Rectangular Hungarian assignment for rows <= columns."""

    row_count = len(costs)
    if row_count == 0:
        return ()
    column_count = len(costs[0])
    if row_count > column_count or any(len(row) != column_count for row in costs):
        raise ValueError("assignment matrix must be rectangular with rows <= columns")
    u = [0] * (row_count + 1)
    v = [0] * (column_count + 1)
    p = [0] * (column_count + 1)
    way = [0] * (column_count + 1)
    infinity = 10**18
    for row_index in range(1, row_count + 1):
        p[0] = row_index
        minimum = [infinity] * (column_count + 1)
        used = [False] * (column_count + 1)
        column0 = 0
        while True:
            used[column0] = True
            current_row = p[column0]
            delta = infinity
            column1 = 0
            for column in range(1, column_count + 1):
                if used[column]:
                    continue
                candidate = costs[current_row - 1][column - 1] - u[current_row] - v[column]
                if candidate < minimum[column]:
                    minimum[column] = candidate
                    way[column] = column0
                if minimum[column] < delta:
                    delta = minimum[column]
                    column1 = column
            for column in range(column_count + 1):
                if used[column]:
                    u[p[column]] += delta
                    v[column] -= delta
                else:
                    minimum[column] -= delta
            column0 = column1
            if p[column0] == 0:
                break
        while True:
            column1 = way[column0]
            p[column0] = p[column1]
            column0 = column1
            if column0 == 0:
                break
    assignment = [-1] * row_count
    for column in range(1, column_count + 1):
        if p[column]:
            assignment[p[column] - 1] = column - 1
    if any(column < 0 for column in assignment):
        raise AssertionError("assignment did not cover every changed case")
    return tuple(assignment)


def match_unchanged(
    changed: Sequence[CaseProfile], controls: Sequence[CaseProfile]
) -> tuple[tuple[CaseProfile, CaseProfile, int], ...]:
    changed_ordered = tuple(sorted(changed, key=lambda item: item.case.query_id))
    controls_ordered = tuple(sorted(controls, key=lambda item: item.case.query_id))
    costs = [
        [_matching_cost(treated, control) for control in controls_ordered]
        for treated in changed_ordered
    ]
    assignment = _minimum_assignment(costs)
    return tuple(
        (treated, controls_ordered[column], costs[row][column])
        for row, (treated, column) in enumerate(zip(changed_ordered, assignment, strict=True))
    )


def build_contexts(profile: CaseProfile) -> dict[str, tuple[EvidenceCandidate, ...]]:
    noise_count = min(len(profile.benign), len(profile.harmful))
    support_ids = {item.evidence_id for item in profile.support}
    support_last = tuple(
        item for item in profile.case.topk10 if item.evidence_id not in support_ids
    ) + profile.support
    contexts = {
        "K_topk": profile.case.topk10,
        "S_legacy_selected": profile.case.selected,
        "O_support_only": profile.support,
        "OB_support_benign": profile.support + profile.benign[:noise_count],
        "OH_support_harmful": profile.support + profile.harmful[:noise_count],
        "OP_support_last": support_last,
    }
    if set(item.evidence_id for item in support_last) != set(
        item.evidence_id for item in profile.case.topk10
    ):
        raise AssertionError("O-P must preserve the K evidence set")
    if len(contexts["OB_support_benign"]) != len(contexts["OH_support_harmful"]):
        raise AssertionError("O+B and O+H must have equal context lengths")
    return contexts


def _balance_summary(profiles: Sequence[CaseProfile]) -> dict[str, object]:
    categorical: dict[str, dict[str, int]] = {}
    for key in (
        "question_type",
        "reference_shape",
        "support_bucket",
        "harmful_visible",
        "benign_visible",
        "chain_eligible_topk10",
    ):
        categorical[key] = dict(
            sorted(Counter(str(item.matching_row()[key]) for item in profiles).items())
        )
    numeric = {}
    for key in ("support_count", "harmful_count", "benign_count", "first_support_rank"):
        values = [int(item.matching_row()[key]) for item in profiles]
        numeric[key] = {
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }
    return {"queries": len(profiles), "categorical": categorical, "numeric": numeric}


def prepare_rows(
    profiles: Sequence[CaseProfile],
    *,
    changed_limit: int | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    changed = [item for item in profiles if item.case.selector_changed]
    controls = [item for item in profiles if not item.case.selector_changed]
    if changed_limit is not None:
        if changed_limit <= 0:
            raise ValueError("--changed-limit must be positive")
        changed = sorted(changed, key=lambda item: item.case.query_id)[:changed_limit]
    pairs = match_unchanged(changed, controls)
    runtime_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    for changed_profile, control_profile, match_cost in pairs:
        pair_id = changed_profile.case.query_id
        for cohort, profile in (
            ("selector_changed", changed_profile),
            ("matched_unchanged", control_profile),
        ):
            contexts = build_contexts(profile)
            runtime_rows.append(
                {
                    "schema_version": "full-flow-b100-runtime-context-v1",
                    "query_id": profile.case.query_id,
                    "question": profile.case.question,
                    "component_id": profile.case.component_id,
                    "selector_changed": profile.case.selector_changed,
                    "matched_pair_id": pair_id,
                    "cohort": cohort,
                    "contexts": {
                        arm: [item.model_dump(mode="json") for item in contexts[arm]]
                        for arm in ARMS
                    },
                }
            )
            roles = {
                item.evidence_id: (
                    "support"
                    if item in profile.support
                    else "harmful"
                    if item in profile.harmful
                    else "benign"
                )
                for item in profile.case.topk10
            }
            audit_rows.append(
                {
                    "schema_version": "full-flow-b100-context-audit-v1",
                    "query_id": profile.case.query_id,
                    "matched_pair_id": pair_id,
                    "cohort": cohort,
                    "match_cost": match_cost,
                    "matching_profile": profile.matching_row(),
                    "relevant_document_ids": sorted(profile.relevant_document_ids),
                    "topk_evidence_roles": roles,
                    "noise_pair_count": min(len(profile.benign), len(profile.harmful)),
                    "noise_pair_eligible": bool(profile.benign and profile.harmful),
                    "contexts": {
                        arm: [item.evidence_id for item in contexts[arm]] for arm in ARMS
                    },
                }
            )
    runtime_rows.sort(key=lambda row: str(row["query_id"]))
    audit_rows.sort(key=lambda row: str(row["query_id"]))
    matching = {
        "schema_version": "full-flow-b100-matching-v1",
        "pairs": len(pairs),
        "total_cost": sum(item[2] for item in pairs),
        "exact_question_type": sum(
            left.question_type == right.question_type for left, right, _ in pairs
        ),
        "exact_reference_shape": sum(
            left.reference_shape == right.reference_shape for left, right, _ in pairs
        ),
        "exact_support_bucket": sum(
            left.support_bucket == right.support_bucket for left, right, _ in pairs
        ),
        "changed": _balance_summary([item[0] for item in pairs]),
        "matched_unchanged": _balance_summary([item[1] for item in pairs]),
        "pair_rows": [
            {
                "changed_query_id": left.case.query_id,
                "control_query_id": right.case.query_id,
                "cost": cost,
            }
            for left, right, cost in pairs
        ],
    }
    return runtime_rows, audit_rows, matching


def _stable_value(query_id: str, namespace: str) -> int:
    digest = hashlib.sha256(f"{SEED}:{namespace}:{query_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def arm_order(query_id: str) -> tuple[str, ...]:
    offset = _stable_value(query_id, "b100-arm-order") % len(ARMS)
    ordered = ARMS[offset:] + ARMS[:offset]
    if _stable_value(query_id, "b100-arm-direction") % 2:
        ordered = tuple(reversed(ordered))
    return ordered


def _runtime_contexts(path: Path) -> list[dict[str, object]]:
    expected_keys = {
        "schema_version",
        "query_id",
        "question",
        "component_id",
        "selector_changed",
        "matched_pair_id",
        "cohort",
        "contexts",
    }
    output: list[dict[str, object]] = []
    seen: set[str] = set()
    for raw in _jsonl(path):
        if set(raw) != expected_keys:
            raise ValueError("runtime context row contains missing or forbidden fields")
        if raw["schema_version"] != "full-flow-b100-runtime-context-v1":
            raise ValueError("unexpected B100 runtime context schema")
        query_id = str(raw["query_id"])
        if query_id in seen:
            raise ValueError(f"duplicate B100 runtime query: {query_id}")
        seen.add(query_id)
        raw_contexts = raw["contexts"]
        if not isinstance(raw_contexts, Mapping) or set(raw_contexts) != set(ARMS):
            raise ValueError(f"query {query_id} does not contain exactly the six B100 arms")
        contexts: dict[str, list[dict[str, object]]] = {}
        for arm in ARMS:
            values = raw_contexts[arm]
            if not isinstance(values, list):
                raise ValueError(f"query {query_id}, arm {arm} context is not a list")
            candidates = [EvidenceCandidate.model_validate(value) for value in values]
            contexts[arm] = [item.model_dump(mode="json") for item in candidates]
        output.append({**raw, "contexts": contexts})
    return output


def _run_one(
    generator: TracedGenerator,
    query: Query,
    checklist: QueryChecklist,
    evidence: tuple[EvidenceCandidate, ...],
) -> dict[str, object]:
    started = time.perf_counter()
    try:
        generation = generator.generate(
            query,
            checklist,
            SelectedEvidenceSet(query_id=query.query_id, evidence=evidence),
        )
        trace = generator.last_trace
        if trace is None:
            raise RuntimeError("trace-enabled Generator produced no trace")
    except Exception as error:  # noqa: BLE001 -- failures are measured outcomes
        traceback.print_exc()
        return {
            "generation": GenerationResult(
                query_id=query.query_id,
                answer="",
                cited_evidence_ids=(),
            ).model_dump(mode="json"),
            "trace": None,
            "error": f"{type(error).__name__}: {error}",
            "seconds": time.perf_counter() - started,
        }
    return {
        "generation": generation.model_dump(mode="json"),
        "trace": trace.model_dump(mode="json"),
        "error": None,
        "seconds": time.perf_counter() - started,
    }


def run_contexts(
    rows: Sequence[Mapping[str, object]], *, generator: TracedGenerator
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    started = time.perf_counter()
    for index, row in enumerate(rows, 1):
        query_id = str(row["query_id"])
        query = Query(query_id=query_id, text=str(row["question"]))
        checklist = QueryChecklist(query_id=query_id, focus=query.text, required_facts=())
        raw_contexts = cast(Mapping[str, Sequence[object]], row["contexts"])
        order = arm_order(query_id)
        arms: dict[str, dict[str, object]] = {}
        for arm in order:
            evidence = tuple(
                EvidenceCandidate.model_validate(value) for value in raw_contexts[arm]
            )
            arms[arm] = _run_one(generator, query, checklist, evidence)
        output.append(
            {
                "schema_version": "full-flow-b100-generation-row-v1",
                "query_id": query_id,
                "component_id": row["component_id"],
                "selector_changed": row["selector_changed"],
                "matched_pair_id": row["matched_pair_id"],
                "cohort": row["cohort"],
                "arm_order": list(order),
                "context_evidence_ids": {
                    arm: [str(item["evidence_id"]) for item in raw_contexts[arm]]
                    for arm in ARMS
                },
                "arms": arms,
            }
        )
        if index % 5 == 0 or index == len(rows):
            elapsed = time.perf_counter() - started
            print(
                f"[B100 run] {index}/{len(rows)}; {elapsed / index:.2f}s/case",
                flush=True,
            )
    return output


def _runtime_environment() -> dict[str, object]:
    torch = __import__("torch")
    transformers = __import__("transformers")
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu_names": [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ],
    }


def _generation(value: object) -> GenerationResult:
    if not isinstance(value, Mapping):
        raise ValueError("generation arm is not an object")
    return GenerationResult.model_validate(value["generation"])


def _trace_empty_reason(value: object, generation: GenerationResult) -> str:
    if generation.answer.strip():
        return "nonempty"
    if not isinstance(value, Mapping):
        return "missing_trace"
    trace = value.get("trace")
    if not isinstance(trace, Mapping):
        return "missing_trace"
    reason = str(trace.get("final_empty_reason", "")).strip()
    return reason or "unclassified_empty"


def _trace_details(value: object) -> tuple[str, tuple[str, ...]]:
    if not isinstance(value, Mapping):
        return "missing_trace", ()
    trace = value.get("trace")
    if not isinstance(trace, Mapping):
        return "missing_trace", ()
    draft = trace.get("draft")
    splitter_status = "missing_draft"
    if isinstance(draft, Mapping):
        splitter = draft.get("splitter")
        splitter_status = (
            str(splitter.get("status", "missing_status"))
            if isinstance(splitter, Mapping)
            else "not_run"
        )
    claims = trace.get("claims")
    dispositions = tuple(
        str(claim.get("final_disposition", "missing_disposition"))
        for claim in claims
        if isinstance(claim, Mapping)
    ) if isinstance(claims, list) else ()
    return splitter_status, dispositions


def _normalized_text(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.casefold()).split())


def _reference_text_visible(
    evidence: Sequence[EvidenceCandidate], references: Sequence[str]
) -> bool:
    context = _normalized_text(" ".join(item.text for item in evidence))
    return any(
        normalized and normalized in context
        for reference in references
        if (normalized := _normalized_text(reference))
    )


def _mean(values: Iterable[float | None]) -> float | None:
    scored = [value for value in values if value is not None]
    return None if not scored else sum(scored) / len(scored)


def score_rows(
    generation_rows: Sequence[Mapping[str, Any]],
    runtime_rows: Sequence[Mapping[str, Any]],
    audit_rows: Sequence[Mapping[str, Any]],
    gold: Mapping[str, GoldCase],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    runtime = {str(row["query_id"]): row for row in runtime_rows}
    audit = {str(row["query_id"]): row for row in audit_rows}
    generated = {str(row["query_id"]): row for row in generation_rows}
    if not (set(runtime) == set(audit) == set(generated) == set(gold)):
        raise ValueError("B100 score inputs do not cover exactly the same query IDs")
    cases: list[dict[str, object]] = []
    per_case_metrics: dict[str, dict[str, dict[str, object]]] = {}
    component_ids: dict[str, str] = {}
    cohort: dict[str, str] = {}
    noise_eligible: dict[str, bool] = {}
    support_bucket: dict[str, str] = {}
    errors = dict.fromkeys(ARMS, 0)
    for query_id in sorted(generated):
        row = generated[query_id]
        runtime_row = runtime[query_id]
        audit_row = audit[query_id]
        gold_case = gold[query_id]
        references = cast(tuple[str, ...], gold_case.reference_answers)
        relevant = set(gold_case.relevant_document_ids or ())
        component_ids[query_id] = str(row["component_id"])
        cohort[query_id] = str(row["cohort"])
        noise_eligible[query_id] = bool(audit_row["noise_pair_eligible"])
        profile = cast(Mapping[str, object], audit_row["matching_profile"])
        support_bucket[query_id] = str(profile["support_bucket"])
        raw_contexts = cast(Mapping[str, Sequence[object]], runtime_row["contexts"])
        raw_arms = cast(Mapping[str, Mapping[str, object]], row["arms"])
        metrics_by_arm: dict[str, dict[str, object]] = {}
        for arm in ARMS:
            arm_row = raw_arms[arm]
            generation = _generation(arm_row)
            if arm_row.get("error"):
                errors[arm] += 1
            candidates = tuple(
                EvidenceCandidate.model_validate(value) for value in raw_contexts[arm]
            )
            candidate_by_id = {item.evidence_id: item for item in candidates}
            cited_documents = {
                candidate_by_id[evidence_id].document_id
                for evidence_id in generation.cited_evidence_ids
                if evidence_id in candidate_by_id
            }
            visible_support = {item.document_id for item in candidates} & relevant
            matched = answer_match(strip_annotations(generation.answer), references).value
            if matched is None:
                raise ValueError(f"query {query_id} has unscored answer match")
            citation_precision = (
                None
                if not cited_documents
                else len(cited_documents & relevant) / len(cited_documents)
            )
            citation_recall = (
                None
                if not visible_support
                else len(cited_documents & visible_support) / len(visible_support)
            )
            reference_visible = _reference_text_visible(candidates, references)
            splitter_status, claim_dispositions = _trace_details(arm_row)
            metrics_by_arm[arm] = {
                "answer": generation.answer,
                "cited_evidence_ids": list(generation.cited_evidence_ids),
                "answer_match": float(matched),
                "coverage": float(bool(generation.answer.strip())),
                "gold_document_citation_precision": citation_precision,
                "visible_support_citation_recall": citation_recall,
                "visible_support_documents": len(visible_support),
                "support_present_but_wrong": bool(visible_support and matched == 0.0),
                "reference_text_visible": reference_visible,
                "reference_text_visible_but_wrong": bool(
                    reference_visible and matched == 0.0
                ),
                "final_empty_reason": _trace_empty_reason(arm_row, generation),
                "splitter_status": splitter_status,
                "claim_dispositions": list(claim_dispositions),
                "seconds": float(arm_row["seconds"]),
                "error": arm_row.get("error"),
            }
        per_case_metrics[query_id] = metrics_by_arm
        oracle_correct = metrics_by_arm["O_support_only"]["answer_match"] == 1.0
        cases.append(
            {
                "schema_version": "full-flow-b100-case-v1",
                "query_id": query_id,
                "question": runtime_row["question"],
                "component_id": row["component_id"],
                "matched_pair_id": row["matched_pair_id"],
                "cohort": row["cohort"],
                "reference_answers": list(references),
                "matching_profile": profile,
                "noise_pair_eligible": noise_eligible[query_id],
                "topk_evidence_roles": audit_row["topk_evidence_roles"],
                "metrics": metrics_by_arm,
                "observations": {
                    "retriever_support_absent": support_bucket[query_id] == "absent",
                    "support_only_failed": not oracle_correct,
                    "benign_noise_correct_to_wrong": bool(
                        oracle_correct
                        and metrics_by_arm["OB_support_benign"]["answer_match"] == 0.0
                    ),
                    "harmful_noise_correct_to_wrong": bool(
                        oracle_correct
                        and metrics_by_arm["OH_support_harmful"]["answer_match"] == 0.0
                    ),
                    "support_last_correct_to_wrong": bool(
                        metrics_by_arm["K_topk"]["answer_match"] == 1.0
                        and metrics_by_arm["OP_support_last"]["answer_match"] == 0.0
                    ),
                    "selector_wrong_when_K_right": bool(
                        metrics_by_arm["K_topk"]["answer_match"] == 1.0
                        and metrics_by_arm["S_legacy_selected"]["answer_match"] == 0.0
                    ),
                },
            }
        )

    def scope_report(query_ids: set[str]) -> dict[str, object]:
        if not query_ids:
            return {
                "queries": 0,
                "aggregate": {
                    arm: {
                        "answer_match": None,
                        "coverage": None,
                        "gold_document_citation_precision": None,
                        "visible_support_citation_recall": None,
                        "support_present_but_wrong": 0,
                        "reference_text_visible": 0,
                        "reference_text_visible_but_wrong": 0,
                        "final_empty_reasons": {},
                        "splitter_status": {},
                        "claim_dispositions": {},
                        "seconds_total": 0.0,
                        "seconds_mean": None,
                    }
                    for arm in ARMS
                },
                "comparisons": {
                    label: {
                        "queries": 0,
                        "paired": None,
                        "wrong_to_right": 0,
                        "right_to_wrong": 0,
                        "exact_answer_text_equal": 0,
                        "answer_metric_equal": 0,
                    }
                    for label, _, _ in PAIR_SPECS
                },
            }
        aggregate: dict[str, object] = {}
        for arm in ARMS:
            items = [per_case_metrics[query_id][arm] for query_id in sorted(query_ids)]
            aggregate[arm] = {
                "answer_match": _mean(cast(float, item["answer_match"]) for item in items),
                "coverage": _mean(cast(float, item["coverage"]) for item in items),
                "gold_document_citation_precision": _mean(
                    cast(float | None, item["gold_document_citation_precision"])
                    for item in items
                ),
                "visible_support_citation_recall": _mean(
                    cast(float | None, item["visible_support_citation_recall"])
                    for item in items
                ),
                "support_present_but_wrong": sum(
                    bool(item["support_present_but_wrong"]) for item in items
                ),
                "reference_text_visible": sum(
                    bool(item["reference_text_visible"]) for item in items
                ),
                "reference_text_visible_but_wrong": sum(
                    bool(item["reference_text_visible_but_wrong"]) for item in items
                ),
                "final_empty_reasons": dict(
                    sorted(Counter(str(item["final_empty_reason"]) for item in items).items())
                ),
                "splitter_status": dict(
                    sorted(Counter(str(item["splitter_status"]) for item in items).items())
                ),
                "claim_dispositions": dict(
                    sorted(
                        Counter(
                            str(disposition)
                            for item in items
                            for disposition in cast(
                                Sequence[object], item["claim_dispositions"]
                            )
                        ).items()
                    )
                ),
                "seconds_total": sum(cast(float, item["seconds"]) for item in items),
                "seconds_mean": _mean(cast(float, item["seconds"]) for item in items),
            }
        comparisons: dict[str, object] = {}
        for label, after, before in PAIR_SPECS:
            eligible_ids = set(query_ids)
            if label == "OH_minus_OB":
                eligible_ids = {
                    query_id for query_id in eligible_ids if noise_eligible[query_id]
                }
            after_scores = {
                query_id: cast(float, per_case_metrics[query_id][after]["answer_match"])
                for query_id in eligible_ids
            }
            before_scores = {
                query_id: cast(float, per_case_metrics[query_id][before]["answer_match"])
                for query_id in eligible_ids
            }
            components = {
                query_id: component_ids[query_id] for query_id in eligible_ids
            }
            comparison = (
                asdict(compare_paired(after_scores, before_scores, component_ids=components))
                if eligible_ids
                else None
            )
            wrong_to_right = sum(
                before_scores[query_id] == 0.0 and after_scores[query_id] == 1.0
                for query_id in eligible_ids
            )
            right_to_wrong = sum(
                before_scores[query_id] == 1.0 and after_scores[query_id] == 0.0
                for query_id in eligible_ids
            )
            raw_equal = sum(
                per_case_metrics[query_id][after]["answer"]
                == per_case_metrics[query_id][before]["answer"]
                for query_id in eligible_ids
            )
            comparisons[label] = {
                "queries": len(eligible_ids),
                "paired": comparison,
                "wrong_to_right": wrong_to_right,
                "right_to_wrong": right_to_wrong,
                "exact_answer_text_equal": raw_equal,
                "answer_metric_equal": sum(
                    after_scores[query_id] == before_scores[query_id]
                    for query_id in eligible_ids
                ),
            }
        return {
            "queries": len(query_ids),
            "aggregate": aggregate,
            "comparisons": comparisons,
        }

    all_ids = set(generated)
    report = {
        "schema_version": "full-flow-b100-report-v1",
        "status": "COMPLETE",
        "queries": len(all_ids),
        "errors_by_arm": errors,
        "all": scope_report(all_ids),
        "selector_changed": scope_report(
            {query_id for query_id in all_ids if cohort[query_id] == "selector_changed"}
        ),
        "matched_unchanged": scope_report(
            {query_id for query_id in all_ids if cohort[query_id] == "matched_unchanged"}
        ),
        "single_support": scope_report(
            {query_id for query_id in all_ids if support_bucket[query_id] == "single"}
        ),
        "multi_support": scope_report(
            {query_id for query_id in all_ids if support_bucket[query_id] == "multi"}
        ),
        "noise_pair_eligible_queries": sum(noise_eligible.values()),
        "metric_notes": {
            "answer_match": "post-generation normalized containment against reference answers",
            "gold_document_citation_precision": (
                "post-generation cited-document overlap with official relevant documents"
            ),
            "visible_support_citation_recall": (
                "post-generation fraction of relevant documents visible in this arm that were cited"
            ),
            "reference_text_visible": (
                "conservative normalized exact reference-string containment in this arm's evidence"
            ),
        },
    }
    return cases, report


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{100 * value:.2f}%"


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# B100 Controlled Context Diagnostic",
        "",
        f"**Status:** `{report['status']}`",
        "",
        "## All matched diagnostic queries",
        "",
        "| Arm | Answer | Coverage | Gold-doc citation precision | Visible-support citation recall | Reference visible but wrong |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    scope = report["all"]
    for arm in ARMS:
        item = scope["aggregate"][arm]
        lines.append(
            f"| {arm} | {_pct(item['answer_match'])} | {_pct(item['coverage'])} | "
            f"{_pct(item['gold_document_citation_precision'])} | "
            f"{_pct(item['visible_support_citation_recall'])} | "
            f"{item['reference_text_visible_but_wrong']} |"
        )
    lines.extend(
        [
            "",
            "## Paired answer comparisons",
            "",
            "| Comparison | n | Delta | 95% CI | Wrong to right | Right to wrong | Exact text equal |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for label, _, _ in PAIR_SPECS:
        item = scope["comparisons"][label]
        paired = item["paired"]
        lines.append(
            f"| {label} | {item['queries']} | {_pct(paired['delta'])} | "
            f"[{_pct(paired['ci_low'])}, {_pct(paired['ci_high'])}] | "
            f"{item['wrong_to_right']} | {item['right_to_wrong']} | "
            f"{item['exact_answer_text_equal']}/{item['queries']} |"
        )
    lines.extend(
        [
            "",
            "This is an offline diagnostic matrix. Oracle arms are not deployable, and the six",
            "answers are parallel experiment outputs rather than repeated answers in the formal system.",
            "",
        ]
    )
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--queries", required=True, type=Path)
    prepare.add_argument("--candidate-pool", required=True, type=Path)
    prepare.add_argument("--roles", required=True, type=Path)
    prepare.add_argument("--decision-trace", required=True, type=Path)
    prepare.add_argument("--gold", required=True, type=Path)
    prepare.add_argument("--output-dir", required=True, type=Path)
    prepare.add_argument("--changed-limit", type=int)
    run = commands.add_parser("run")
    run.add_argument("--contexts", required=True, type=Path)
    run.add_argument("--granite-snapshot", required=True, type=Path)
    run.add_argument("--true-snapshot", required=True, type=Path)
    run.add_argument("--output-dir", required=True, type=Path)
    run.add_argument("--max-error-rate", type=float, default=DEFAULT_MAX_ERROR_RATE)
    score = commands.add_parser("score")
    score.add_argument("--contexts", required=True, type=Path)
    score.add_argument("--audit", required=True, type=Path)
    score.add_argument("--generations", required=True, type=Path)
    score.add_argument("--gold", required=True, type=Path)
    score.add_argument("--output-cases", required=True, type=Path)
    score.add_argument("--output-json", required=True, type=Path)
    score.add_argument("--output-report", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        output_dir = args.output_dir.resolve()
        if output_dir.exists() and any(output_dir.iterdir()):
            raise ValueError(f"output directory must be absent or empty: {output_dir}")
        cases = joint.load_runtime_cases(
            queries_path=args.queries,
            candidate_pool_path=args.candidate_pool,
            roles_path=args.roles,
            decision_trace_path=args.decision_trace,
        )
        wanted = {case.query_id for case in cases}
        gold = _load_gold(args.gold, wanted)
        chain = _load_chain_eligibility(args.roles, wanted)
        profiles = [
            build_profile(
                case,
                gold[case.query_id],
                chain_eligible_topk10=chain[case.query_id],
            )
            for case in cases
        ]
        runtime_rows, audit_rows, matching = prepare_rows(
            profiles, changed_limit=args.changed_limit
        )
        if args.changed_limit is None:
            changed_count = sum(row["selector_changed"] is True for row in runtime_rows)
            control_count = sum(row["selector_changed"] is False for row in runtime_rows)
            if (changed_count, control_count) != (
                EXPECTED_CHANGED,
                EXPECTED_MATCHED_UNCHANGED,
            ):
                raise ValueError("formal B100 sample counts differ from the frozen protocol")
        contexts_path = output_dir / "runtime_contexts.jsonl"
        audit_path = output_dir / "context_audit.jsonl"
        _write_jsonl(contexts_path, runtime_rows)
        _write_jsonl(audit_path, audit_rows)
        manifest = {
            "schema_version": "full-flow-b100-prepare-manifest-v1",
            "status": "COMPLETE",
            "git_commit": _git_commit(),
            "queries": len(runtime_rows),
            "selector_changed_queries": sum(
                row["selector_changed"] is True for row in runtime_rows
            ),
            "matched_unchanged_queries": sum(
                row["selector_changed"] is False for row in runtime_rows
            ),
            "gold_used_for_offline_context_construction": True,
            "reference_answers_persisted_to_runtime": False,
            "runtime_gold_fields_present": False,
            "source_sha256": {
                "queries": _sha256_file(args.queries),
                "candidate_pool": _sha256_file(args.candidate_pool),
                "roles": _sha256_file(args.roles),
                "decision_trace": _sha256_file(args.decision_trace),
                "gold": _sha256_file(args.gold),
            },
            "runtime_contexts_sha256": _sha256_file(contexts_path),
            "context_audit_sha256": _sha256_file(audit_path),
            "matching": matching,
        }
        _write_json(output_dir / "prepare_manifest.json", manifest)
        print(json.dumps(manifest, ensure_ascii=True, sort_keys=True))
        return 0
    if args.command == "run":
        if not 0.0 <= args.max_error_rate <= 1.0:
            raise ValueError("--max-error-rate must be in [0, 1]")
        output_dir = args.output_dir.resolve()
        generation_path = output_dir / "generations.jsonl"
        manifest_path = output_dir / "run_manifest.json"
        if generation_path.exists() or manifest_path.exists():
            raise ValueError("B100 run outputs already exist")
        runtime_rows = _runtime_contexts(args.contexts)
        llm = GraniteLLMClient(
            model_id=str(args.granite_snapshot.resolve()),
            config=GraniteGenerationConfig(
                max_new_tokens=256,
                temperature=0.0,
                top_p=1.0,
            ),
        )
        nli = TrueNLIModel(model_id=str(args.true_snapshot.resolve()))
        generator = VerifyAnnotateGenerator(
            llm=llm,
            nli=nli,
            entity_gate="observe",
            abstain_when_unverified=False,
            trace_enabled=True,
        )
        generated = run_contexts(runtime_rows, generator=generator)
        _write_jsonl(generation_path, generated)
        attempts = len(generated)
        errors = {
            arm: sum(bool(row["arms"][arm].get("error")) for row in generated)
            for arm in ARMS
        }
        trace_missing = {
            arm: sum(row["arms"][arm].get("trace") is None for row in generated)
            for arm in ARMS
        }
        manifest = {
            "schema_version": "full-flow-b100-run-manifest-v1",
            "status": "COMPLETE",
            "script_sha256": _sha256_file(Path(__file__)),
            "git_commit": _git_commit(),
            "queries": len(generated),
            "attempts_by_arm": dict.fromkeys(ARMS, attempts),
            "errors_by_arm": errors,
            "trace_missing_by_arm": trace_missing,
            "gold_loaded_at_runtime": False,
            "runtime_contexts_sha256": _sha256_file(args.contexts),
            "prompt_sha256": {
                "draft": _sha256_bytes(DRAFT_PROMPT.encode()),
                "split": _sha256_bytes(SPLIT_PROMPT.encode()),
                "faithfulness": _sha256_bytes(FAITHFULNESS_PROMPT.encode()),
            },
            "decode": {
                "max_new_tokens": 256,
                "temperature": 0.0,
                "top_p": 1.0,
                "do_sample": False,
            },
            "arm_order": "seed-13 stable cyclic offset and direction over six arms",
            "same_process": True,
            "shared_granite_and_true_instances": True,
            "environment": _runtime_environment(),
            "generations_sha256": _sha256_file(generation_path),
        }
        _write_json(manifest_path, manifest)
        broken = {
            arm: errors[arm] / attempts
            for arm in ARMS
            if attempts and errors[arm] / attempts > args.max_error_rate
        }
        if broken or any(trace_missing.values()):
            _write_json(
                output_dir / "INVALID_RUNTIME.json",
                {"excess_errors": broken, "trace_missing": trace_missing},
            )
            return 1
        print(json.dumps(manifest, ensure_ascii=True, sort_keys=True))
        return 0
    if args.command == "score":
        runtime_rows = _runtime_contexts(args.contexts)
        generation_rows = _jsonl(args.generations)
        audit_rows = _jsonl(args.audit)
        wanted = {str(row["query_id"]) for row in runtime_rows}
        gold = _load_gold(args.gold, wanted)
        cases, report = score_rows(generation_rows, runtime_rows, audit_rows, gold)
        report["generations_sha256"] = _sha256_file(args.generations)
        report["runtime_contexts_sha256"] = _sha256_file(args.contexts)
        report["context_audit_sha256"] = _sha256_file(args.audit)
        report["gold_sha256"] = _sha256_file(args.gold)
        report["scoring_git_commit"] = _git_commit()
        report["scoring_script_sha256"] = _sha256_file(Path(__file__))
        _write_jsonl(args.output_cases, cases)
        _write_json(args.output_json, report)
        args.output_report.parent.mkdir(parents=True, exist_ok=True)
        args.output_report.write_text(_markdown(report), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=True, sort_keys=True))
        return 0
    raise AssertionError(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
