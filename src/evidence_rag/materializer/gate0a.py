"""Gate 0A: the data-side exit criterion for M0 (§6 checklist, axes frozen in §4).

Gate 0B accepts a MODEL; Gate 0A accepts the DATA the model will be judged on. Both are written
the same way and for the same reason: a checklist in a document is not a mechanism, and this
project has now written that sentence four times (M0 §9.11, §10.8, §11.9a).

WHAT MAKES THIS AUDIT DIFFERENT FROM A SET COMPARISON. Every axis here is a "zero overlap"
claim, and zero is exactly what a broken input produces. An existing split read without its
mutation log contributes no synthetic families; a gold passage whose article title will not
parse becomes its own parent and cannot collide with anything; an audit run against no existing
split at all reports five clean axes. All three are green boards produced by an audit that could
not look, which is the failure mode this codebase keeps paying for. So each axis asserts that
BOTH sides are non-empty before it believes an empty intersection — the same line
`relations/gate0b.py` takes on a class that is absent from an evaluation set: score it 0 and
FAIL, because that is a split error, not "nothing to measure".

THE VERDICT HAS THREE VALUES, NOT TWO. `INCOMPLETE` exists because §6 is a zero-violation
checklist and a run that never evaluated an item has not passed it. The candidate-window check
cannot run before retrieval has, so the honest reading of a freshly built set is INCOMPLETE, not
PASS-with-a-footnote. `not_applicable` is the separate case where an item has no producer at
all under D1=A — it passes, but the reason travels with the report instead of disappearing.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

from evidence_rag.contracts.models import CandidateSet
from evidence_rag.infrastructure.datasets import DatasetBundle
from evidence_rag.materializer.injector import alias_occurrences
from evidence_rag.materializer.provenance import MutationRecord
from evidence_rag.materializer.sealed600 import (
    PROTOCOL_VERSION,
    SEALED_MANIFEST_FILE,
    SealedManifest,
    SplitFingerprint,
    required_pool_size,
    verify_artifacts,
)
from evidence_rag.selector.answer_norm import canonicalize_answer

CheckStatus = Literal["pass", "fail", "not_applicable", "unevaluated"]
Verdict = Literal["PASS", "FAIL", "INCOMPLETE"]

# M0 §6: the only two admissible sources of a primary label. The mapping is asserted, not merely
# read, so a manifest that forgets to declare one of its label files fails rather than passing
# on the strength of the file it did declare.
EXPECTED_LABEL_PROVENANCE = {
    "gold_cases.jsonl": "official",
    "provenance.jsonl": "deterministic_rule",
}

_EXAMPLES = 5


@dataclass(frozen=True)
class AxisReport:
    axis: str
    n_sealed: int
    n_existing: int
    n_overlap: int
    examples: tuple[str, ...]
    failure: str | None


@dataclass(frozen=True)
class CheckReport:
    name: str
    status: CheckStatus
    detail: str


@dataclass(frozen=True)
class Gate0AReport:
    axes: tuple[AxisReport, ...]
    checks: tuple[CheckReport, ...]

    @property
    def failures(self) -> tuple[str, ...]:
        return tuple(
            [axis.axis for axis in self.axes if axis.failure is not None]
            + [check.name for check in self.checks if check.status == "fail"]
        )

    @property
    def unevaluated(self) -> tuple[str, ...]:
        return tuple(check.name for check in self.checks if check.status == "unevaluated")

    @property
    def verdict(self) -> Verdict:
        if self.failures:
            return "FAIL"
        if self.unevaluated:
            return "INCOMPLETE"
        return "PASS"

    @property
    def passes(self) -> bool:
        return self.verdict == "PASS"

    def payload(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "failures": list(self.failures),
            "unevaluated": list(self.unevaluated),
            "not_applicable": [
                {"name": check.name, "reason": check.detail}
                for check in self.checks
                if check.status == "not_applicable"
            ],
            "axes": [
                {
                    "axis": axis.axis,
                    "n_sealed": axis.n_sealed,
                    "n_existing": axis.n_existing,
                    "n_overlap": axis.n_overlap,
                    "examples": list(axis.examples),
                    "failure": axis.failure,
                }
                for axis in self.axes
            ],
            "checks": [
                {"name": check.name, "status": check.status, "detail": check.detail}
                for check in self.checks
            ],
        }


def unevaluated(name: str, reason: str) -> CheckReport:
    """An item nothing looked at. Downgrades the verdict to INCOMPLETE."""
    return CheckReport(name=name, status="unevaluated", detail=reason)


def not_applicable(name: str, reason: str) -> CheckReport:
    """An item with no producer under the current protocol. Passes, but says why in the report."""
    return CheckReport(name=name, status="not_applicable", detail=reason)


def gate0a_report(
    *, axes: Sequence[AxisReport], checks: Sequence[CheckReport]
) -> Gate0AReport:
    return Gate0AReport(axes=tuple(axes), checks=tuple(checks))


def _axis(
    name: str,
    sealed: frozenset[str],
    existing: frozenset[str],
    *,
    also_sealed: frozenset[str] | None = None,
    also_existing: frozenset[str] | None = None,
) -> AxisReport:
    """One axis, with the vacuity guard in front of the intersection.

    `also_*` carries the second value of the query axis (M0 §4 asks for "query_id 与规范化 query
    文本双重零重叠"), so both halves are subject to the same emptiness check: an id-only
    comparison would pass a question re-asked under a new id, and a text-only comparison would
    pass an id reused for a rephrased question.
    """
    pairs = [(sealed, existing)]
    if also_sealed is not None and also_existing is not None:
        pairs.append((also_sealed, also_existing))
    n_sealed = sum(len(left) for left, _ in pairs)
    n_existing = sum(len(right) for _, right in pairs)
    for left, right in pairs:
        if not left or not right:
            return AxisReport(
                axis=name,
                n_sealed=n_sealed,
                n_existing=n_existing,
                n_overlap=0,
                examples=(),
                failure=(
                    "vacuous: one side of this axis is empty, so a zero intersection says the "
                    "audit could not look, not that the sets are disjoint"
                ),
            )
    overlap = sorted({value for left, right in pairs for value in left & right})
    return AxisReport(
        axis=name,
        n_sealed=n_sealed,
        n_existing=n_existing,
        n_overlap=len(overlap),
        examples=tuple(overlap[:_EXAMPLES]),
        failure=f"{len(overlap)} shared value(s) with an existing split" if overlap else None,
    )


def audit_axes(
    sealed: SplitFingerprint, existing: Sequence[SplitFingerprint]
) -> tuple[AxisReport, ...]:
    """The five axes of M0 §4, in the order the protocol table lists them.

    Existing splits are UNIONED rather than checked one at a time: the claim being audited is
    "this query/page/entity/passage/family appears in no prior split", and a per-split loop that
    stopped at the first clean split would answer a different question.
    """
    if not existing:
        raise ValueError(
            "audit_axes needs at least one existing split: five axes compared against nothing "
            "return five zero intersections, which is a green board produced by an empty input "
            "rather than by a clean sealed set (M0 §4)."
        )

    def union(field: str) -> frozenset[str]:
        values: set[str] = set()
        for split in existing:
            values |= getattr(split, field)
        return frozenset(values)

    return (
        _axis(
            "query",
            sealed.query_ids,
            union("query_ids"),
            also_sealed=sealed.query_texts,
            also_existing=union("query_texts"),
        ),
        _axis("parent_page", sealed.parent_pages, union("parent_pages")),
        _axis("answer_entity", sealed.answer_entities, union("answer_entities")),
        _axis("passage_hash", sealed.passage_hashes, union("passage_hashes")),
        _axis("synthetic_family", sealed.synthetic_families, union("synthetic_families")),
    )


def _check(name: str, problems: Sequence[str], detail: str) -> CheckReport:
    if problems:
        return CheckReport(
            name=name,
            status="fail",
            detail="; ".join(problems[:_EXAMPLES])
            + (f" (+{len(problems) - _EXAMPLES} more)" if len(problems) > _EXAMPLES else ""),
        )
    return CheckReport(name=name, status="pass", detail=detail)


def audit_size_derivation(manifest: SealedManifest) -> CheckReport:
    """Recompute M0 §4's arithmetic from the manifest's OWN recorded inputs.

    The builder checks this before it starts; this checks the artefact afterwards. They are not
    redundant — the audit is the only one of the two that still works on a manifest written by
    an older build, a patched build, or by hand.
    """
    problems: list[str] = []
    measured = 1 - manifest.dev_sample_injected / manifest.dev_sample_queries
    if abs(measured - manifest.dev_skip_rate) >= 0.001:
        problems.append(
            f"recorded skip rate {manifest.dev_skip_rate} is not 1 - "
            f"{manifest.dev_sample_injected}/{manifest.dev_sample_queries} = {measured:.4f}"
        )
    required = required_pool_size(
        target=manifest.target_injected, skip_rate=manifest.dev_skip_rate
    )
    if manifest.required_pool_size != required:
        problems.append(
            f"recorded required pool {manifest.required_pool_size} != derived {required}"
        )
    if manifest.pool_size < required:
        problems.append(f"pool {manifest.pool_size} has no headroom over required {required}")
    if len(manifest.pool_query_ids) != manifest.pool_size:
        problems.append(
            f"pool list holds {len(manifest.pool_query_ids)} ids, pool_size says "
            f"{manifest.pool_size}"
        )
    if len(manifest.query_ids) != manifest.n_injected:
        problems.append(
            f"sealed list holds {len(manifest.query_ids)} ids, n_injected says "
            f"{manifest.n_injected}"
        )
    if manifest.n_injected != manifest.target_injected:
        problems.append(
            f"n_injected {manifest.n_injected} != target {manifest.target_injected}: the sealed "
            "set is not the size the protocol froze"
        )
    if len(set(manifest.query_ids)) != len(manifest.query_ids):
        problems.append("the sealed query list contains duplicates")
    return _check(
        "size_derivation",
        problems,
        f"{manifest.n_injected} sealed from a pool of {manifest.pool_size} "
        f"(required {required} at skip rate {manifest.dev_skip_rate})",
    )


def audit_manifest_freeze(
    directory: Path, manifest: SealedManifest, protocol_document_sha256: str
) -> CheckReport:
    """Every recorded artefact still hashes to its frozen value, and nothing rode along unrecorded.

    The second half matters as much as the first: hashing only what the manifest lists would let
    a file added after the freeze sit in the sealed directory unaudited, and downstream tooling
    reads the directory, not the manifest's file list.
    """
    directory = Path(directory)
    problems = list(verify_artifacts(directory, manifest))
    if manifest.protocol_version != PROTOCOL_VERSION:
        problems.append(
            f"manifest was frozen under {manifest.protocol_version}, this code implements "
            f"{PROTOCOL_VERSION}"
        )
    if manifest.protocol_document_sha256 != protocol_document_sha256:
        problems.append(
            "M0_PROTOCOL_FREEZE.md has changed since the freeze: recorded "
            f"{manifest.protocol_document_sha256[:12]}, now {protocol_document_sha256[:12]}. The "
            "sealed set was frozen against one text of the protocol; auditing it against another "
            "is auditing it against rules that moved afterwards (M0 §0)."
        )
    recorded = set(manifest.artifact_sha256)
    present = {
        path.name
        for path in directory.iterdir()
        if path.is_file() and path.name != SEALED_MANIFEST_FILE
    }
    for name in sorted(present - recorded):
        problems.append(f"{name}: present in the sealed directory but not in the freeze record")
    return _check(
        "manifest_freeze", problems, f"{len(recorded)} artefacts match their frozen hashes"
    )


def audit_label_provenance(
    manifest: SealedManifest, bundle: DatasetBundle, records: Sequence[MutationRecord]
) -> CheckReport:
    """M0 §6: every primary label is `official` or `deterministic_rule`, and every query has one.

    The manifest's closed-set field already makes a third provenance unwritable. What is checked
    here is the other half: that the two declared files actually CARRY a label for every sealed
    query. A query with no gold answer has no primary label at all, and "none" is not one of the
    two admissible sources.
    """
    problems: list[str] = []
    if dict(manifest.label_provenance) != EXPECTED_LABEL_PROVENANCE:
        problems.append(
            f"declared label provenance {dict(manifest.label_provenance)} != the two required "
            f"entries {EXPECTED_LABEL_PROVENANCE}"
        )
    answered = {
        gold_case.query_id for gold_case in bundle.gold_cases if gold_case.reference_answers
    }
    mutated = {record.query_id for record in records}
    for query in bundle.queries:
        if query.query_id not in answered:
            problems.append(f"{query.query_id}: no official gold answer (qrels label missing)")
        if query.query_id not in mutated:
            problems.append(f"{query.query_id}: no mutation record (deterministic label missing)")
    return _check(
        "label_provenance",
        problems,
        f"{len(answered)} official gold labels, {len(mutated)} deterministic mutation labels",
    )


def audit_counterfactuals(
    bundle: DatasetBundle, records: Sequence[MutationRecord]
) -> CheckReport:
    """M0 §6: the mutation is reversible from the log alone and leaves no gold alias behind.

    Both halves are load-bearing for the harm metric. If the twin does not reverse to its needle,
    it is no longer "the same passage with one fact changed" and the harmful label stops meaning
    what the experiment says it means; if a gold alias survives in the twin, a selector that
    keeps that passage is scored harmful for holding text that is in fact correct.
    """
    problems: list[str] = []
    documents = {document.document_id: document for document in bundle.documents}
    aliases_by_query = {
        gold_case.query_id: tuple(gold_case.reference_answers or ())
        for gold_case in bundle.gold_cases
    }
    sealed_queries = {query.query_id for query in bundle.queries}
    logged_queries = {record.query_id for record in records}
    for query_id in sorted(sealed_queries - logged_queries):
        problems.append(f"{query_id}: sealed but absent from the mutation log")
    for query_id in sorted(logged_queries - sealed_queries):
        problems.append(f"{query_id}: in the mutation log but not in the sealed queries")

    for record in records:
        needle = documents.get(record.needle_document_id)
        twin = documents.get(record.counterfactual_document_id)
        if needle is None or twin is None:
            problems.append(f"{record.query_id}: needle or twin missing from documents.jsonl")
            continue
        if sha256(needle.text.encode("utf-8")).hexdigest() != record.text_hash_before:
            problems.append(f"{record.query_id}: needle text does not match text_hash_before")
        if sha256(twin.text.encode("utf-8")).hexdigest() != record.text_hash_after:
            problems.append(f"{record.query_id}: twin text does not match text_hash_after")
        start, _end = record.char_span
        restored = (
            twin.text[:start]
            + record.gold_alias_used
            + twin.text[start + len(record.replacement_value) :]
        )
        if restored != needle.text:
            problems.append(f"{record.query_id}: twin does not reverse to its needle")
        for alias in aliases_by_query.get(record.query_id, ()):
            if alias_occurrences(twin.text, alias):
                problems.append(f"{record.query_id}: residual gold alias {alias!r} in the twin")
        blocked = {canonicalize_answer(alias) for alias in aliases_by_query.get(record.query_id, ())}
        if canonicalize_answer(record.replacement_value) in blocked:
            problems.append(
                f"{record.query_id}: replacement {record.replacement_value!r} is canonically a "
                "gold alias, so the twin asserts the true answer"
            )

    logged_twins = {record.counterfactual_document_id for record in records}
    for document_id in sorted(documents):
        if document_id.startswith("cf::") and document_id not in logged_twins:
            problems.append(f"{document_id}: synthetic document with no mutation record")
    return _check(
        "counterfactual_reversibility",
        problems,
        f"{len(records)} mutations reverse exactly and leave no gold alias",
    )


def audit_candidates(
    bundle: DatasetBundle,
    records: Sequence[MutationRecord],
    candidate_sets: Sequence[CandidateSet],
    *,
    top_n: int,
) -> CheckReport:
    """M0 §6 item 3: per-question candidate count, Top-N ids and derived flags consistent.

    A short window is the silent one. Nothing downstream reports how many candidates a query
    had, so a query retrieved with 14 passages instead of 20 changes its own recall and harmful
    denominators and reads as an ordinary data point.

    Twins from OTHER queries are counted and reported, not failed: every twin lives in the one
    shared corpus, so BM25 can legitimately return query 2's twin for query 1. The harmful label
    keys on the query's own twin, and this count exists so that pool composition is a number in
    the report rather than an argument later.
    """
    problems: list[str] = []
    document_ids = {document.document_id for document in bundle.documents}
    own_twin = {record.query_id: record.counterfactual_document_id for record in records}
    sealed_queries = {query.query_id for query in bundle.queries}
    windows = {candidate_set.query_id: candidate_set for candidate_set in candidate_sets}
    if len(windows) != len(candidate_sets):
        problems.append("two candidate sets share a query id")
    for query_id in sorted(sealed_queries - set(windows)):
        problems.append(f"{query_id}: sealed but has no candidate window")
    for query_id in sorted(set(windows) - sealed_queries):
        problems.append(f"{query_id}: candidate window for a query outside the sealed set")

    foreign_twins = 0
    for query_id in sorted(sealed_queries & set(windows)):
        window = windows[query_id]
        if len(window.candidates) != top_n:
            problems.append(
                f"{query_id}: {len(window.candidates)} candidates, every query must have {top_n}"
            )
        ranks = sorted(candidate.retrieval_rank for candidate in window.candidates)
        if ranks != list(range(1, len(window.candidates) + 1)):
            problems.append(f"{query_id}: retrieval ranks are not 1..{len(window.candidates)}")
        for candidate in window.candidates:
            if candidate.document_id not in document_ids:
                problems.append(
                    f"{query_id}: candidate {candidate.document_id} is not in the sealed corpus"
                )
            elif candidate.document_id.startswith("cf::") and candidate.document_id != own_twin.get(
                query_id
            ):
                foreign_twins += 1
    return _check(
        "candidate_windows",
        problems,
        f"{len(windows)} windows of {top_n}, foreign_twin={foreign_twins}",
    )


def render(report: Gate0AReport) -> str:
    return json.dumps(report.payload(), indent=2, sort_keys=True)
