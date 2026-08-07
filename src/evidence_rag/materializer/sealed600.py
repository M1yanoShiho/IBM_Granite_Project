"""fresh NIAH sealed-600 construction (M0 §4; algorithm from TRAINING_PLAN §4.2).

The sealed 600 is the ONE confirmatory test set in this project. Every S1-S6 conclusion so far
was measured on dev (M0 §4, "现状缺口"), so this set is the only artefact that can carry a
headline number — which is why its construction is a pre-registration problem rather than a
data-loading problem, and why almost everything here is a guard.

THREE THINGS THIS MODULE REFUSES TO DO, each because doing it quietly would produce a set that
looks right and is not:

  1. TOP UP A SHORT POOL. M0 §4 derives the pool from a measured rate: dev sampled 2000 queries
     and landed 1479 injections, so 600 injected questions need 600/(1-.261) = 812 eligible
     queries and the protocol takes 900 for headroom. A build that discovers mid-run that it is
     short has exactly one way to continue — add samples — and that is "补样本 = 看结果后改数据".
     So both shortfalls (eligible pool, injected target) raise, and the message carries the
     rejection breakdown so the operator can see WHICH assumption broke.

  2. AUDIT A PARENT PAGE IT COULD NOT RESOLVE. M0 §3.4 makes title resolution a hard
     prerequisite for the second leakage axis: "即使 support_unit 保持 document,解析本身也不可省
     —— 没有它 sealed 600 建不出来". `ParentIndex.parent_of` falls back to the document id, which
     is the right conservative choice for counting support and the WRONG one for a leakage
     audit: an unresolved document can never collide with anything, so the axis would report
     zero overlap because it could not look. Unresolved gold pages therefore reject the query
     here and raise on the existing-split side.

  3. DEPEND ON THE INTERPRETER'S RNG. Which 600 of the injected pool get sealed is decided by
     `frozen_order`, a sort on sha256(seed:query_id) rather than `random.shuffle`, whose stream
     is a CPython implementation detail. A frozen dataset that reproduces only on one build of
     one interpreter is not frozen.

The replacement bank is NOT built from this set's own answers. TRAINING_PLAN §4.2 pins it to
the seed=42 NIAH-train answer bank, and the difference is not cosmetic: a self-built bank would
draw replacements from the sealed set's own gold values, so one query's counterfactual could
assert another query's gold answer. `materializer/cli.py` builds a self-bank because for the
train/dev injections that IS the frozen bank; the sealed builder takes it as an input instead.
"""

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
from math import ceil
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from evidence_rag.contracts.models import CandidateSet, Document, Query, RetrieverProvenance
from evidence_rag.infrastructure.datasets import (
    DatasetBundle,
    DatasetManifest,
    GoldCase,
    JsonlDatasetAdapter,
)
from evidence_rag.materializer.answer_bank import AnswerBank
from evidence_rag.materializer.injector import find_injection_target, inject_counterfactual
from evidence_rag.materializer.provenance import MutationRecord, read_provenance, write_provenance
from evidence_rag.materializer.source_parent import parse_parent, write_parent_index
from evidence_rag.relations.task_probe import synthetic_family
from evidence_rag.selector.answer_norm import canonicalize_answer

# The protocol version this builder implements. It is written into the manifest and re-checked
# by the Gate 0A audit: a sealed set built under one protocol and audited under another is the
# single most expensive kind of mistake available here, and the version string is the only
# machine-readable trace of which rules were in force.
PROTOCOL_VERSION: Literal["g2-proto-4"] = "g2-proto-4"

SEALED_MANIFEST_FILE = "sealed_manifest.json"
# The §3.4 title sidecar, written as part of the sealed set rather than as a later optional step.
PARENT_INDEX_FILE = "source_parent.jsonl"
SEALED_DATASET_ID = "niah/dpr-w100-nq-sealed600"
SEALED_SPLIT = "sealed-test"

# M0 §4: "检索器冻结 = bm25". Graph 2.0's claim is "given a FIXED candidate pool, the selector is
# more reliable", so the pool is the frozen module interface. §3.5's G-FC judgement is made
# against a baseline of 0.4355 and §5.2's recall non-inferiority against a gate-off reference;
# both were measured on the bm25 pool, so a pool from another retriever does not merely add a
# confound, it compares across pools while printing entirely plausible numbers. The teammates'
# StrongBM25/hybrid arms are a separate generalisation table and never enter the C1 judgement.
FROZEN_RETRIEVER = "bm25"

# The second half of the freeze record. It cannot be written by the builder: the candidate pool
# does not exist until the frozen retriever has run over the sealed corpus, which is after the
# sealed manifest is frozen and must stay that way (a manifest that could be rewritten later is
# not a freeze). So the pin is a separate write-once file, and `audit_manifest_freeze` knows it
# is the ONE file allowed to appear in a sealed directory after the freeze.
CANDIDATE_FREEZE_FILE = "candidate_freeze.json"

# M0 §4's size derivation, kept as its INPUTS rather than as the single number 900. Recomputing
# from the inputs is what makes a later edit to the target or the skip rate fail loudly instead
# of silently shrinking the headroom.
DEV_SAMPLE_QUERIES = 2000
DEV_SAMPLE_INJECTED = 1479
DEV_SKIP_RATE = 0.261
TARGET_INJECTED = 600
POOL_SIZE = 900

LabelProvenance = Literal["official", "deterministic_rule"]

# How many offending ids an error message carries. Enough to debug, short enough to read.
_EXAMPLES = 5


def required_pool_size(*, target: int = TARGET_INJECTED, skip_rate: float = DEV_SKIP_RATE) -> int:
    """Eligible queries needed to land `target` injections at the measured skip rate."""
    if not 0.0 <= skip_rate < 1.0:
        raise ValueError(f"skip rate must be in [0, 1): {skip_rate}")
    if target <= 0:
        raise ValueError(f"target must be positive: {target}")
    return ceil(target / (1 - skip_rate))


def assert_pool_headroom(
    *,
    pool_size: int = POOL_SIZE,
    target: int = TARGET_INJECTED,
    skip_rate: float = DEV_SKIP_RATE,
) -> int:
    """Check the frozen pool covers the derivation BEFORE any corpus is streamed.

    Called at the top of the build and again by the audit, from the manifest's own recorded
    numbers. A pool that is short by arithmetic cannot be rescued by luck, and finding that out
    after a 21M-passage pass is finding it out at the worst possible moment.
    """
    required = required_pool_size(target=target, skip_rate=skip_rate)
    if pool_size < required:
        raise ValueError(
            f"pool of {pool_size} has no headroom: landing {target} injected questions at the "
            f"measured skip rate {skip_rate} needs {required} eligible queries (M0 §4). Raise "
            "the pool before building; topping up after a short run is changing the data on "
            "the basis of a result."
        )
    return required


def normalize_query_text(text: str) -> str:
    """Casefolded, whitespace-collapsed, terminal punctuation dropped.

    Axis 1 is "query_id AND normalised text" precisely because NQ contains the same question
    under different ids. Comparing raw strings would let "Who sank it?" and "who sank it" count
    as two independent questions.
    """
    return " ".join(text.split()).casefold().strip(" \t?.!,;:")


def passage_hash(text: str) -> str:
    """Axis 4's value: sha256 of the gold passage text exactly as it is stored."""
    return sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def frozen_order(query_ids: Iterable[str], *, seed: int) -> tuple[str, ...]:
    """A deterministic permutation that does not depend on the interpreter.

    `random.Random(seed).sample` would do the same job until the day CPython changes its
    sampling algorithm — it already changed what it accepts in 3.11 — and a sealed set that
    reproduces on one interpreter build is not sealed. Sorting on sha256(seed:id) is a pure
    function of the inputs, stable forever, and uniform enough that the drawn pool carries no
    structure from the source dataset's own id ordering.
    """
    ids = list(query_ids)
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate query id in the frozen order: the pool must be a set")
    return tuple(
        sorted(ids, key=lambda item: (sha256(f"{seed}:{item}".encode()).hexdigest(), item))
    )


@dataclass(frozen=True)
class SplitFingerprint:
    """One split reduced to the five leakage axes of M0 §4.

    Defaults are empty so a test can vary one axis, which is also why `gate0a.audit_axes`
    refuses an axis whose existing side is empty: an empty set never collides, and "zero
    overlap" computed against nothing is not evidence of anything.
    """

    name: str
    query_ids: frozenset[str] = field(default_factory=frozenset)
    query_texts: frozenset[str] = field(default_factory=frozenset)
    parent_pages: frozenset[str] = field(default_factory=frozenset)
    answer_entities: frozenset[str] = field(default_factory=frozenset)
    passage_hashes: frozenset[str] = field(default_factory=frozenset)
    synthetic_families: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class SourceQuery:
    """A candidate query from the unused DPR/NQ pool, before any eligibility rule is applied."""

    query_id: str
    text: str
    answers: tuple[str, ...]
    gold_document_ids: tuple[str, ...]


@dataclass(frozen=True)
class PoolEvaluation:
    eligible_query_ids: tuple[str, ...]
    rejections: Mapping[str, int]
    n_considered: int


@dataclass(frozen=True)
class PoolSelection:
    pool_query_ids: tuple[str, ...]
    rejections: Mapping[str, int]
    n_considered: int


@dataclass(frozen=True)
class InjectedSelection:
    query_ids: tuple[str, ...]
    records: tuple[MutationRecord, ...]
    twins: tuple[Document, ...]
    rejections: Mapping[str, int]
    n_attempted: int


def parent_page_of(document: Document) -> str:
    """The normalised article title, or a raise.

    Every caller here needs the parent for a LEAKAGE comparison, where an unresolved page is
    indistinguishable from a page that happens not to collide. `ParentIndex.parent_of`'s
    self-parent fallback is correct for counting independent support and wrong here, so this
    function has no fallback at all.
    """
    parent = parse_parent(document.text)
    if parent is None:
        raise ValueError(
            f"document {document.document_id!r} has no parsable article title, so its parent "
            "page cannot enter the leakage audit (M0 §3.4). An unresolved page never collides, "
            "which would make axis 2 report zero overlap because it could not look."
        )
    return parent


def fingerprint_bundle(
    name: str, bundle: DatasetBundle, records: Sequence[MutationRecord]
) -> SplitFingerprint:
    """Reduce a materialised split to its five axes.

    Only GOLD passages contribute to the parent-page and passage-hash axes. The corpus itself
    is a fresh reservoir sample of the same 21M-passage dump on both sides, so distractor pages
    overlap by construction and requiring zero overlap there would be requiring the impossible.
    What must not repeat is the needle: TRAINING_PLAN §4.4 groups NIAH by parent page, answer
    entity and synthetic family, and those are properties of the gold.
    """
    text_by_id = {document.document_id: document for document in bundle.documents}
    gold_documents: list[Document] = []
    for gold_case in bundle.gold_cases:
        for document_id in gold_case.relevant_document_ids or ():
            document = text_by_id.get(document_id)
            if document is None:
                raise ValueError(
                    f"split {name!r} names gold document {document_id!r} that its own "
                    "documents.jsonl does not contain"
                )
            gold_documents.append(document)
    return SplitFingerprint(
        name=name,
        query_ids=frozenset(query.query_id for query in bundle.queries),
        query_texts=frozenset(normalize_query_text(query.text) for query in bundle.queries),
        parent_pages=frozenset(parent_page_of(document) for document in gold_documents),
        answer_entities=frozenset(
            canonicalize_answer(answer)
            for gold_case in bundle.gold_cases
            for answer in gold_case.reference_answers or ()
        ),
        passage_hashes=frozenset(passage_hash(document.text) for document in gold_documents),
        synthetic_families=frozenset(synthetic_family(record) for record in records),
    )


def read_split_fingerprint(directory: Path) -> SplitFingerprint:
    """Fingerprint an existing split directory (`manifest.json` + `provenance.jsonl`).

    The mutation log is REQUIRED, not optional. Axis 5 compares synthetic families, and a split
    read without its log contributes an empty family set — after which the axis reports zero
    overlap because it had nothing to compare rather than because the sets are disjoint. A split
    that genuinely carries no mutations says so with an empty file, which is a deliberate act
    and leaves a trace in the repository.
    """
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    provenance_path = directory / "provenance.jsonl"
    if not manifest_path.is_file():
        raise ValueError(f"existing split {directory} has no manifest.json")
    if not provenance_path.is_file():
        raise ValueError(
            f"existing split {directory} has no provenance.jsonl, so its synthetic families "
            "cannot enter axis 5. Point at the INJECTED directory (runs/niah-*-injected), or "
            "write an empty provenance.jsonl to state that this split carries no mutations."
        )
    bundle = JsonlDatasetAdapter.load(manifest_path)
    return fingerprint_bundle(directory.name, bundle, read_provenance(provenance_path))


def _pool_rejection(
    candidate: SourceQuery,
    documents: Mapping[str, Document],
    existing: Sequence[SplitFingerprint],
) -> str | None:
    """The first rule this candidate breaks, or None.

    Injectability is deliberately NOT checked here. TRAINING_PLAN §4.2's eligibility rules
    (single normalised gold value, an alias occurring exactly once, a same-class replacement)
    already live in the injector, and restating them to get a prettier rejection histogram is
    how two definitions of "eligible" drift apart. The pool is what survives the LEAKAGE axes;
    the injector then skips its own share, which is the .261 M0 §4 measured.
    """
    gold_documents = [
        documents[document_id]
        for document_id in candidate.gold_document_ids
        if document_id in documents
    ]
    if not gold_documents:
        return "no_gold_document"
    if not candidate.answers:
        # Not an injector rule: axis 3 is undefined without an answer, so such a query could not
        # be audited even if it could be injected.
        return "no_reference_answers"
    if any(candidate.query_id in split.query_ids for split in existing):
        return "leak_query_id"
    normalized_text = normalize_query_text(candidate.text)
    if any(normalized_text in split.query_texts for split in existing):
        return "leak_query_text"
    entities = {canonicalize_answer(answer) for answer in candidate.answers}
    if any(entities & split.answer_entities for split in existing):
        return "leak_answer_entity"
    parents: set[str] = set()
    for document in gold_documents:
        parent = parse_parent(document.text)
        if parent is None:
            return "unresolved_parent"
        parents.add(parent)
    if any(parents & split.parent_pages for split in existing):
        return "leak_parent_page"
    # Every gold passage is checked, not just the one the injector will pick as the needle:
    # which passage becomes the needle is decided later, and an axis that depends on that choice
    # is an axis that can be changed by an unrelated injector fix.
    hashes = {passage_hash(document.text) for document in gold_documents}
    if any(hashes & split.passage_hashes for split in existing):
        return "leak_passage_hash"
    return None


def evaluate_pool(
    *,
    candidates: Sequence[SourceQuery],
    documents: Mapping[str, Document],
    existing: Sequence[SplitFingerprint],
) -> PoolEvaluation:
    """Apply axes 1-4 to every candidate and count the rejections by reason."""
    eligible: list[str] = []
    rejections: dict[str, int] = {}
    for candidate in candidates:
        reason = _pool_rejection(candidate, documents, existing)
        if reason is None:
            eligible.append(candidate.query_id)
        else:
            rejections[reason] = rejections.get(reason, 0) + 1
    return PoolEvaluation(
        eligible_query_ids=tuple(eligible),
        rejections=rejections,
        n_considered=len(candidates),
    )


def select_pool(
    *,
    candidates: Sequence[SourceQuery],
    documents: Mapping[str, Document],
    existing: Sequence[SplitFingerprint],
    seed: int,
    pool_size: int = POOL_SIZE,
) -> PoolSelection:
    """Freeze the pool: eligible queries, in the frozen order, cut to `pool_size`."""
    evaluation = evaluate_pool(candidates=candidates, documents=documents, existing=existing)
    ordered = frozen_order(evaluation.eligible_query_ids, seed=seed)
    if len(ordered) < pool_size:
        raise ValueError(
            f"only {len(ordered)} eligible queries survived the leakage axes, {pool_size} are "
            f"required (M0 §4). Rejections by reason: {json.dumps(dict(evaluation.rejections), sort_keys=True)}. "
            "Widen the SOURCE pool (a different split, or more queries from it) and rebuild "
            "from scratch; do not lower the pool size to fit what survived."
        )
    return PoolSelection(
        pool_query_ids=ordered[:pool_size],
        rejections=evaluation.rejections,
        n_considered=evaluation.n_considered,
    )


def inject_pool(
    *,
    pool_query_ids: Sequence[str],
    candidates_by_id: Mapping[str, SourceQuery],
    documents: Mapping[str, Document],
    bank: AnswerBank,
    existing: Sequence[SplitFingerprint],
    seed: int,
    target: int = TARGET_INJECTED,
) -> InjectedSelection:
    """Inject counterfactuals down the frozen pool and stop at `target`.

    Stopping at the target rather than sealing everything that injects is what makes the sealed
    set a pure function of the frozen pool: the headroom exists so the target can be reached,
    not so the set can grow to whatever the yield happened to be.
    """
    query_ids: list[str] = []
    records: list[MutationRecord] = []
    twins: list[Document] = []
    families: set[str] = set()
    rejections: dict[str, int] = {}
    attempted = 0
    for query_id in pool_query_ids:
        if len(query_ids) == target:
            break
        attempted += 1
        # KeyError, not a skip: the pool and the candidate index are two views of one frozen
        # decision, and a member that cannot be resolved means they disagree.
        candidate = candidates_by_id[query_id]
        gold_case = GoldCase(
            query_id=candidate.query_id,
            relevant_document_ids=candidate.gold_document_ids,
            reference_answers=candidate.answers,
        )
        target_passage = find_injection_target(gold_case, documents)
        if target_passage is None:
            rejections["not_injectable"] = rejections.get("not_injectable", 0) + 1
            continue
        injection = inject_counterfactual(target_passage, bank, seed=seed)
        if injection is None:
            rejections["no_replacement"] = rejections.get("no_replacement", 0) + 1
            continue
        twin, record = injection
        family = synthetic_family(record)
        if any(family in split.synthetic_families for split in existing):
            rejections["leak_synthetic_family"] = rejections.get("leak_synthetic_family", 0) + 1
            continue
        if family in families:
            # Axis 5 within the set rather than across splits. R051 resamples per query and the
            # grouping unit is the family, so two sealed questions in one family would be one
            # group counted twice.
            rejections["duplicate_synthetic_family"] = (
                rejections.get("duplicate_synthetic_family", 0) + 1
            )
            continue
        families.add(family)
        query_ids.append(query_id)
        records.append(record)
        twins.append(twin)
    if len(query_ids) < target:
        raise ValueError(
            f"the frozen pool yielded {len(query_ids)} injected questions, target is {target} "
            f"(M0 §4). Rejections by reason: {json.dumps(rejections, sort_keys=True)}. "
            "不得跑到一半发现不够再补 — rebuild from a wider source pool rather than adding "
            "queries to this one, and record the realised skip rate against the .261 the "
            "protocol derived the pool from."
        )
    return InjectedSelection(
        query_ids=tuple(query_ids),
        records=tuple(records),
        twins=tuple(twins),
        rejections=rejections,
        n_attempted=attempted,
    )


def _write_jsonl(path: Path, records: Iterable[object]) -> None:
    lines = [
        json.dumps(record, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        for record in records
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_sealed_dataset(
    directory: Path,
    *,
    documents: Sequence[Document],
    queries: Sequence[Query],
    gold_cases: Sequence[GoldCase],
    records: Sequence[MutationRecord],
    seed: int = 42,
) -> dict[str, str]:
    """Write the sealed artefacts and return {filename: sha256} for the freeze record.

    The parent sidecar is written HERE, not left to a later `build_source_parent` run. M0 §3.4
    calls title resolution a hard prerequisite of this dataset; a sealed set whose sidecar is a
    separate optional step can be evaluated with `support_unit=parent` requested and the sidecar
    absent, and the only thing standing between that and a silently document-counted run is an
    operator remembering a command.

    The bundle is loaded back through `JsonlDatasetAdapter` before the hashes are taken, so a
    set that cannot be read is never frozen.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = DatasetManifest(
        dataset_id=SEALED_DATASET_ID,
        dataset_version=f"sealed600-seed{seed}",
        split=SEALED_SPLIT,
        documents_file="documents.jsonl",
        queries_file="queries.jsonl",
        gold_cases_file="gold_cases.jsonl",
    )
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest.model_dump(mode="json"), ensure_ascii=True, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    _write_jsonl(
        directory / manifest.documents_file,
        (document.model_dump(mode="json") for document in documents),
    )
    _write_jsonl(
        directory / manifest.queries_file, (query.model_dump(mode="json") for query in queries)
    )
    _write_jsonl(
        directory / manifest.gold_cases_file,
        (gold_case.model_dump(mode="json") for gold_case in gold_cases),
    )
    write_provenance(directory / "provenance.jsonl", records)
    parent_by_document = {
        document.document_id: parse_parent(document.text)
        for document in documents
        if parse_parent(document.text) is not None
    }
    write_parent_index(
        directory / PARENT_INDEX_FILE,
        {key: value for key, value in parent_by_document.items() if value is not None},
    )
    JsonlDatasetAdapter.load(manifest_path)
    return {
        name: sha256_file(directory / name)
        for name in (
            "manifest.json",
            manifest.documents_file,
            manifest.queries_file,
            manifest.gold_cases_file,
            "provenance.jsonl",
            PARENT_INDEX_FILE,
        )
    }


class SealedManifest(BaseModel):
    """The freeze record: what was built, from what, under which protocol, hashed.

    Written ONCE, before any run. `label_provenance` is a closed set rather than free text so
    M0 §6's "primary label 的 provenance 只能是 official 或 deterministic_rule" is enforced at
    write time — an LLM-judged label cannot be recorded here at all, which is a stronger
    guarantee than auditing a string afterwards.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    protocol_version: Literal["g2-proto-4"]
    protocol_document_sha256: str
    dataset_id: str
    dataset_version: str
    split: str
    source_dataset_id: str
    seed: int
    corpus_size: int
    target_injected: int
    dev_sample_queries: int
    dev_sample_injected: int
    dev_skip_rate: float
    required_pool_size: int
    pool_size: int
    n_considered: int
    n_injected: int
    answer_bank_hash: str
    answer_bank_source: str
    existing_splits: tuple[str, ...]
    label_provenance: Mapping[str, LabelProvenance]
    pool_rejections: Mapping[str, int]
    injection_rejections: Mapping[str, int]
    pool_query_ids: tuple[str, ...]
    query_ids: tuple[str, ...]
    artifact_sha256: Mapping[str, str]


def freeze_sealed_manifest(directory: Path, manifest: SealedManifest) -> Path:
    """Write the manifest, refusing to replace one that already exists.

    An overwrite is what a top-up run looks like from the outside: same path, same schema, a
    different set. Refusing means a second build has to remove the freeze by hand, which is a
    deliberate act that leaves a trace.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / SEALED_MANIFEST_FILE
    if path.exists():
        raise ValueError(
            f"{path} is already frozen. A sealed set is built once; rebuilding into the same "
            "directory would replace the freeze record with no trace that it changed. Build "
            "into a new directory, or remove the frozen manifest deliberately."
        )
    path.write_text(
        json.dumps(manifest.model_dump(mode="json"), ensure_ascii=True, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return path


def read_sealed_manifest(directory: Path) -> SealedManifest:
    path = Path(directory) / SEALED_MANIFEST_FILE
    if not path.is_file():
        raise ValueError(f"{path} does not exist: this directory holds no frozen sealed set")
    return SealedManifest.model_validate_json(path.read_text(encoding="utf-8"))


class CandidateFreeze(BaseModel):
    """The frozen identity of the candidate pool the sealed 600 is evaluated on (M0 §4).

    Two things are pinned, and neither is sufficient alone. `retriever` answers "was this pool
    built by the frozen retriever" — without it a strong-bm25 pool passes every shape check
    identically. `candidate_sha256` answers "is this the SAME pool" — without it two bm25 runs
    over two corpus builds both name bm25 and hand §3.5 and §5.2 different windows.

    `sealed_manifest_sha256` binds the pin to one sealed set, so a pin cannot be carried into a
    directory whose 600 questions it never described.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    protocol_version: Literal["g2-proto-4"]
    sealed_manifest_sha256: str
    candidate_file: str
    candidate_sha256: str
    n_windows: int
    top_n: int
    retriever: RetrieverProvenance


def freeze_candidate_pin(directory: Path, pin: CandidateFreeze) -> Path:
    """Write the pin, refusing to replace one that already exists.

    Same reasoning as `freeze_sealed_manifest`. Re-pinning is what "we re-ran retrieval" looks
    like from the outside, and quietly moving the pin to whatever the newest run produced would
    make the record follow the data instead of constraining it.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / CANDIDATE_FREEZE_FILE
    if path.exists():
        raise ValueError(
            f"{path} is already pinned. The candidate pool is frozen once; replacing the pin "
            "would let a later retrieval run redefine the windows every pre-registered "
            "threshold in §3.5 and §5.2 was measured against. Remove it deliberately if the "
            "pool genuinely has to be rebuilt, and say so in the tracker."
        )
    path.write_text(
        json.dumps(pin.model_dump(mode="json"), ensure_ascii=True, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return path


def read_candidate_pin(directory: Path) -> CandidateFreeze | None:
    """The pin, or None when retrieval has not been pinned yet.

    None is a legitimate state — it is the state a freshly built sealed set is in — so this
    returns rather than raises. What must not happen is for the audit to read None as "fine".
    """
    path = Path(directory) / CANDIDATE_FREEZE_FILE
    if not path.is_file():
        return None
    return CandidateFreeze.model_validate_json(path.read_text(encoding="utf-8"))


def candidate_retrievers(
    candidate_sets: Sequence[CandidateSet],
) -> tuple[RetrieverProvenance | None, ...]:
    """The distinct producers a candidate file names, in order of first appearance.

    `None` is carried as a value rather than dropped, because "some windows name a retriever and
    some do not" is a different and worse finding than "none of them do": the first is a file
    somebody edited, the second is a file written before the field existed. Callers have to be
    able to tell them apart.
    """
    seen: list[RetrieverProvenance | None] = []
    for candidate_set in candidate_sets:
        if candidate_set.retriever not in seen:
            seen.append(candidate_set.retriever)
    return tuple(seen)


def sole_retriever(candidate_sets: Sequence[CandidateSet]) -> RetrieverProvenance:
    """The single producer of a candidate file, or a raise.

    Used where a pool is being ADMITTED rather than reported on, so every ambiguity is fatal:
    the audit needs to describe what it found, but nothing may pin a pool it cannot identify.
    """
    if not candidate_sets:
        raise ValueError(
            "no candidate windows: an empty pool has no producer to check and no shape to "
            "violate, so every check over it would pass by having nothing to look at (M0 §4)."
        )
    found = candidate_retrievers(candidate_sets)
    if found == (None,):
        raise ValueError(
            "this candidate file names no retriever, so there is no way to establish it came "
            f"from the frozen {FROZEN_RETRIEVER} pool (M0 §4). It must be re-retrieved, not "
            "re-labelled: writing the id in now would record an assumption as a measurement."
        )
    if len(found) != 1:
        names = sorted("none" if item is None else item.name for item in found)
        raise ValueError(
            f"this candidate file names {len(found)} different producers ({', '.join(names)}), "
            "so it is not the output of one retrieval run (M0 §4)."
        )
    only = found[0]
    assert only is not None  # len(found) == 1 and found != (None,)
    return only


def verify_artifacts(directory: Path, manifest: SealedManifest) -> tuple[str, ...]:
    """Re-hash every recorded artefact. Returns one description per mismatch, empty when clean.

    This is what makes "manifest 与 hash 在任何实验之前冻结" a control rather than a claim: the
    hashes are recorded at build time and compared at audit time, so an edit between the two —
    including a well-meant one — is visible.
    """
    directory = Path(directory)
    problems: list[str] = []
    for name, expected in sorted(manifest.artifact_sha256.items()):
        path = directory / name
        if not path.is_file():
            problems.append(f"{name}: recorded in the manifest but missing from {directory}")
            continue
        actual = sha256_file(path)
        if actual != expected:
            problems.append(f"{name}: sha256 {actual} does not match the frozen {expected}")
    return tuple(problems)
