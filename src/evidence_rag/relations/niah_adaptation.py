"""NIAH domain-adaptation half of M0 §3.8, and the sealed-600 constraint that gates it.

§3.8 pairs VitaminC main training with "NIAH train 的 mutation-log 对做域适配", under one hard
constraint:

    NIAH 域适配对的 parent page 必须与 sealed-600 零重叠.

SEALED-600 DOES NOT EXIST YET (§8 item 2; it is the other line's deliverable), so this half is
NOT legitimately runnable today. That is the whole reason this module refuses rather than warns.
A training run that skipped the overlap check would not look any different afterwards: five
checkpoints, an out-of-fold report, and a Gate 0B number in exactly the range a valid run
produces. The only moment the omission is visible is now, before the run.

WHAT UNBLOCKS IT. `load_sealed_parents` reads the sealed-600 build directory through the
corpus builder's own frozen fingerprint,
`materializer.sealed600.read_split_fingerprint(directory).parent_pages`. An earlier draft of
this module re-derived the parent set from `documents.jsonl` to avoid skew between the set that
was audited and the set that was checked; the builder now writes a hashed fingerprint as part
of the build, which answers that at the source and is the better seam. Both sides normalise
with `source_parent.normalize_parent`, which is the entire point of sharing the interface: two
normalisations would report zero overlap for two spellings of one article.

TWO PLACES WHERE THE SAFE DEFAULT ELSEWHERE IS THE WRONG DEFAULT HERE:

  1. `ParentIndex.parent_of` treats an unresolved document as its own parent, and §3.4 argues
     that is safe because it can only fail to merge two related sources, never merge two
     independent ones. For a LEAKAGE check the direction of harm reverses — a synthetic parent
     can never collide with a sealed title, so the check would pass exactly for the documents it
     could not resolve. This module requires an explicit parent for every document it touches.
  2. `task_probe.build_probe_pairs` SKIPS a record whose query or documents are missing, which
     is right for a probe whose denominators are reported. In a training set a skip silently
     changes what the model saw, and no field anywhere records it. This module raises.

THE TWIN ROWS' TRAINING TARGET IS NOW RULED: REFUTES (§3.8(a)). They carry `NOT_SUPPORTED`,
which A2 made a DERIVED label with no logit in a three-class head, so a target had to be named.
The reasoning on record: the pairs are genuine contradictions (A2 §10.7(B) says so even while
renaming the METRIC); the consumers' collapse still counts REFUTES as not-supported, so neither
0B-2 metric moves; and UNKNOWN is then trained only by VitaminC's NEI class, which is where
0B-1 needs it. `twin_label` remains REQUIRED with no default even so — the ruling makes one
value correct, and a default would make the correct value invisible in the manifest.
"""

import importlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from evidence_rag.materializer.provenance import MutationRecord
from evidence_rag.materializer.source_parent import normalize_parent
from evidence_rag.relations.claims import HypothesisForm, build_hypothesis
from evidence_rag.relations.models import RelationLabel
from evidence_rag.relations.task_probe import (
    GoldAnswerSource,
    build_probe_pairs,
    synthetic_family,
)
from evidence_rag.relations.training import (
    NIAH_SOURCE,
    TrainingExample,
    family_key,
    page_key,
)

PAIRS_PER_RECORD = 4

# The twin rows are the ones §3.8 leaves undeclared. SUPPORTS is refused because "everything
# SUPPORTS" is one of the two degenerate strategies the 0B-2 joint gate exists to block, and
# NOT_SUPPORTED because a three-class head has no logit for it.
PERMITTED_TWIN_LABELS = (RelationLabel.REFUTES, RelationLabel.UNKNOWN)


@dataclass(frozen=True)
class SealedCorpus:
    """The sealed-600 parent pages, plus enough provenance to say which corpus was checked.

    `parent_pages_sha256` is derived here from the pages themselves rather than read off the
    builder's fingerprint, so it depends on no field beyond the documented `parent_pages`. It is
    in the run manifest for one reason: "zero overlap with sealed-600" is an unfalsifiable claim
    in a report unless the reader can tell WHICH sealed-600 it was checked against.
    """

    parents: frozenset[str]
    parent_pages_sha256: str
    source: str

    @property
    def n_parent_pages(self) -> int:
        return len(self.parents)


def load_sealed_parents(directory: Path) -> SealedCorpus:
    """Read the sealed-600 build's frozen split fingerprint and take its parent-page set.

    The fingerprint is checked for degeneracy rather than trusted: `read_split_fingerprint`
    raises on an unresolvable title, so the "a sealed document has no parseable parent" hole is
    closed on the builder's side — but an empty or blank-valued page set would still make every
    subsequent zero-overlap check vacuously true, and that is the failure this whole module
    exists to refuse.
    """
    try:
        sealed600 = importlib.import_module("evidence_rag.materializer.sealed600")
    except ImportError as error:
        raise ValueError(
            "the sealed-600 corpus builder (evidence_rag.materializer.sealed600) is not"
            " importable, so §3.8's zero-overlap constraint cannot be evaluated and the NIAH"
            " domain-adaptation half must not run. That module is the corpus line's deliverable;"
            f" until it lands, run the VitaminC half alone. ({error})"
        ) from error

    fingerprint = sealed600.read_split_fingerprint(Path(directory))
    parents = frozenset(normalize_parent(page) for page in fingerprint.parent_pages)
    # normalize_parent is idempotent, so re-applying it to values the builder already
    # normalised cannot change them. It is applied anyway because the guarantee that matters is
    # that BOTH sides of the comparison were put into one space by one function.
    if not parents:
        raise ValueError(
            f"the sealed-600 fingerprint at {directory} lists no parent pages. Zero overlap"
            " against an empty set is true for every input, so this would let §3.8's constraint"
            " report success while checking nothing."
        )
    if any(not page for page in parents):
        raise ValueError(
            f"the sealed-600 fingerprint at {directory} contains a blank parent page. A blank"
            " normalises to '' and can never match a real title, so it is a hole in the audit"
            " that looks like a clean row."
        )
    return SealedCorpus(
        parents=parents,
        parent_pages_sha256=sha256(
            "\n".join(sorted(parents)).encode("utf-8")
        ).hexdigest(),
        source=str(directory),
    )


@dataclass(frozen=True)
class DevEvaluationQueries:
    """The dev-side evaluation run's query ids, plus enough provenance to say which dev set.

    Mirror of `SealedCorpus`, for the second leakage axis the 2026-08-09 finding opened: the
    adaptation pool and the dev evaluation run were drawn from ONE pool (202 of 2000 queries
    shared), the manifests' `split` labels all read 'dev' — including `runs/niah-train*` — and
    no code compared the two query sets. The comparison is therefore over ID SETS, and nothing
    on this path ever reads a `split` label: the label lied; the sets cannot.
    """

    query_ids: frozenset[str]
    query_ids_sha256: str
    source: str

    @property
    def n_queries(self) -> int:
        return len(self.query_ids)


def load_dev_queries(manifest_path: Path) -> DevEvaluationQueries:
    """Read the dev evaluation run's query-id set off its dataset manifest.

    Contracts only on `queries_file` — the one field the guard needs — and applies the same
    degeneracy checks as `load_sealed_parents`: an empty set makes every disjointness check
    vacuously true, and a blank id can never match a real one, so both refuse.
    """
    manifest_path = Path(manifest_path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"cannot read the dev evaluation manifest at {manifest_path}: {error}. Without it"
            " the adaptation set's disjointness from the dev evaluation cannot be checked, and"
            " the NIAH half must not run."
        ) from error
    queries_file = manifest.get("queries_file")
    if not queries_file:
        raise ValueError(
            f"the dev evaluation manifest at {manifest_path} names no queries_file, so its"
            " query-id set cannot be read."
        )
    ids = {
        str(json.loads(line)["query_id"])
        for line in (manifest_path.parent / queries_file)
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    }
    if not ids:
        raise ValueError(
            f"the dev evaluation run at {manifest_path} lists no queries. Disjointness against"
            " an empty set is true for every input, so this would let the check report success"
            " while checking nothing."
        )
    if any(not query_id.strip() for query_id in ids):
        raise ValueError(
            f"the dev evaluation run at {manifest_path} contains a blank query id. A blank can"
            " never match a real id, so it is a hole in the audit that looks like a clean row."
        )
    return DevEvaluationQueries(
        query_ids=frozenset(ids),
        query_ids_sha256=sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest(),
        source=str(manifest_path),
    )


def partition_dev_overlap(
    records: Sequence[MutationRecord], dev_queries: DevEvaluationQueries
) -> tuple[tuple[MutationRecord, ...], tuple[str, ...]]:
    """The sanctioned filter (ruling 2026-08-09 / A4): drop dev-overlapping families, visibly.

    Returns the kept records AND the sorted excluded query ids, because the excluded count
    belongs in the run manifest — `build_niah_examples` refuses overlapping records rather than
    filtering them itself precisely so that the removal cannot happen without leaving this
    trace.
    """
    kept = tuple(
        record for record in records if record.query_id not in dev_queries.query_ids
    )
    excluded = tuple(
        sorted(
            {
                record.query_id
                for record in records
                if record.query_id in dev_queries.query_ids
            }
        )
    )
    return kept, excluded


def build_niah_examples(
    *,
    records: Sequence[MutationRecord],
    question_by_query: Mapping[str, str],
    text_by_document: Mapping[str, str],
    parent_by_document: Mapping[str, str],
    sealed: SealedCorpus | None,
    dev_queries: DevEvaluationQueries | None,
    twin_label: RelationLabel,
    hypothesis_form: HypothesisForm = build_hypothesis,
    gold_answer_source: GoldAnswerSource = "canonical",
) -> tuple[TrainingExample, ...]:
    """Build the domain-adaptation rows, or refuse.

    `sealed` is positionally unavoidable and may not be omitted: `None` is the state the project
    is actually in today, and it has to stop the run rather than turn the check off.
    """
    if sealed is None:
        raise ValueError(
            "the NIAH domain-adaptation half cannot run without a sealed-600 parent-page set."
            " §3.8's hard constraint is that these pairs' parent pages have ZERO overlap with"
            " sealed-600, and sealed-600 is not built yet (§8 item 2). Pass the sealed corpus's"
            " documents.jsonl to `load_sealed_parents` once it exists. Running the VitaminC half"
            " alone is legitimate and is what the CLI does by default; running this half with the"
            " check disabled is not, and would be undetectable afterwards."
        )
    if twin_label not in PERMITTED_TWIN_LABELS:
        raise ValueError(
            f"twin_label {twin_label} is not one of"
            f" {[label.value for label in PERMITTED_TWIN_LABELS]}. §3.8 does not declare a"
            " three-class target for the mutation-log twin rows, so the caller must; but"
            " SUPPORTS would train the degenerate arm 0B-2's joint gate exists to block, and"
            " NOT_SUPPORTED is a derived label with no logit in a three-class head."
        )
    if dev_queries is None:
        raise ValueError(
            "the NIAH domain-adaptation half cannot run without the dev evaluation run's"
            " query-id set. The 2026-08-09 finding: the adaptation pool and the dev evaluation"
            " run were drawn from one pool (202 of 2000 queries shared), the manifests' split"
            " labels all read 'dev' and carry no meaning, and nothing compared the two sets."
            " Pass load_dev_queries(<dev evaluation manifest.json>); running with the check off"
            " would be undetectable afterwards, exactly like the sealed-600 case above."
        )
    leaking = sorted(
        {
            record.query_id
            for record in records
            if record.query_id in dev_queries.query_ids
        }
    )
    if leaking:
        raise ValueError(
            f"{len(leaking)} record(s) belong to queries in the dev evaluation set (e.g."
            f" {leaking[:5]}). Ruling 2026-08-09 (A4): dev-overlapping families are EXCLUDED"
            " from the adaptation set — call partition_dev_overlap() first and record its"
            " excluded count in the run manifest. This builder refuses rather than filtering"
            " silently, because a silent filter would leave no trace of how much of the pool"
            " was removed."
        )

    # Pass 1: resolve every parent page and run the §3.8 constraint BEFORE any pair is built,
    # so an overlapping corpus stops the run rather than being reported after the work.
    parents_by_record = [_record_parents(record, parent_by_document) for record in records]
    overlapping = sorted(
        {
            parent
            for parents in parents_by_record
            for parent in parents.values()
            if parent in sealed.parents
        }
    )
    if overlapping:
        raise ValueError(
            f"{len(overlapping)} NIAH parent page(s) overlap sealed-600 (e.g."
            f" {overlapping[:5]}), which §3.8 forbids without qualification: the model would be"
            " adapted on articles it is later evaluated over. Drop those queries from the NIAH"
            " train split and rebuild the mutation log; do not filter them here, because a"
            " filter applied at training time leaves the pool the sealed-600 evaluation draws"
            " from unchanged."
        )

    examples: list[TrainingExample] = []
    for record, parents in zip(records, parents_by_record, strict=True):
        pairs = build_probe_pairs(
            records=[record],
            question_by_query=question_by_query,
            text_by_document=text_by_document,
            hypothesis_form=hypothesis_form,
            gold_answer_source=gold_answer_source,
        )
        if len(pairs) != PAIRS_PER_RECORD:
            raise ValueError(
                f"mutation record {record.query_id!r} produced {len(pairs)} pairs rather than"
                f" {PAIRS_PER_RECORD}: its query text or one of its two documents is missing"
                " from the manifest, so the adaptation set is incomplete. build_probe_pairs"
                " skips such a record by design, which is right for a probe whose denominators"
                " are reported and wrong for a training set, where the skip leaves no trace."
            )
        keys = (
            *sorted({page_key(parent) for parent in parents.values()}),
            family_key(synthetic_family(record)),
        )
        for pair in pairs:
            label = twin_label if pair.label is RelationLabel.NOT_SUPPORTED else pair.label
            examples.append(
                TrainingExample(
                    premise=pair.premise,
                    hypothesis=pair.hypothesis,
                    label=label,
                    group_keys=keys,
                    source=NIAH_SOURCE,
                )
            )
    return tuple(examples)


def _record_parents(
    record: MutationRecord, parent_by_document: Mapping[str, str]
) -> dict[str, str]:
    """Parent page of the needle and of its counterfactual copy, both required.

    Both are checked rather than assuming they agree. The counterfactual is a mutated copy of
    the needle document and should carry the same title paragraph — but "should" is what the
    injector guarantees, and this is the audit that would notice if it stopped being true.
    """
    parents: dict[str, str] = {}
    missing: list[str] = []
    for document_id in (record.needle_document_id, record.counterfactual_document_id):
        parent = parent_by_document.get(document_id)
        if parent is None:
            missing.append(document_id)
            continue
        parents[document_id] = normalize_parent(parent)
    if missing:
        raise ValueError(
            f"mutation record {record.query_id!r} references document(s) {missing} with no"
            " parent page in the index. ParentIndex.parent_of would fall back to the"
            " document_id, which §3.4 justifies for vote counting — a synthetic parent can only"
            " fail to merge two related sources. On this path the same fallback silently PASSES"
            " the sealed-600 overlap check for precisely the documents whose article is unknown."
        )
    return parents
