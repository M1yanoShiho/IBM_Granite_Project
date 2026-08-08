import re
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    SerializerFunctionWrapHandler,
    model_serializer,
    model_validator,
)


def _omit_if_none(data: dict[str, Any], key: str) -> dict[str, Any]:
    """Drop a None-valued optional key so artifacts serialize exactly as they did
    before the field existed — committed dataset/corpus signatures and frozen
    artifacts stay byte-stable, and adding provenance to one producer does not
    force a re-freeze of every artifact that predates it."""
    if data.get(key) is None:
        data.pop(key, None)
    return data


def _omit_absent_metadata(data: dict[str, Any]) -> dict[str, Any]:
    """Drop a None ``metadata`` key so plain-text corpora serialize exactly as
    they did before multimodal provenance existed."""
    return _omit_if_none(data, "metadata")

NonEmpty = Annotated[str, Field(min_length=1)]
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
PositiveRank = Annotated[int, Field(ge=1)]
PositiveLimit = Annotated[int, Field(ge=1)]


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Query(FrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    query_id: NonEmpty
    text: NonEmpty


class QueryChecklist(FrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    query_id: NonEmpty
    focus: NonEmpty
    required_facts: tuple[NonEmpty, ...]
    constraints: tuple[NonEmpty, ...] = ()


class SourceMetadata(FrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    source_type: Literal["txt", "pdf", "image"]
    file_name: NonEmpty
    page_number: PositiveRank | None = None
    image_path: NonEmpty | None = None


class Document(FrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    document_id: NonEmpty
    text: NonEmpty
    source_uri: NonEmpty
    metadata: SourceMetadata | None = None

    @model_serializer(mode="wrap")
    def _serialize(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        return _omit_absent_metadata(handler(self))


class EvidenceCandidate(FrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    evidence_id: NonEmpty
    document_id: NonEmpty
    chunk_id: NonEmpty
    text: NonEmpty
    source_uri: NonEmpty
    retrieval_score: FiniteFloat
    retrieval_rank: PositiveRank
    metadata: SourceMetadata | None = None

    @model_serializer(mode="wrap")
    def _serialize(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        return _omit_absent_metadata(handler(self))


class RetrieverProvenance(FrozenModel):
    """Which retriever produced a candidate pool, recorded by the run that produced it.

    M0 §4 freezes the retriever to bm25 because the Graph 2.0 claim is conditional on a FIXED
    candidate pool: change the retriever and pool composition becomes a confounding variable,
    and §3.5's G-FC baseline and §5.2's recall reference — both measured on the bm25 pool —
    start being compared across pools while printing entirely plausible numbers. A pool that
    does not say what built it cannot be audited against that freeze, because a pool from a
    different retriever has exactly the same shape.

    The parameter digest is part of the identity, not decoration. ``bm25 k1=1.5 b=0.75`` and
    ``bm25 k1=0.9 b=0.4`` are one name over two different pools, and §4's freeze is a statement
    about the pool.
    """

    schema_version: Literal["1.0"] = "1.0"
    name: NonEmpty
    implementation_version: NonEmpty
    parameters_sha256: Digest


class CandidateSet(FrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    query_id: NonEmpty
    candidates: tuple[EvidenceCandidate, ...]
    retriever: RetrieverProvenance | None = None
    """The producer, stamped by the retrieval stage. ``None`` means a pool written before this
    field existed: it is NOT a claim that the pool came from the frozen retriever, and
    ``materializer/gate0a.py`` refuses to read it as one."""

    @model_validator(mode="after")
    def unique_ids_and_ranks(self) -> "CandidateSet":
        ids = tuple(item.evidence_id for item in self.candidates)
        ranks = tuple(item.retrieval_rank for item in self.candidates)
        if len(ids) != len(set(ids)):
            raise ValueError("candidate evidence IDs must be unique")
        if len(ranks) != len(set(ranks)):
            raise ValueError("candidate retrieval ranks must be unique")
        return self

    @model_serializer(mode="wrap")
    def _serialize(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        return _omit_if_none(handler(self), "retriever")


class SelectionItem(FrozenModel):
    evidence_id: NonEmpty
    selection_score: FiniteFloat
    selection_rank: PositiveRank


class SelectionResult(FrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    query_id: NonEmpty
    items: tuple[SelectionItem, ...]

    @model_validator(mode="after")
    def unique_ids_and_ranks(self) -> "SelectionResult":
        ids = tuple(item.evidence_id for item in self.items)
        ranks = tuple(item.selection_rank for item in self.items)
        if len(ids) != len(set(ids)):
            raise ValueError("selected evidence IDs must be unique")
        if len(ranks) != len(set(ranks)):
            raise ValueError("selection ranks must be unique")
        if ranks != tuple(range(1, len(ranks) + 1)):
            raise ValueError("selection items must be ordered by consecutive ranks")
        return self


class SelectedEvidenceSet(FrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    query_id: NonEmpty
    evidence: tuple[EvidenceCandidate, ...]

    @model_validator(mode="after")
    def unique_ids(self) -> "SelectedEvidenceSet":
        ids = tuple(item.evidence_id for item in self.evidence)
        if len(ids) != len(set(ids)):
            raise ValueError("selected evidence IDs must be unique")
        return self


UNVERIFIED_ANNOTATION = "[unverified]"
"""Marks a sentence the Generator kept but could not verify.

It lives in ``contracts`` because ``GenerationResult`` now depends on it: the
contract's guarantee is stated in terms of this label, so the label is part of
the contract rather than an implementation detail of one generator."""

REVIEW_ANNOTATION = "[may warrant review]"
"""Marks a sentence that IS supported and IS cited, but which a secondary check
disagreed about.

Deliberately not "the evidence conflicts with this claim". That phrasing is a
factual assertion about the evidence and two blind adjudications put it wrong
about 70% of the time; asserting it would be worse than the deletion it replaces,
which at least asserted nothing. This label asserts low confidence about itself.

It means something entirely different from ``UNVERIFIED_ANNOTATION`` and the two
must never be conflated by a reader: ``[unverified]`` is "no supporting evidence
was found, and this sentence carries no citation", while this is "supporting
evidence was found and cited, and a screening check flagged it anyway". A flagged
sentence always carries a citation."""

_ANNOTATIONS = (UNVERIFIED_ANNOTATION, REVIEW_ANNOTATION)

_SENTENCE_SPLIT = re.compile(r"[.!?]+(?:\s|$)")

_ABBREVIATIONS = frozenset(
    """
    mr mrs ms dr prof rev hon st mt ft jr sr inc ltd co corp dept est
    vs v etc eg ie no nos vol op fig al approx dept univ
    jan feb mar apr jun jul aug sep sept oct nov dec
    mon tue tues wed thu thur thurs fri sat sun
    ave blvd rd gen col sgt capt lt maj pres sen gov
    """.split()
)
"""Tokens whose trailing period does not end a sentence.

Kept deliberately small: every entry has to be a word that essentially never ends
a sentence, because a wrong entry here MERGES two real sentences, which is the
more damaging error. ``may``/``march``/``august`` are absent for that reason.
"""

_TRAILING_TOKEN = re.compile(r"([A-Za-z][A-Za-z.]*)$")
_DOTTED_ACRONYM = re.compile(r"(?:[A-Za-z]\.)+[A-Za-z]$")


def _terminator_ends_a_sentence(text: str, terminator_start: int) -> bool:
    """Whether the terminator at ``terminator_start`` really ends a sentence.

    It does not when the period belongs to an abbreviation or an initial. This is
    load-bearing rather than cosmetic: ``GenerationResult`` requires every sentence
    of an uncited answer to carry the unverified label, so counting ``Mount St.
    Helens erupted.`` as two sentences with one label made the validator REJECT a
    correctly annotated answer and destroy it.
    """
    match = _TRAILING_TOKEN.search(text[:terminator_start])
    if match is None:
        return True
    token = match.group(1)
    if len(token) == 1:  # an initial: "Patrick S. Castagne"
        return False
    if token.lower() in _ABBREVIATIONS:  # "Mount St. Helens"
        return False
    return not _DOTTED_ACRONYM.fullmatch(token)  # "the 1913 U.S. Open"


def split_sentences(text: str) -> list[str]:
    """Sentences in ``text``, treating abbreviations and initials as interior.

    The single sentence rule the contract and the scorer both use, so a sentence
    the validator demands a label for is the same sentence the metric scores.
    """
    parts: list[str] = []
    start = 0
    for match in _SENTENCE_SPLIT.finditer(text):
        if not _terminator_ends_a_sentence(text, match.start()):
            continue
        piece = text[start : match.end()].strip()
        if piece:
            parts.append(piece)
        start = match.end()
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def ends_with_abbreviation(part: str) -> bool:
    """True when ``part``'s trailing period is an abbreviation's, not a sentence's.

    Lets a caller that splits with someone else's tokenizer (the scorer uses
    ALCE's) repair the same false boundaries without adopting a different rule.
    """
    stripped = part.rstrip()
    if not stripped.endswith("."):
        return False
    return not _terminator_ends_a_sentence(stripped, len(stripped) - 1)


def is_unverified_annotation(sentence: str) -> bool:
    """True for a sentence the Generator kept without being able to verify it.

    False for a review-flagged sentence: that one is verified and cited.
    """
    return UNVERIFIED_ANNOTATION in sentence


def is_review_flagged(sentence: str) -> bool:
    """True for a cited sentence a secondary check flagged for review."""
    return REVIEW_ANNOTATION in sentence


def strip_annotations(text: str) -> str:
    """The text with every annotation label removed -- the prose alone.

    What a judge, a correctness scorer, or a reader-facing surface should see. A
    label left in would be scored as part of the sentence.
    """
    for label in _ANNOTATIONS:
        text = text.replace(label, " ")
    return " ".join(text.split())


def strip_unverified_annotation(text: str) -> str:
    """The text with annotation labels removed, for correctness scoring or display."""
    return strip_annotations(text)


def count_sentences(answer: str) -> int:
    """Sentences in an answer, ignoring the annotation labels themselves.

    The label is appended *after* a sentence's terminator, so it must be removed
    before counting or it would be read as a sentence of its own.
    """
    return len(split_sentences(strip_unverified_annotation(answer)))


class GenerationResult(FrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    query_id: NonEmpty
    answer: str
    cited_evidence_ids: tuple[NonEmpty, ...]

    @model_validator(mode="after")
    def answer_and_citations_match(self) -> "GenerationResult":
        if not self.answer.strip() and self.cited_evidence_ids:
            raise ValueError("empty answer cannot contain citations")
        if len(self.cited_evidence_ids) != len(set(self.cited_evidence_ids)):
            raise ValueError("citations must be unique")
        if self.answer.strip() and not self.cited_evidence_ids:
            # The old rule rejected this outright, which guaranteed "every answer
            # is grounded". That guarantee capped the annotate policy: a query
            # whose claims all failed verification had to abstain wholesale and
            # throw its annotations away. The invariant is SWAPPED, not dropped --
            # an answer may now be entirely uncited, but only if every sentence in
            # it is labelled unverified. The guarantee becomes "every ungrounded
            # sentence is labelled", which is the honest one for this method.
            if count_sentences(self.answer) > self.answer.count(UNVERIFIED_ANNOTATION):
                raise ValueError(
                    "uncited answer must mark every sentence with "
                    f"{UNVERIFIED_ANNOTATION!r}"
                )
        return self


class PipelineRun(FrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    query: Query
    checklist: QueryChecklist
    top_k: PositiveLimit
    max_selected: PositiveLimit
    candidates: CandidateSet
    selection: SelectionResult
    selected: SelectedEvidenceSet
    generation: GenerationResult

    @model_validator(mode="after")
    def stages_are_consistent(self) -> "PipelineRun":
        expected = self.query.query_id
        if self.checklist.query_id != expected:
            raise ValueError("checklist query ID differs")
        stage_query_ids = (
            self.candidates.query_id,
            self.selection.query_id,
            self.selected.query_id,
            self.generation.query_id,
        )
        if any(query_id != expected for query_id in stage_query_ids):
            raise ValueError("pipeline run query IDs differ")
        if len(self.candidates.candidates) > self.top_k:
            raise ValueError("candidate count exceeds top_k")
        if len(self.selection.items) > self.max_selected:
            raise ValueError("selection count exceeds max_selected")

        candidates_by_id = {
            item.evidence_id: item for item in self.candidates.candidates
        }
        selected_ids = tuple(item.evidence_id for item in self.selection.items)
        unknown = tuple(
            evidence_id
            for evidence_id in selected_ids
            if evidence_id not in candidates_by_id
        )
        if unknown:
            raise ValueError(f"selection contains unknown evidence: {unknown}")
        expected_selected = tuple(
            candidates_by_id[evidence_id] for evidence_id in selected_ids
        )
        if self.selected.evidence != expected_selected:
            raise ValueError("selected evidence does not match selection")

        cited = set(self.generation.cited_evidence_ids)
        unknown_citations = cited - set(selected_ids)
        if unknown_citations:
            raise ValueError(
                f"generation cites unselected evidence: {sorted(unknown_citations)}"
            )
        return self
