"""End-to-end: build a sealed set from a fake ir_datasets provider, then audit it."""

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pytest

from evidence_rag.cli.build_sealed600 import main as build_main
from evidence_rag.cli.gate0a import main as gate0a_main
from evidence_rag.cli.pin_candidates import main as pin_main
from evidence_rag.contracts.models import (
    CandidateSet,
    Document,
    EvidenceCandidate,
    Query,
    RetrieverProvenance,
)
from evidence_rag.infrastructure.datasets import DatasetManifest, GoldCase
from evidence_rag.materializer.answer_bank import build_answer_bank
from evidence_rag.materializer.provenance import (
    MutationRecord,
    read_provenance,
    write_provenance,
)
from evidence_rag.materializer.sealed600 import (
    FROZEN_RETRIEVER,
    read_candidate_pin,
    read_sealed_manifest,
    sha256_file,
)

BM25 = RetrieverProvenance(
    name=FROZEN_RETRIEVER, implementation_version="bm25-v1", parameters_sha256="1" * 64
)
STRONG_BM25 = RetrieverProvenance(
    name="strong-bm25", implementation_version="strong-bm25-v1", parameters_sha256="2" * 64
)


@dataclass(frozen=True)
class FakeDoc:
    doc_id: str
    title: str
    text: str


@dataclass(frozen=True)
class FakeQuery:
    query_id: str
    text: str
    answers: tuple[str, ...]


@dataclass(frozen=True)
class FakeQrel:
    query_id: str
    doc_id: str
    relevance: int


class FakeDataset:
    """Ten fresh questions, one gold passage each, plus distractors."""

    def __init__(self) -> None:
        self.docs = [
            FakeDoc(f"d{index}", f"Article {index}", f"The event happened in 19{index:02d}.")
            for index in range(10)
        ]
        self.docs += [FakeDoc(f"x{index}", f"Filler {index}", "Nothing here.") for index in range(5)]

    def docs_iter(self) -> Iterable[FakeDoc]:
        return iter(self.docs)

    def queries_iter(self) -> Iterable[FakeQuery]:
        return iter(
            [
                FakeQuery(f"q{index}", f"When did event {index} happen?", (f"19{index:02d}",))
                for index in range(10)
            ]
        )

    def qrels_iter(self) -> Iterable[FakeQrel]:
        return iter([FakeQrel(f"q{index}", f"d{index}", 1) for index in range(10)])


class FakeProvider:
    def load(self, dataset_id: str) -> FakeDataset:
        assert dataset_id == "dpr-w100/natural-questions/dev"
        return FakeDataset()


def write_train_split(directory: Path) -> Path:
    """A prior split: it supplies the frozen answer bank AND one axis-collision to prove the
    filter runs (q0 is already used there)."""
    directory.mkdir(parents=True, exist_ok=True)
    manifest = DatasetManifest(
        dataset_id="niah/dpr-w100-nq",
        dataset_version="train",
        split="train",
        documents_file="documents.jsonl",
        queries_file="queries.jsonl",
        gold_cases_file="gold_cases.jsonl",
    )
    (directory / "manifest.json").write_text(manifest.model_dump_json(), encoding="utf-8")
    (directory / "documents.jsonl").write_text(
        Document(
            document_id="t1", text="Prior Article\n\nA prior fact from 1500.", source_uri="x://t1"
        ).model_dump_json()
        + "\n",
        encoding="utf-8",
    )
    (directory / "queries.jsonl").write_text(
        Query(query_id="q0", text="A prior question?").model_dump_json() + "\n", encoding="utf-8"
    )
    (directory / "gold_cases.jsonl").write_text(
        GoldCase(
            query_id="q0", relevant_document_ids=("t1",), reference_answers=("1500", "1600")
        ).model_dump_json()
        + "\n",
        encoding="utf-8",
    )
    # A real mutation record, not an empty log: the family axis refuses to call an empty
    # comparison set "zero overlap", so a prior split with no mutations cannot be audited
    # against. This one uses a contrast the sealed set cannot reproduce (its golds are 19xx).
    write_provenance(
        directory / "provenance.jsonl",
        (
            MutationRecord(
                query_id="q0",
                needle_document_id="t1",
                counterfactual_document_id="cf::q0::t1",
                gold_value="1500",
                gold_alias_used="1500",
                replacement_value="1600",
                string_class="year",
                seed=42,
                char_span=(0, 4),
                text_hash_before="a" * 64,
                text_hash_after="b" * 64,
                # The real hash of the bank this split's answers produce: the builder rebuilds
                # the bank and refuses to continue unless it reproduces this value.
                answer_bank_hash=build_answer_bank(["1500", "1600"], seed=42).content_hash,
            ),
        ),
    )
    return directory


def build(tmp_path: Path, **extra: str) -> tuple[Path, Path, Path]:
    train = write_train_split(tmp_path / "train")
    protocol = tmp_path / "M0.md"
    protocol.write_text("# protocol text\n", encoding="utf-8")
    output = tmp_path / "sealed"
    argv = [
        "--split", "dev",
        "--output", str(output),
        "--train-manifest", str(train / "manifest.json"),
        "--existing-split", str(train),
        "--protocol-doc", str(protocol),
        "--corpus-size", "12",
        "--seed", "42",
        "--pool-size", extra.get("pool_size", "5"),
        "--target", extra.get("target", "3"),
    ]
    assert build_main(argv, provider=FakeProvider()) == 0
    return output, train, protocol


def test_the_builder_writes_a_sealed_set_and_freezes_its_manifest(tmp_path: Path) -> None:
    output, _train, _protocol = build(tmp_path)
    manifest = read_sealed_manifest(output)
    assert len(manifest.query_ids) == 3
    assert len(manifest.pool_query_ids) == 5
    assert "q0" not in manifest.pool_query_ids  # the prior split already used that query id
    assert set(manifest.artifact_sha256) == {
        "manifest.json",
        "documents.jsonl",
        "queries.jsonl",
        "gold_cases.jsonl",
        "provenance.jsonl",
        "source_parent.jsonl",
    }
    assert len(read_provenance(output / "provenance.jsonl")) == 3


def test_the_replacement_values_come_from_the_train_bank_not_from_the_sealed_answers(
    tmp_path: Path,
) -> None:
    """TRAINING_PLAN §4.2 pins the bank to NIAH train. A self-built bank would draw one sealed
    query's counterfactual from another sealed query's gold answer, so the poison for question A
    would be the true answer to question B — inside the same corpus the selector searches."""
    output, _train, _protocol = build(tmp_path)
    replacements = {record.replacement_value for record in read_provenance(output / "provenance.jsonl")}
    assert replacements <= {"1500", "1600"}


def test_a_second_build_into_the_same_directory_is_refused(tmp_path: Path) -> None:
    output, train, protocol = build(tmp_path)
    with pytest.raises(ValueError, match="already frozen"):
        build_main(
            [
                "--split", "dev",
                "--output", str(output),
                "--train-manifest", str(train / "manifest.json"),
                "--existing-split", str(train),
                "--protocol-doc", str(protocol),
                "--corpus-size", "12", "--seed", "42", "--pool-size", "5", "--target", "3",
            ],
            provider=FakeProvider(),
        )


def test_a_pool_without_headroom_is_refused_before_the_corpus_is_streamed(tmp_path: Path) -> None:
    train = write_train_split(tmp_path / "train")
    protocol = tmp_path / "M0.md"
    protocol.write_text("# protocol\n", encoding="utf-8")
    with pytest.raises(ValueError, match="headroom"):
        build_main(
            [
                "--split", "dev",
                "--output", str(tmp_path / "sealed"),
                "--train-manifest", str(train / "manifest.json"),
                "--existing-split", str(train),
                "--protocol-doc", str(protocol),
                "--pool-size", "3", "--target", "3",
            ],
            provider=FakeProvider(),
        )
    assert not (tmp_path / "sealed").exists()


def test_the_audit_of_a_fresh_set_is_incomplete_until_retrieval_has_run(tmp_path: Path) -> None:
    output, train, protocol = build(tmp_path)
    exit_code = gate0a_main(
        [
            "--sealed", str(output),
            "--existing-split", str(train),
            "--protocol-doc", str(protocol),
            "--no-candidates",
            "--utility-labels-absent",
            "--output", str(tmp_path / "gate0a.json"),
        ]
    )
    assert exit_code == 1
    report = (tmp_path / "gate0a.json").read_text(encoding="utf-8")
    assert '"verdict": "INCOMPLETE"' in report
    assert "utility_range_and_derived_flags" in report


def test_the_audit_refuses_to_run_without_the_utility_acknowledgement(tmp_path: Path) -> None:
    output, train, protocol = build(tmp_path)
    with pytest.raises(SystemExit):
        gate0a_main(
            [
                "--sealed", str(output),
                "--existing-split", str(train),
                "--protocol-doc", str(protocol),
                "--no-candidates",
            ]
        )


def test_the_audit_fails_when_the_protocol_document_changed_after_the_freeze(
    tmp_path: Path,
) -> None:
    output, train, protocol = build(tmp_path)
    protocol.write_text("# protocol text, amended\n", encoding="utf-8")
    assert (
        gate0a_main(
            [
                "--sealed", str(output),
                "--existing-split", str(train),
                "--protocol-doc", str(protocol),
                "--no-candidates",
                "--utility-labels-absent",
            ]
        )
        == 1
    )


# ------------------------------------------------------- retrieval, pinning, and PASS (M0 §4)


def write_candidates(
    path: Path,
    sealed: Path,
    *,
    retriever: RetrieverProvenance | None = BM25,
    twin_first: bool = False,
) -> Path:
    """A Top-2 pool over the sealed set: each query's own needle and its own twin.

    Written by hand rather than by running BM25, because what is under test is the audit's
    treatment of the pool's RECORDED producer, not the ranking that produced it. `twin_first`
    flips the ranking to stand for a second, equally legitimate run of the same retriever.
    """
    windows = []
    for record in read_provenance(sealed / "provenance.jsonl"):
        document_ids = (record.needle_document_id, record.counterfactual_document_id)
        if twin_first:
            document_ids = tuple(reversed(document_ids))
        windows.append(
            CandidateSet(
                query_id=record.query_id,
                retriever=retriever,
                candidates=tuple(
                    EvidenceCandidate(
                        evidence_id=f"{record.query_id}:{document_id}",
                        document_id=document_id,
                        chunk_id=document_id,
                        text="text",
                        source_uri=f"fixture://{document_id}",
                        retrieval_score=1.0 / (rank + 1),
                        retrieval_rank=rank + 1,
                    )
                    for rank, document_id in enumerate(document_ids)
                ),
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(window.model_dump_json() + "\n" for window in windows), encoding="utf-8"
    )
    return path


def gate0a(sealed: Path, train: Path, protocol: Path, *extra: str) -> int:
    return gate0a_main(
        [
            "--sealed", str(sealed),
            "--existing-split", str(train),
            "--protocol-doc", str(protocol),
            "--utility-labels-absent",
            *extra,
        ]
    )


def test_a_pinned_bm25_pool_takes_the_verdict_from_incomplete_to_pass(tmp_path: Path) -> None:
    """The whole point of the mechanism, end to end. Before retrieval the honest verdict is
    INCOMPLETE; it becomes PASS only once the pool names bm25 AND its bytes match the pin."""
    output, train, protocol = build(tmp_path)
    candidates = write_candidates(tmp_path / "bm25" / "candidate_sets.jsonl", output)

    assert gate0a(output, train, protocol, "--candidates", str(candidates), "--top-n", "2") == 1

    assert pin_main(["--sealed", str(output), "--candidates", str(candidates), "--top-n", "2"]) == 0
    pin = read_candidate_pin(output)
    assert pin is not None
    assert pin.retriever == BM25
    assert pin.candidate_sha256 == sha256_file(candidates)

    assert gate0a(output, train, protocol, "--candidates", str(candidates), "--top-n", "2") == 0


def test_a_pool_that_changed_after_pinning_takes_the_verdict_back_to_fail(
    tmp_path: Path,
) -> None:
    output, train, protocol = build(tmp_path)
    candidates = write_candidates(tmp_path / "bm25" / "candidate_sets.jsonl", output)
    assert pin_main(["--sealed", str(output), "--candidates", str(candidates), "--top-n", "2"]) == 0

    # Same retriever, re-run: the windows are legitimate, they are simply not the ones every
    # pre-registered threshold in §3.5 and §5.2 was measured against.
    write_candidates(candidates, output, twin_first=True)
    assert gate0a(output, train, protocol, "--candidates", str(candidates), "--top-n", "2") == 1


def test_the_pin_refuses_a_pool_from_a_retriever_other_than_the_frozen_one(
    tmp_path: Path,
) -> None:
    """M0 §4's freeze, enforced at the moment the pool would be admitted rather than argued
    about afterwards."""
    output, _train, _protocol = build(tmp_path)
    candidates = write_candidates(
        tmp_path / "strong" / "candidate_sets.jsonl", output, retriever=STRONG_BM25
    )
    with pytest.raises(ValueError, match="strong-bm25"):
        pin_main(["--sealed", str(output), "--candidates", str(candidates), "--top-n", "2"])
    assert read_candidate_pin(output) is None


def test_the_pin_refuses_a_pool_that_names_no_retriever_at_all(tmp_path: Path) -> None:
    """The pin must not become the place where "it was probably bm25" gets written down. An old
    provenance-less pool has to be re-retrieved, not re-labelled."""
    output, _train, _protocol = build(tmp_path)
    candidates = write_candidates(
        tmp_path / "old" / "candidate_sets.jsonl", output, retriever=None
    )
    with pytest.raises(ValueError, match="names no retriever"):
        pin_main(["--sealed", str(output), "--candidates", str(candidates), "--top-n", "2"])
    assert read_candidate_pin(output) is None


def test_the_pin_refuses_a_pool_that_is_not_over_the_sealed_queries(tmp_path: Path) -> None:
    output, _train, _protocol = build(tmp_path)
    candidates = write_candidates(tmp_path / "bm25" / "candidate_sets.jsonl", output)
    lines = candidates.read_text(encoding="utf-8").splitlines(keepends=True)
    candidates.write_text("".join(lines[:-1]), encoding="utf-8")
    with pytest.raises(ValueError, match="sealed"):
        pin_main(["--sealed", str(output), "--candidates", str(candidates), "--top-n", "2"])


def test_declaring_that_retrieval_has_not_run_is_refused_once_a_pin_exists(
    tmp_path: Path,
) -> None:
    """`--no-candidates` is a statement of fact, not a way to skip an item. With a pin on disk
    the statement is false, and letting it through would turn the one check that verifies pool
    provenance into an opt-out."""
    output, train, protocol = build(tmp_path)
    candidates = write_candidates(tmp_path / "bm25" / "candidate_sets.jsonl", output)
    assert pin_main(["--sealed", str(output), "--candidates", str(candidates), "--top-n", "2"]) == 0
    with pytest.raises(SystemExit):
        gate0a(output, train, protocol, "--no-candidates")
