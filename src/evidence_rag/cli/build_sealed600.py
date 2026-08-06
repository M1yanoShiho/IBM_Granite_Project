"""CLI: build and freeze the fresh NIAH sealed 600 (M0 §4, TRAINING_PLAN §4.2).

ONE streaming pass over dpr-w100 (21M passages), then everything else in memory. Run it on a
login node; it is pure CPU.

    export PYTHONPATH=src IR_DATASETS_HOME=/user/work/$USER/ir_datasets
    python -m evidence_rag.cli.build_sealed600 \
      --split dev \
      --output runs/niah-sealed600 \
      --train-manifest runs/niah-train-injected/manifest.json \
      --existing-split runs/niah-injected \
      --existing-split runs/niah-train-injected \
      --protocol-doc docs/selector/M0_PROTOCOL_FREEZE.md \
      --corpus-size 100000 --seed 42

--split has NO default. Which pool the fresh queries come from is a protocol decision (the
unused remainder of NQ dev, or the train pool), it is recorded in the manifest, and a default
would let it be made by accident.

--train-manifest is REQUIRED and is not the set being built. TRAINING_PLAN §4.2 pins
replacements to the seed=42 NIAH-train answer bank; building the bank from the sealed set's own
answers — which is what `materializer/cli.py` correctly does for train and dev — would let one
sealed query's counterfactual assert another sealed query's gold answer.

--existing-split takes INJECTED directories (`runs/niah-*-injected`), because axis 5 compares
synthetic families and those live in `provenance.jsonl`. A pre-injection directory has no such
file and the builder refuses it rather than comparing against an empty family set.

The build writes nothing until every guard has passed, and `sealed_manifest.json` cannot be
overwritten. A second run into the same directory is refused: that is what a top-up would look
like, and the protocol calls a top-up "看结果后改数据".
"""

import argparse
import importlib
import json
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from evidence_rag.contracts.models import Document, Query
from evidence_rag.infrastructure.datasets import (
    GoldCase,
    JsonlDatasetAdapter,
    normalize_document,
    normalize_query,
)
from evidence_rag.materializer.answer_bank import build_answer_bank
from evidence_rag.materializer.base_loader import (
    DOCUMENT_SOURCE_URI,
    BaseDataset,
    document_text,
)
from evidence_rag.materializer.provenance import read_provenance
from evidence_rag.materializer.sealed600 import (
    DEV_SAMPLE_INJECTED,
    DEV_SAMPLE_QUERIES,
    DEV_SKIP_RATE,
    POOL_SIZE,
    PROTOCOL_VERSION,
    SEALED_DATASET_ID,
    SEALED_SPLIT,
    TARGET_INJECTED,
    SealedManifest,
    SourceQuery,
    assert_pool_headroom,
    freeze_sealed_manifest,
    frozen_order,
    inject_pool,
    read_split_fingerprint,
    select_pool,
    sha256_file,
    write_sealed_dataset,
)


class BaseProvider(Protocol):
    def load(self, dataset_id: str) -> BaseDataset: ...


class _IrDatasetsProvider:
    def load(self, dataset_id: str) -> BaseDataset:
        dataset: BaseDataset = importlib.import_module("ir_datasets").load(dataset_id)
        return dataset


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and freeze the NIAH sealed 600")
    parser.add_argument("--split", required=True, help="dpr-w100/natural-questions split to draw from")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--train-manifest",
        required=True,
        type=Path,
        help="NIAH train manifest; the frozen replacement bank is built from ITS answers",
    )
    parser.add_argument(
        "--existing-split",
        required=True,
        action="append",
        type=Path,
        dest="existing_splits",
        help="an injected split directory to audit against; repeatable",
    )
    parser.add_argument("--protocol-doc", required=True, type=Path)
    parser.add_argument("--corpus-size", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pool-size", type=int, default=POOL_SIZE)
    parser.add_argument("--target", type=int, default=TARGET_INJECTED)
    return parser


def _reservoir(
    dataset: BaseDataset, gold_document_ids: set[str], *, size: int, seed: int
) -> tuple[dict[str, Document], list[Document]]:
    """One pass: keep every gold passage, reservoir-sample the rest.

    base_loader has its own copy of Algorithm R and they are deliberately not shared. It knows
    its distractor budget before it starts, because its query set is fixed up front; here the
    budget depends on how many gold passages the sealed 600 turn out to need, which is not known
    until selection and injection have run. So this over-samples to `size` and trims afterwards.
    """
    rng = random.Random(seed)
    gold: dict[str, Document] = {}
    reservoir: list[Document] = []
    seen = 0
    for source_doc in dataset.docs_iter():
        document = normalize_document(
            Document(
                document_id=source_doc.doc_id,
                text=document_text(source_doc),
                source_uri=DOCUMENT_SOURCE_URI.format(document_id=source_doc.doc_id),
            )
        )
        if document.document_id in gold_document_ids:
            gold[document.document_id] = document
            continue
        seen += 1
        if len(reservoir) < size:
            reservoir.append(document)
        else:
            index = rng.randrange(seen)
            if index < size:
                reservoir[index] = document
    return gold, reservoir


def main(argv: Sequence[str] | None = None, *, provider: BaseProvider | None = None) -> int:
    arguments = _parser().parse_args(argv)
    # Before anything is read: a pool without headroom cannot be rescued by luck, and finding
    # that out after a 21M-passage pass is finding it out at the worst possible moment.
    required = assert_pool_headroom(
        pool_size=arguments.pool_size, target=arguments.target, skip_rate=DEV_SKIP_RATE
    )
    protocol_hash = sha256_file(arguments.protocol_doc)
    existing = tuple(read_split_fingerprint(path) for path in arguments.existing_splits)

    train = JsonlDatasetAdapter.load(arguments.train_manifest)
    bank = build_answer_bank(
        [
            answer
            for gold_case in train.gold_cases
            for answer in (gold_case.reference_answers or ())
        ],
        seed=arguments.seed,
    )
    # "seed=42 冻结的 NIAH-train answer bank" names a bank that already exists on disk, and the
    # train mutation log records its hash on every record. Rebuilding it from the same manifest
    # must reproduce that hash: if it does not, this build is drawing replacements from a
    # DIFFERENT bank than the one the protocol froze, and nothing downstream would show it —
    # every replacement value would still be a plausible same-class string.
    train_log = arguments.train_manifest.parent / "provenance.jsonl"
    if not train_log.is_file():
        raise ValueError(
            f"{train_log} does not exist. --train-manifest must point at the INJECTED train "
            "directory, whose mutation log identifies the frozen answer bank (TRAINING_PLAN §4.2)."
        )
    recorded = {record.answer_bank_hash for record in read_provenance(train_log)}
    if recorded != {bank.content_hash}:
        raise ValueError(
            f"the bank rebuilt from {arguments.train_manifest} hashes to {bank.content_hash}, "
            f"but the train mutation log was written with {sorted(recorded)}. The sealed set's "
            "replacements would come from a bank nobody can identify."
        )

    active = provider if provider is not None else _IrDatasetsProvider()
    dataset = active.load(f"dpr-w100/natural-questions/{arguments.split}")

    query_by_id: dict[str, Query] = {}
    answers_by_id: dict[str, tuple[str, ...]] = {}
    for source_query in dataset.queries_iter():
        query = normalize_query(Query(query_id=source_query.query_id, text=source_query.text))
        query_by_id[query.query_id] = query
        raw = tuple(getattr(source_query, "answers", ()) or ())
        answers_by_id[query.query_id] = tuple(
            sorted({answer.strip() for answer in raw if answer.strip()})
        )

    positives: dict[str, set[str]] = {}
    for qrel in dataset.qrels_iter():
        if int(qrel.relevance) > 0 and qrel.query_id in query_by_id:
            positives.setdefault(qrel.query_id, set()).add(qrel.doc_id)

    gold_document_ids = {doc_id for ids in positives.values() for doc_id in ids}
    gold_documents, reservoir = _reservoir(
        dataset, gold_document_ids, size=arguments.corpus_size, seed=arguments.seed
    )

    candidates = tuple(
        SourceQuery(
            query_id=query_id,
            text=query_by_id[query_id].text,
            answers=answers_by_id[query_id],
            gold_document_ids=tuple(sorted(positives[query_id])),
        )
        for query_id in sorted(positives)
    )
    pool = select_pool(
        candidates=candidates,
        documents=gold_documents,
        existing=existing,
        seed=arguments.seed,
        pool_size=arguments.pool_size,
    )
    injected = inject_pool(
        pool_query_ids=pool.pool_query_ids,
        candidates_by_id={item.query_id: item for item in candidates},
        documents=gold_documents,
        bank=bank,
        existing=existing,
        seed=arguments.seed,
        target=arguments.target,
    )

    sealed = {query_id: True for query_id in injected.query_ids}
    sealed_gold_ids = sorted(
        {
            document_id
            for query_id in injected.query_ids
            for document_id in positives[query_id]
            if document_id in gold_documents
        }
    )
    budget = arguments.corpus_size - len(sealed_gold_ids)
    if budget < 0:
        raise ValueError(
            f"the sealed set needs {len(sealed_gold_ids)} gold passages, which does not fit a "
            f"corpus of {arguments.corpus_size}: raise --corpus-size rather than dropping gold"
        )
    # Non-sealed gold passages rejoin the distractor pool: routing them away during the pass was
    # bookkeeping, and leaving them out would make the sealed corpus systematically free of other
    # questions' answers — a pool that is easier than the dev pool it is compared against.
    distractor_ids = [document.document_id for document in reservoir] + [
        document_id for document_id in gold_documents if document_id not in set(sealed_gold_ids)
    ]
    by_id = {document.document_id: document for document in reservoir} | gold_documents
    chosen = frozen_order(distractor_ids, seed=arguments.seed)[:budget]

    documents = tuple(
        by_id[document_id] for document_id in sorted(set(sealed_gold_ids) | set(chosen))
    ) + injected.twins
    queries = tuple(query_by_id[query_id] for query_id in injected.query_ids)
    gold_cases = tuple(
        GoldCase(
            query_id=query_id,
            relevant_document_ids=tuple(sorted(positives[query_id])),
            reference_answers=answers_by_id[query_id],
        )
        for query_id in injected.query_ids
    )
    hashes = write_sealed_dataset(
        arguments.output,
        documents=documents,
        queries=queries,
        gold_cases=gold_cases,
        records=injected.records,
        seed=arguments.seed,
    )
    manifest = SealedManifest(
        protocol_version=PROTOCOL_VERSION,
        protocol_document_sha256=protocol_hash,
        dataset_id=SEALED_DATASET_ID,
        dataset_version=f"sealed600-seed{arguments.seed}",
        split=SEALED_SPLIT,
        source_dataset_id=f"dpr-w100/natural-questions/{arguments.split}",
        seed=arguments.seed,
        corpus_size=arguments.corpus_size,
        target_injected=arguments.target,
        dev_sample_queries=DEV_SAMPLE_QUERIES,
        dev_sample_injected=DEV_SAMPLE_INJECTED,
        dev_skip_rate=DEV_SKIP_RATE,
        required_pool_size=required,
        pool_size=arguments.pool_size,
        n_considered=pool.n_considered,
        n_injected=len(injected.query_ids),
        answer_bank_hash=bank.content_hash,
        answer_bank_source=str(arguments.train_manifest),
        existing_splits=tuple(str(path) for path in arguments.existing_splits),
        label_provenance={
            "gold_cases.jsonl": "official",
            "provenance.jsonl": "deterministic_rule",
        },
        pool_rejections=dict(pool.rejections),
        injection_rejections=dict(injected.rejections),
        pool_query_ids=pool.pool_query_ids,
        query_ids=injected.query_ids,
        artifact_sha256=hashes,
    )
    freeze_sealed_manifest(arguments.output, manifest)
    print(
        json.dumps(
            {
                "sealed": len(sealed),
                "pool": len(pool.pool_query_ids),
                "considered": pool.n_considered,
                "attempted": injected.n_attempted,
                "realised_skip_rate": round(
                    1 - len(injected.query_ids) / injected.n_attempted, 4
                )
                if injected.n_attempted
                else None,
                "protocol_skip_rate": DEV_SKIP_RATE,
                "documents": len(documents),
                "pool_rejections": dict(pool.rejections),
                "injection_rejections": dict(injected.rejections),
                "manifest": str(arguments.output / "sealed_manifest.json"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
