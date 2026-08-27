"""Reproducible full-corpus indexing inputs for Experiment 05."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import random
import sys
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

CORPUS_SCHEMA_VERSION = "experiment05.corpus_manifest.v1"
DENSE_INDEX_SCHEMA_VERSION = "experiment05.dense_index_manifest.v1"
BM25_INDEX_SCHEMA_VERSION = "experiment05.bm25_index_manifest.v1"


@dataclass(frozen=True)
class DenseIndexParameters:
    dimension: int = 768
    nlist: int = 8192
    m: int = 96
    nbits: int = 8
    nprobe: int = 64
    train_size: int = 327_680
    batch_size: int = 512
    checkpoint_interval: int = 1_000_000
    seed: int = 13

    def validate(self) -> None:
        positive = (
            self.dimension,
            self.nlist,
            self.m,
            self.nbits,
            self.nprobe,
            self.train_size,
            self.batch_size,
            self.checkpoint_interval,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("dense index parameters must be positive")
        if self.dimension % self.m:
            raise ValueError("dense embedding dimension must be divisible by PQ m")
        if self.nprobe > self.nlist:
            raise ValueError("dense nprobe cannot exceed nlist")


def _clean(text: object) -> str:
    return " ".join(str(text).split())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _passage(
    *,
    passage_id: str,
    title: str,
    body: str,
    source_kind: str,
    source_id: str,
    start_unit: int | None,
    end_unit: int | None,
) -> dict[str, Any]:
    contents = f"{title}\n{body}" if title else body
    return {
        "id": passage_id,
        "contents": contents,
        "title": title,
        "source_kind": source_kind,
        "source_id": source_id,
        "start_unit": start_unit,
        "end_unit": end_unit,
    }


def _page_units(
    paragraphs: Sequence[object], *, max_words: int
) -> Iterator[tuple[str, int, int]]:
    current_words: list[str] = []
    current_start: int | None = None
    current_end: int | None = None

    def flush() -> tuple[str, int, int] | None:
        nonlocal current_words, current_start, current_end
        if not current_words or current_start is None or current_end is None:
            return None
        value = (" ".join(current_words), current_start, current_end)
        current_words = []
        current_start = None
        current_end = None
        return value

    for paragraph_index, raw_paragraph in enumerate(paragraphs, start=1):
        words = _clean(raw_paragraph).split()
        if not words:
            continue
        if len(words) > max_words:
            pending = flush()
            if pending is not None:
                yield pending
            for offset in range(0, len(words), max_words):
                yield " ".join(words[offset : offset + max_words]), paragraph_index, paragraph_index
            continue
        if current_words and len(current_words) + len(words) > max_words:
            pending = flush()
            if pending is not None:
                yield pending
        if current_start is None:
            current_start = paragraph_index
        current_end = paragraph_index
        current_words.extend(words)

    pending = flush()
    if pending is not None:
        yield pending


def iter_kilt_passages(path: Path, *, max_words: int = 100) -> Iterator[dict[str, Any]]:
    if max_words <= 0:
        raise ValueError("max_words must be positive")
    passage_index = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            page = json.loads(line)
            if not isinstance(page, Mapping):
                raise ValueError(f"KILT page line {line_number} is not an object")
            source_id = _clean(page.get("wikipedia_id"))
            title = _clean(page.get("wikipedia_title"))
            paragraphs = page.get("text")
            if not source_id or not title or not isinstance(paragraphs, list):
                raise ValueError(f"KILT page line {line_number} has invalid provenance fields")
            body_paragraphs = paragraphs[1:] if paragraphs and _clean(paragraphs[0]) == title else paragraphs
            for body, start, end in _page_units(body_paragraphs, max_words=max_words):
                yield _passage(
                    passage_id=f"kilt-{passage_index:09d}",
                    title=title,
                    body=body,
                    source_kind="kilt-paragraph",
                    source_id=source_id,
                    start_unit=start,
                    end_unit=end,
                )
                passage_index += 1


def iter_dpr_passages(path: Path) -> Iterator[dict[str, Any]]:
    csv.field_size_limit(1 << 30)
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != ["id", "text", "title"]:
            raise ValueError("DPR passage header differs from frozen id/text/title schema")
        for passage_index, raw in enumerate(reader):
            line_number = passage_index + 2
            source_id = _clean(raw.get("id"))
            title = _clean(raw.get("title"))
            body = _clean(raw.get("text"))
            if not source_id or not body:
                raise ValueError(f"DPR passage line {line_number} has blank ID or text")
            yield _passage(
                passage_id=f"dpr-{passage_index:09d}",
                title=title,
                body=body,
                source_kind="dpr-passage",
                source_id=source_id,
                start_unit=None,
                end_unit=None,
            )


def _canonical_line(row: Mapping[str, Any]) -> bytes:
    payload = dict(row)
    contents = str(payload["contents"])
    payload["contents_sha256"] = hashlib.sha256(contents.encode("utf-8")).hexdigest()
    return json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8") + b"\n"


def read_corpus_manifest(corpus_dir: Path) -> dict[str, Any]:
    manifest_path = corpus_dir / "corpus_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("corpus manifest must be a JSON object")
    if manifest.get("schema_version") != CORPUS_SCHEMA_VERSION:
        raise ValueError("corpus manifest schema differs from frozen v1")
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("corpus manifest contains no shards")
    observed_count = 0
    for shard in shards:
        if not isinstance(shard, Mapping):
            raise ValueError("corpus manifest has a malformed shard row")
        path = corpus_dir / str(shard["file"])
        if not path.is_file() or _sha256(path) != shard.get("sha256"):
            raise ValueError(f"corpus shard hash differs: {path}")
        observed_count += int(shard["count"])
    if observed_count != manifest.get("passage_count"):
        raise ValueError("corpus manifest passage count differs from shard counts")
    return manifest


def _rows_from_manifest(corpus_dir: Path, manifest: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
    prefix = "kilt" if manifest["source_kind"] == "kilt" else "dpr"
    expected_keys = {
        "contents",
        "contents_sha256",
        "end_unit",
        "id",
        "source_id",
        "source_kind",
        "start_unit",
        "title",
    }
    ordinal = 0
    ordered_ids_digest = hashlib.sha256()
    for shard in manifest["shards"]:
        with (corpus_dir / str(shard["file"])).open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict) or set(row) != expected_keys:
                    raise ValueError(f"normalized corpus row {line_number} has invalid schema")
                expected_id = f"{prefix}-{ordinal:09d}"
                if row["id"] != expected_id:
                    raise ValueError("normalized corpus passage IDs are not ordinal and contiguous")
                contents = str(row["contents"])
                if hashlib.sha256(contents.encode("utf-8")).hexdigest() != row["contents_sha256"]:
                    raise ValueError("normalized corpus contents hash differs")
                if ordinal:
                    ordered_ids_digest.update(b"\n")
                ordered_ids_digest.update(expected_id.encode("utf-8"))
                ordinal += 1
                yield row
    if ordinal != manifest["passage_count"]:
        raise ValueError("normalized corpus yielded a different passage count")
    if ordered_ids_digest.hexdigest() != manifest["ordered_passage_ids_sha256"]:
        raise ValueError("normalized corpus ordered passage ID hash differs")


def corpus_rows(corpus_dir: Path) -> Iterator[dict[str, Any]]:
    manifest = read_corpus_manifest(corpus_dir)
    yield from _rows_from_manifest(corpus_dir, manifest)


def training_ordinals(passage_count: int, train_size: int, *, seed: int) -> tuple[int, ...]:
    if passage_count <= 0 or train_size <= 0:
        raise ValueError("passage_count and train_size must be positive")
    if train_size >= passage_count:
        return tuple(range(passage_count))
    offset = random.Random(seed).random()
    ordinals = tuple(
        int((index + offset) * passage_count / train_size) for index in range(train_size)
    )
    if len(ordinals) != len(set(ordinals)):
        raise ValueError("stratified dense training ordinals are not unique")
    return ordinals


def _model_files(snapshot: Path) -> tuple[dict[str, object], ...]:
    if not snapshot.is_dir():
        return ()
    files: list[dict[str, object]] = []
    for path in sorted(candidate for candidate in snapshot.rglob("*") if candidate.is_file()):
        files.append(
            {
                "file": str(path.relative_to(snapshot)),
                "size": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return tuple(files)


def _object_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _encoded(encoder: Any, texts: Sequence[str], *, batch_size: int, dimension: int) -> Any:
    import numpy as np

    vectors = np.asarray(
        encoder.encode(
            list(texts),
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ),
        dtype="float32",
    )
    if vectors.shape != (len(texts), dimension):
        raise ValueError(
            f"dense encoder returned shape {vectors.shape}, expected {(len(texts), dimension)}"
        )
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("dense encoder returned a zero vector")
    return np.ascontiguousarray(vectors / norms, dtype="float32")


def _write_state(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def build_dense_index(
    *,
    corpus_dir: Path,
    index_dir: Path,
    model_snapshot: Path,
    parameters: DenseIndexParameters | None = None,
    device: str = "cuda",
    encoder: Any | None = None,
    faiss_module: Any | None = None,
) -> dict[str, Any]:
    parameters = parameters or DenseIndexParameters()
    parameters.validate()
    manifest = read_corpus_manifest(corpus_dir)
    passage_count = int(manifest["passage_count"])
    if parameters.train_size < parameters.nlist:
        raise ValueError("dense train_size must be at least nlist")
    model_files = _model_files(model_snapshot)
    if encoder is None:
        if not model_files:
            raise ValueError("dense model snapshot is missing")
        from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]

        encoder = SentenceTransformer(str(model_snapshot), device=device)
    if faiss_module is None:
        import faiss as imported_faiss  # type: ignore[import-not-found]

        faiss_module = imported_faiss
    assert faiss_module is not None

    parameter_payload = asdict(parameters)
    parameter_payload["metric"] = "inner_product_on_l2_normalized_vectors"
    corpus_manifest_sha256 = _sha256(corpus_dir / "corpus_manifest.json")
    model_snapshot_sha256 = _object_sha256(model_files)
    build_signature = _object_sha256(
        {
            "corpus_manifest_sha256": corpus_manifest_sha256,
            "model_snapshot_sha256": model_snapshot_sha256,
            "parameters": parameter_payload,
        }
    )
    index_dir.mkdir(parents=True, exist_ok=True)
    partial_path = index_dir / "dense.partial.faiss"
    state_path = index_dir / "dense_build_state.json"
    state: dict[str, Any] | None = None
    index: Any

    if partial_path.is_file() and state_path.is_file() and hasattr(faiss_module, "read_index"):
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("build_signature") != build_signature:
            raise ValueError("dense partial index signature differs from frozen build")
        index = faiss_module.read_index(str(partial_path))
        if int(index.ntotal) != int(state["faiss_ntotal"]):
            raise ValueError("dense partial index and build state disagree")
    else:
        train_ordinals = set(
            training_ordinals(passage_count, parameters.train_size, seed=parameters.seed)
        )
        train_texts = [
            str(row["contents"])
            for ordinal, row in enumerate(_rows_from_manifest(corpus_dir, manifest))
            if ordinal in train_ordinals
        ]
        if len(train_texts) != min(passage_count, parameters.train_size):
            raise ValueError("dense training sample count differs from frozen selection")
        train_batches = [
            _encoded(
                encoder,
                train_texts[offset : offset + parameters.batch_size],
                batch_size=parameters.batch_size,
                dimension=parameters.dimension,
            )
            for offset in range(0, len(train_texts), parameters.batch_size)
        ]
        import numpy as np

        training_vectors = np.concatenate(train_batches, axis=0)
        quantizer = faiss_module.IndexFlatIP(parameters.dimension)
        index = faiss_module.IndexIVFPQ(
            quantizer,
            parameters.dimension,
            parameters.nlist,
            parameters.m,
            parameters.nbits,
            faiss_module.METRIC_INNER_PRODUCT,
        )
        if hasattr(index, "cp"):
            index.cp.seed = parameters.seed
            index.cp.niter = 25
        if hasattr(index, "pq") and hasattr(index.pq, "cp"):
            index.pq.cp.seed = parameters.seed
            index.pq.cp.niter = 25
        index.train(training_vectors)
        if not index.is_trained:
            raise ValueError("FAISS dense index did not become trained")
        faiss_module.write_index(index, str(partial_path))
        state = {
            "schema_version": "experiment05.dense_build_state.v1",
            "build_signature": build_signature,
            "faiss_ntotal": 0,
            "trained": True,
        }
        _write_state(state_path, state)

    starting_total = int(index.ntotal)
    if starting_total > passage_count:
        raise ValueError("dense partial index contains more rows than the corpus")
    batch_texts: list[str] = []
    next_checkpoint = (
        (starting_total // parameters.checkpoint_interval) + 1
    ) * parameters.checkpoint_interval
    for ordinal, row in enumerate(_rows_from_manifest(corpus_dir, manifest)):
        if ordinal < starting_total:
            continue
        batch_texts.append(str(row["contents"]))
        if len(batch_texts) < parameters.batch_size and ordinal + 1 < passage_count:
            continue
        vectors = _encoded(
            encoder,
            batch_texts,
            batch_size=parameters.batch_size,
            dimension=parameters.dimension,
        )
        index.add(vectors)
        batch_texts = []
        if int(index.ntotal) >= next_checkpoint:
            faiss_module.write_index(index, str(partial_path))
            _write_state(
                state_path,
                {
                    "schema_version": "experiment05.dense_build_state.v1",
                    "build_signature": build_signature,
                    "faiss_ntotal": int(index.ntotal),
                    "trained": True,
                },
            )
            next_checkpoint += parameters.checkpoint_interval
    if int(index.ntotal) != passage_count:
        raise ValueError("dense FAISS ntotal differs from full corpus passage count")
    if hasattr(index, "nprobe"):
        index.nprobe = parameters.nprobe
    final_path = index_dir / "dense.faiss"
    faiss_module.write_index(index, str(final_path))
    _write_state(
        state_path,
        {
            "schema_version": "experiment05.dense_build_state.v1",
            "build_signature": build_signature,
            "faiss_ntotal": int(index.ntotal),
            "trained": True,
            "complete": True,
        },
    )
    dense_manifest: dict[str, Any] = {
        "schema_version": DENSE_INDEX_SCHEMA_VERSION,
        "status": "FROZEN_FULL_CORPUS_INDEX",
        "dataset": manifest["dataset"],
        "corpus_manifest_sha256": corpus_manifest_sha256,
        "corpus_ordered_passage_ids_sha256": manifest["ordered_passage_ids_sha256"],
        "passage_count": passage_count,
        "faiss_ntotal": int(index.ntotal),
        "parameters": parameter_payload,
        "model_snapshot": str(model_snapshot),
        "model_snapshot_sha256": model_snapshot_sha256,
        "model_files": list(model_files),
        "build_signature": build_signature,
        "index_file": final_path.name,
        "index_sha256": _sha256(final_path),
        "builder_sha256": _sha256(Path(__file__).resolve()),
    }
    (index_dir / "dense_index_manifest.json").write_text(
        json.dumps(dense_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return dense_manifest


def pyserini_index_command(
    *, corpus_dir: Path, index_dir: Path, threads: int
) -> list[str]:
    if threads <= 0:
        raise ValueError("threads must be positive")
    return [
        sys.executable,
        "-m",
        "pyserini.index.lucene",
        "--collection",
        "JsonCollection",
        "--input",
        str(corpus_dir / "shards"),
        "--index",
        str(index_dir),
        "--generator",
        "DefaultLuceneDocumentGenerator",
        "--threads",
        str(threads),
        "--storePositions",
        "--storeDocvectors",
        "--storeRaw",
    ]


def freeze_bm25_manifest(
    *,
    corpus_dir: Path,
    index_dir: Path,
    document_count: int,
    pyserini_version: str,
    k1: float = 0.9,
    b: float = 0.4,
) -> dict[str, Any]:
    corpus_manifest = read_corpus_manifest(corpus_dir)
    passage_count = int(corpus_manifest["passage_count"])
    if document_count != passage_count:
        raise ValueError(
            f"BM25 document count {document_count} differs from corpus {passage_count}"
        )
    files: list[dict[str, object]] = []
    for path in sorted(candidate for candidate in index_dir.rglob("*") if candidate.is_file()):
        if path.name == "bm25_index_manifest.json":
            continue
        files.append(
            {
                "file": str(path.relative_to(index_dir)),
                "size": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    if not files:
        raise ValueError("BM25 index contains no files")
    manifest: dict[str, Any] = {
        "schema_version": BM25_INDEX_SCHEMA_VERSION,
        "status": "FROZEN_FULL_CORPUS_INDEX",
        "dataset": corpus_manifest["dataset"],
        "corpus_manifest_sha256": _sha256(corpus_dir / "corpus_manifest.json"),
        "corpus_ordered_passage_ids_sha256": corpus_manifest[
            "ordered_passage_ids_sha256"
        ],
        "passage_count": passage_count,
        "document_count": document_count,
        "implementation": "pyserini-lucene-bm25",
        "pyserini_version": pyserini_version,
        "search_parameters": {"k1": k1, "b": b},
        "stored_fields": ["positions", "docvectors", "raw"],
        "index_files": files,
        "index_tree_sha256": _object_sha256(files),
        "builder_sha256": _sha256(Path(__file__).resolve()),
    }
    (index_dir / "bm25_index_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def normalize_corpus(
    *,
    dataset: str,
    source_kind: str,
    source_path: Path,
    output_dir: Path,
    max_words: int = 100,
    shard_size: int = 250_000,
) -> dict[str, Any]:
    if source_kind not in {"kilt", "dpr"}:
        raise ValueError("source_kind must be kilt or dpr")
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    iterator = (
        iter_kilt_passages(source_path, max_words=max_words)
        if source_kind == "kilt"
        else iter_dpr_passages(source_path)
    )
    shard_dir = output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shards: list[dict[str, object]] = []
    ordered_ids_digest = hashlib.sha256()
    passage_count = 0
    shard_index = 0
    shard_handle: Any | None = None
    shard_path: Path | None = None
    shard_rows = 0

    def finish_shard() -> None:
        nonlocal shard_handle, shard_path, shard_rows
        if shard_handle is None or shard_path is None:
            return
        shard_handle.flush()
        shard_handle.close()
        final_path = shard_path.with_suffix("")
        shard_path.replace(final_path)
        shards.append(
            {
                "file": str(final_path.relative_to(output_dir)),
                "count": shard_rows,
                "sha256": _sha256(final_path),
            }
        )
        shard_handle = None
        shard_path = None
        shard_rows = 0

    for row in iterator:
        if shard_handle is None:
            final_name = f"part-{shard_index:05d}.jsonl"
            shard_path = shard_dir / f"{final_name}.partial"
            shard_handle = shard_path.open("wb")
            shard_index += 1
        if passage_count:
            ordered_ids_digest.update(b"\n")
        ordered_ids_digest.update(str(row["id"]).encode("utf-8"))
        shard_handle.write(_canonical_line(row))
        shard_rows += 1
        passage_count += 1
        if shard_rows >= shard_size:
            finish_shard()
    finish_shard()
    if passage_count == 0:
        raise ValueError("normalized corpus contains no passages")

    manifest: dict[str, Any] = {
        "schema_version": CORPUS_SCHEMA_VERSION,
        "status": "NORMALIZED_FULL_CORPUS",
        "dataset": dataset,
        "source_kind": source_kind,
        "source_path": str(source_path),
        "source_sha256": _sha256(source_path),
        "chunking": {
            "max_words": max_words if source_kind == "kilt" else None,
            "title_prepended": True,
            "kilt_paragraph_zero_title_removed": source_kind == "kilt",
            "long_paragraph_split_without_overlap": source_kind == "kilt",
        },
        "passage_count": passage_count,
        "ordered_passage_ids_sha256": ordered_ids_digest.hexdigest(),
        "shard_count": len(shards),
        "shards": shards,
        "normalizer_sha256": _sha256(Path(__file__).resolve()),
    }
    manifest_path = output_dir / "corpus_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
