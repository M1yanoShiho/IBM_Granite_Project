"""Thin HTTP/SSE wrapper around the existing three-module RAG pipeline.

Provides a Perplexity-like chat endpoint that streams retrieval candidates,
selection results, and generated answer tokens as SSE events.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from threading import Thread
from typing import AsyncGenerator

from evidence_rag.contracts.models import (
    Document,
    GenerationResult,
    Query,
)
from evidence_rag.infrastructure.corpus import CorpusBuilder, CorpusSnapshot
from evidence_rag.contracts.validation import resolve_selection
from evidence_rag.generator.granite import (
    CITATION_RAG_PROMPT,
    GraniteGenerator,
    GraniteLLMClient,
    _fallback_citation_ids,
    _is_unknown_answer,
    parse_citation_output,
)
from evidence_rag.pipeline.service import EvidenceRAGPipeline
from evidence_rag.query_analysis import RuleBasedQueryAnalyzer
from evidence_rag.retriever.bm25 import BM25Retriever
from evidence_rag.selector.top_k import TopKSelector


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

from fastapi import FastAPI, UploadFile
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Evidence RAG Demo", version="0.1.0")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

from pydantic import BaseModel


class ChatRequestBody(BaseModel):
    query: str


class HealthResponse(BaseModel):
    status: str
    pipeline: str


class SettingsState(BaseModel):
    retriever: str = "bm25"
    selector: str = "top-k"
    generator: str = "local"
    verifier: str = "off"


_current_settings = SettingsState(
    retriever=os.environ.get("DEFAULT_RETRIEVER", "bm25"),
    selector=os.environ.get("DEFAULT_SELECTOR", "top-k"),
    generator="local",
    verifier=os.environ.get("NLI_BACKEND", "off"),
)

AVAILABLE = {
    "retrievers": [
        {"id": "bm25", "label": "BM25", "desc": "Okapi BM25 (k1=1.5, b=0.75)"},
        {"id": "strong-bm25", "label": "Strong BM25", "desc": "BEIR-tuned (k1=0.9, b=0.4)"},
        {"id": "granite-dense", "label": "Granite Dense", "desc": "Dense embeddings (~500MB model)"},
        {"id": "hybrid-rrf", "label": "Hybrid RRF", "desc": "BM25 + Dense, rank fusion"},
        {"id": "hybrid-convex", "label": "Hybrid Convex", "desc": "BM25 + Dense, score fusion"},
        {"id": "query2doc", "label": "Query2Doc", "desc": "LLM pseudo-doc → retrieve"},
        {"id": "hyde", "label": "HyDE", "desc": "LLM hypothetical answer → retrieve"},
        {"id": "decompose", "label": "Decompose", "desc": "LLM sub-questions → RRF fusion"},
    ],
    "selectors": [
        {"id": "top-k", "label": "Top-K", "desc": "Trim by retrieval rank"},
        {"id": "corroboration", "label": "Corroboration", "desc": "LLM answer clustering"},
        {"id": "gated-corroboration", "label": "Gated Corrob", "desc": "+ 4-condition gate"},
        {"id": "gated-coverage", "label": "Gated Coverage", "desc": "+ coverage gate"},
    ],
    "generators": [
        {"id": "local", "label": "Granite 3B", "desc": "Local HF, streaming"},
        {"id": "ollama", "label": "Ollama", "desc": "GGUF quantized, fast"},
    ],
    "verifiers": [
        {"id": "off", "label": "Off", "desc": "No verification pass"},
        {"id": "minicheck", "label": "MiniCheck", "desc": "CPU, recall .620 / FP .020"},
        {"id": "deberta-base", "label": "DeBERTa Base", "desc": "CPU, recall .467 / FP .007"},
        {"id": "deberta-large", "label": "DeBERTa Large", "desc": "CPU, recall .540 / FP .073"},
        {"id": "true", "label": "TRUE (11B)", "desc": "GPU 21GB, recall .747 / FP .007 ★"},
        {"id": "granite-3b", "label": "Granite 3B Judge", "desc": "GPU, recall .767 / FP .067"},
        {"id": "granite-8b", "label": "Granite 8B Judge", "desc": "GPU, recall .900 / FP .240"},
    ],
}


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------

def _sse_event(event: str, data: dict | str) -> str:
    """Format a single SSE event."""
    payload = json.dumps(data, default=str, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


# ---------------------------------------------------------------------------
# Streaming LLM wrapper
# ---------------------------------------------------------------------------

class _LocalStreamingGenerator:
    """Streaming wrapper around local Granite — used when LLM_BACKEND=local (default)."""

    def __init__(self, client: GraniteLLMClient) -> None:
        self._client = client

    async def generate_sse_events(self, prompt: str):
        """Yield SSE event tuples: ('chunk', data) then final full text via 'done'."""
        try:
            from transformers import TextIteratorStreamer
        except ImportError:
            yield ("error", {"message": "streaming requires transformers"})
            return

        import torch

        tokenizer = self._client._tokenizer
        model = self._client._model

        if getattr(tokenizer, "chat_template", None):
            encoded = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
        else:
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids

        model_device = getattr(model, "device", None)
        if model_device is not None and hasattr(input_ids, "to"):
            input_ids = input_ids.to(model_device)

        streamer = TextIteratorStreamer(
            tokenizer,
            skip_special_tokens=True,
            skip_prompt=True,
        )

        generation_kwargs: dict = {
            "input_ids": input_ids,
            "max_new_tokens": self._client.config.max_new_tokens,
            "do_sample": False,
            "streamer": streamer,
        }

        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        full_response = ""
        for token in streamer:
            full_response += token
            yield ("chunk", {"text": token})

        thread.join()
        yield ("done", full_response)


# ---------------------------------------------------------------------------
# Pipeline factory
# ---------------------------------------------------------------------------

_pipeline: EvidenceRAGPipeline | None = None
_pipeline_llm: GraniteLLMClient | None = None
_base_snapshot: CorpusSnapshot | None = None


def _invalidate_pipeline() -> None:
    """Force the pipeline to rebuild on next request (e.g. after file upload)."""
    global _pipeline, _pipeline_llm
    # Free the old model from memory before rebuilding
    if _pipeline_llm is not None:
        del _pipeline_llm
        import gc
        gc.collect()
        try:
            import torch
            torch.mps.empty_cache()
        except Exception:
            pass
    _pipeline = None
    _pipeline_llm = None


def _build_snapshot_from_documents(docs_path: Path) -> "CorpusSnapshot":
    """Build a CorpusSnapshot from a documents.jsonl file."""
    import json as _json
    from evidence_rag.infrastructure.corpus import CorpusBuilder
    from evidence_rag.contracts.models import Document

    documents = [
        Document.model_validate(_json.loads(line))
        for line in docs_path.read_text("utf-8").strip().splitlines()
    ]
    return CorpusBuilder().build(
        documents,
        dataset_signature=f"cli-{docs_path}",
    )


def _get_pipeline() -> EvidenceRAGPipeline:
    """Build or return the cached pipeline (BM25 + TopK + Generator).

    When LLM_BACKEND=anthropic, skips loading the local Granite model entirely.
    Uploaded files are merged into the base corpus on each rebuild.
    """
    global _pipeline, _pipeline_llm, _base_snapshot

    if _pipeline is not None and not _uploaded_docs:
        return _pipeline

    # --- Corpus loading (cached after first load) ---
    if _base_snapshot is None:
        corpus_env = os.environ.get("EVIDENCE_RAG_CORPUS")
        if not corpus_env:
            raise RuntimeError(
                "Set EVIDENCE_RAG_CORPUS to a run directory containing "
                "index/corpus_snapshot.json, or a documents.jsonl file."
            )

        corpus_path = Path(corpus_env)
        if corpus_path.is_dir():
            snapshot_file = corpus_path / "index" / "corpus_snapshot.json"
            if snapshot_file.exists():
                _base_snapshot = CorpusSnapshot.model_validate_json(
                    snapshot_file.read_text("utf-8")
                )
            else:
                docs_file = corpus_path / "documents.jsonl"
                if not docs_file.exists():
                    raise RuntimeError(
                        f"No index/corpus_snapshot.json or documents.jsonl found in {corpus_path}"
                    )
                _base_snapshot = _build_snapshot_from_documents(docs_file)
        elif corpus_path.suffix == ".jsonl":
            _base_snapshot = _build_snapshot_from_documents(corpus_path)
        elif corpus_path.suffix == ".json":
            _base_snapshot = CorpusSnapshot.model_validate_json(
                corpus_path.read_text("utf-8")
            )
        else:
            raise RuntimeError(f"Cannot interpret EVIDENCE_RAG_CORPUS={corpus_env}")

    # --- Merge uploaded documents with base corpus ---
    if _uploaded_docs:
        all_docs = tuple(_base_snapshot.documents) + tuple(_uploaded_docs)
        snapshot = CorpusBuilder().build(
            all_docs,
            dataset_signature=_base_snapshot.manifest.dataset_signature,
        )
    else:
        snapshot = _base_snapshot

    # Shared LLM for retriever wrappers, corroboration selector, and generator
    if _current_settings.generator == "ollama":
        from evidence_rag.generator.ollama import OllamaTextGenerator
        _shared_llm = OllamaTextGenerator()
    else:
        _shared_llm = GraniteLLMClient()

    # --- Select retriever ---
    from evidence_rag.infrastructure.config import ModuleConfig
    from evidence_rag.composition import build_retriever

    retriever_name = _current_settings.retriever
    if retriever_name in {"bm25", "strong-bm25"}:
        retriever = build_retriever(
            ModuleConfig(name=retriever_name, parameters={}), snapshot
        )
    elif retriever_name == "granite-dense":
        retriever = build_retriever(
            ModuleConfig(name="granite-dense", parameters={}), snapshot
        )
    elif retriever_name == "hybrid-rrf":
        retriever = build_retriever(
            ModuleConfig(name="hybrid", parameters={
                "fusion": "rrf",
                "retrievers": [
                    {"name": "bm25", "parameters": {}},
                    {"name": "granite-dense", "parameters": {}},
                ],
            }), snapshot
        )
    elif retriever_name == "hybrid-convex":
        retriever = build_retriever(
            ModuleConfig(name="hybrid", parameters={
                "fusion": "convex",
                "retrievers": [
                    {"name": "bm25", "parameters": {}},
                    {"name": "granite-dense", "parameters": {}},
                ],
            }), snapshot
        )
    elif retriever_name in {"query2doc", "hyde", "decompose"}:
        # Uses the shared LLM (defined after retriever construction)
        from evidence_rag.retriever.granite import GraniteDenseRetriever, GraniteEmbedder
        embedder = GraniteEmbedder()
        dense = GraniteDenseRetriever.from_corpus(snapshot, embedder=embedder)
        if retriever_name == "query2doc":
            from evidence_rag.retriever.granite import Query2DocRetriever
            retriever = Query2DocRetriever(dense, _shared_llm)
        elif retriever_name == "hyde":
            from evidence_rag.retriever.granite import HyDERetriever
            retriever = HyDERetriever(dense, _shared_llm)
        else:
            from evidence_rag.retriever.granite import DecomposingRetriever
            retriever = DecomposingRetriever(dense, _shared_llm)
    else:
        retriever = BM25Retriever.from_corpus(snapshot)  # fallback

    # --- Select selector ---
    selector_name = _current_settings.selector
    if selector_name == "corroboration":
        from evidence_rag.selector.corroboration import CorroborationSelector
        selector: Selector = CorroborationSelector(_shared_llm, alpha=0.6)
    elif selector_name in {"gated-corroboration", "gated-coverage"}:
        from evidence_rag.selector.gated import GatedCorroborationSelector, GatedCoverageSelector
        sel_cls = (
            GatedCoverageSelector
            if selector_name == "gated-coverage"
            else GatedCorroborationSelector
        )
        selector = sel_cls(
            _shared_llm,
            alpha=0.6,
            margin=2,
            support_cap=1,
            top_n=20,
            equivalence="lenient",
        )
    else:
        from evidence_rag.selector.top_k import TopKSelector
        selector = TopKSelector()

    # --- Generator — reuse shared LLM, optionally wrapped with verification ---
    gen = _current_settings.generator
    if gen == "local":
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError:
            raise RuntimeError("ui server requires torch+transformers") from None

    _pipeline_llm = _shared_llm

    if _current_settings.verifier != "off":
        from evidence_rag.generator.verified import VerifiedGenerator
        from evidence_rag.generator.nli import build_nli_model
        nli = build_nli_model()
        _pipeline = EvidenceRAGPipeline(
            retriever=retriever,
            selector=selector,
            generator=VerifiedGenerator(llm=_pipeline_llm, nli=nli),
            query_analyzer=RuleBasedQueryAnalyzer(),
        )
    else:
        _pipeline = EvidenceRAGPipeline(
            retriever=retriever,
            selector=selector,
            generator=GraniteGenerator(llm=_pipeline_llm),
            query_analyzer=RuleBasedQueryAnalyzer(),
        )
    return _pipeline


# ---------------------------------------------------------------------------
# File upload support
# ---------------------------------------------------------------------------

_uploaded_docs: list[Document] = []
_uploaded_filenames: list[str] = []
_upload_dir = Path(os.environ.get("EVIDENCE_RAG_UPLOAD_DIR", "/tmp/evidence-rag-uploads"))
_upload_dir.mkdir(parents=True, exist_ok=True)


def _load_text_doc(path: Path, original_name: str) -> Document:
    """Load a single text/markdown file into a Document."""
    from evidence_rag.contracts.models import SourceMetadata

    return Document(
        document_id=f"upload:{original_name}",
        text=path.read_text("utf-8"),
        source_uri=f"upload://{original_name}",
        metadata=SourceMetadata(source_type="txt", file_name=original_name),
    )


def _load_pdf_doc(path: Path, original_name: str) -> Document | None:
    """Try to load a PDF; returns None if docling is not installed."""
    try:
        from evidence_rag.loaders.pdf_loader import load_pdf
        return load_pdf(str(path), source_uri=f"upload://{original_name}")
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Readiness check."""
    try:
        _get_pipeline()
        status = "ready"
        pipeline = f"bm25 + {_current_settings.selector} + {_current_settings.generator}"
    except RuntimeError as exc:
        status = f"unavailable: {exc}"
        pipeline = ""
    return HealthResponse(status=status, pipeline=pipeline)


@app.post("/chat")
async def chat(body: ChatRequestBody):
    """Main chat endpoint — streams SSE events through the full pipeline."""
    return _StreamingResponse(_event_stream(body.query), media_type="text/event-stream")


class UploadedFile(BaseModel):
    name: str


@app.get("/uploads")
async def list_uploads() -> list[UploadedFile]:
    """List currently uploaded files."""
    return [UploadedFile(name=f) for f in _uploaded_filenames]


@app.post("/upload")
async def upload_file(file: UploadFile):
    """Upload a .txt, .md, or .pdf file to be included in the search corpus."""
    ext = Path(file.filename or "unknown").suffix.lower()
    if ext not in {".txt", ".md", ".pdf"}:
        return {"error": f"Unsupported file type: {ext}"}, 415

    # Save to temp
    safe_name = file.filename or "uploaded"
    dest = _upload_dir / safe_name
    content = await file.read()
    dest.write_bytes(content)

    # Parse
    if ext == ".pdf":
        doc = _load_pdf_doc(dest, safe_name)
        if doc is None:
            return {"error": "PDF support requires docling: pip install 'evidence-rag[ingestion]'"}, 501
    else:
        doc = _load_text_doc(dest, safe_name)

    # Store and rebuild
    _uploaded_docs.append(doc)
    _uploaded_filenames.append(safe_name)
    _invalidate_pipeline()

    return {"ok": True, "name": safe_name, "size": len(content)}


@app.delete("/uploads/{name}")
async def delete_upload(name: str):
    """Remove an uploaded file."""
    global _uploaded_docs, _uploaded_filenames
    idx = next((i for i, f in enumerate(_uploaded_filenames) if f == name), None)
    if idx is None:
        return {"error": "not found"}, 404
    _uploaded_docs.pop(idx)
    _uploaded_filenames.pop(idx)
    (_upload_dir / name).unlink(missing_ok=True)
    _invalidate_pipeline()
    return {"ok": True}


@app.get("/config")
async def get_config() -> dict:
    """Return available settings + current values."""
    return {**AVAILABLE, "current": _current_settings.model_dump()}


@app.put("/config")
async def update_config(body: SettingsState):
    """Update retriever/selector/verifier and rebuild the pipeline."""
    global _current_settings
    _current_settings = body
    # Set NLI backend for VerifiedGenerator
    if body.verifier != "off":
        os.environ["NLI_BACKEND"] = body.verifier
    _invalidate_pipeline()
    return {"ok": True}


def _format_prompt(query_text: str, selected) -> str:
    """Build the citation prompt, same format as GraniteGenerator."""
    context = "\n".join(
        f"[{i}] ({item.evidence_id}) {item.text}"
        for i, item in enumerate(selected.evidence, start=1)
    )
    return CITATION_RAG_PROMPT.format(context=context, question=query_text)


def _parse_response(full_response: str, query_id: str, selected) -> GenerationResult:
    """Parse LLM output into answer + citations (same logic as GraniteGenerator)."""
    answer, citation_indices = parse_citation_output(full_response)
    if _is_unknown_answer(answer):
        return GenerationResult(query_id=query_id, answer="", cited_evidence_ids=())
    cited_ids = GraniteGenerator._citation_ids(citation_indices, selected)
    if not cited_ids:
        cited_ids = _fallback_citation_ids(answer, selected)
    return GenerationResult(
        query_id=query_id,
        answer=answer,
        cited_evidence_ids=cited_ids,
    )


def _make_streaming_generator():
    """Create the streaming generator for the current backend."""
    if _current_settings.generator == "ollama":
        from evidence_rag.generator.ollama import OllamaStreamingGenerator
        return OllamaStreamingGenerator()
    return _LocalStreamingGenerator(_pipeline_llm)


async def _event_stream(query_text: str) -> AsyncGenerator[str, None]:
    """Run the pipeline and yield SSE events at each stage."""
    query_id = f"ui-{abs(hash(query_text))}"
    query = Query(query_id=query_id, text=query_text)

    try:
        pipeline = _get_pipeline()
    except RuntimeError as exc:
        yield _sse_event("error", {"message": str(exc)})
        return

    # --- Phase 1: Retrieval ---
    yield _sse_event("status", {"phase": "retrieving"})
    try:
        candidates = pipeline.retriever.retrieve(query, top_k=20)
    except Exception as exc:
        yield _sse_event("error", {"message": f"retrieval failed: {exc}"})
        return
    yield _sse_event("candidates", candidates.model_dump())

    # --- Phase 2: Selection ---
    yield _sse_event("status", {"phase": "selecting"})
    try:
        selection = pipeline.selector.select(query, candidates, max_selected=10)
    except Exception as exc:
        yield _sse_event("error", {"message": f"selection failed: {exc}"})
        return
    yield _sse_event("selection", selection.model_dump())

    try:
        selected = resolve_selection(candidates, selection)
    except Exception as exc:
        yield _sse_event("error", {"message": f"evidence resolution failed: {exc}"})
        return

    # --- Phase 3: Generation (streaming) ---
    yield _sse_event("status", {"phase": "generating"})

    if not selected.evidence:
        result = GenerationResult(query_id=query_id, answer="", cited_evidence_ids=())
        yield _sse_event("done", result.model_dump())
        return

    if _current_settings.verifier != "off":
        # VerifiedGenerator — multi-step, no streaming
        try:
            checklist = pipeline.query_analyzer.analyze(query)
            result = pipeline.generator.generate(query, checklist, selected)
        except Exception as exc:
            yield _sse_event("error", {"message": f"verified generation failed: {exc}"})
            return
        yield _sse_event("done", result.model_dump())
    elif _current_settings.generator in {"local", "ollama"}:
        # Streaming token-by-token
        prompt = _format_prompt(query_text, selected)
        generator = _make_streaming_generator()

        full_response = ""
        try:
            async for event_type, data in generator.generate_sse_events(prompt):
                if event_type == "error":
                    yield _sse_event("error", data)
                    return
                elif event_type == "chunk":
                    full_response += data["text"]
                    yield _sse_event("chunk", data)
                elif event_type == "done":
                    full_response = data
                    break
        except Exception as exc:
            yield _sse_event("error", {"message": f"generation failed: {exc}"})
            return

        result = _parse_response(full_response, query_id, selected)
        yield _sse_event("done", result.model_dump())


# Alias so FastAPI recognises the streaming response type
def _StreamingResponse(*args, **kwargs):
    from fastapi.responses import StreamingResponse
    return StreamingResponse(*args, **kwargs)


# ---------------------------------------------------------------------------
# Static files (Next.js build output) — mounted after API routes
# ---------------------------------------------------------------------------

_STATIC_DIR = Path(__file__).resolve().parent.parent.parent.parent / "web" / "out"


@app.get("/{full_path:path}")
async def _serve_frontend(full_path: str):
    """Catch-all: serve Next.js static export, falling back to index.html."""
    from fastapi.responses import FileResponse, HTMLResponse

    if not _STATIC_DIR.is_dir():
        return HTMLResponse(
            "<h2>Frontend not built</h2><p>Run: <code>cd web && npm run build</code></p>",
            status_code=503,
        )

    file_path = _STATIC_DIR / full_path
    if file_path.is_file():
        return FileResponse(file_path)

    # SPA fallback — serve index.html for any unmatched route
    index = _STATIC_DIR / "index.html"
    if index.is_file():
        return FileResponse(index)

    return HTMLResponse("<h2>Frontend build incomplete</h2>", status_code=503)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    if _STATIC_DIR.is_dir():
        print(f"Frontend: {_STATIC_DIR} ({sum(1 for _ in _STATIC_DIR.rglob('*') if _.is_file())} files)")
    else:
        print("Frontend not built — run: cd web && npm run build")

    print("Generator: Granite 3B (local HF)")
    print("Starting at http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
