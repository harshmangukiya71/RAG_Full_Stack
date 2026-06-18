"""
main.py — FastAPI application with REST endpoints.
"""
from __future__ import annotations

import asyncio
import functools
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.models import (
    CacheEntriesResponse,
    CacheStatus,
    ClearResponse,
    DocumentInfo,
    EvalReport,
    GraphNeighborsResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)
from app.pipeline import RAGPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# Thread pool for running blocking LLM/evaluation calls without blocking the event loop
_thread_pool = ThreadPoolExecutor(max_workers=2)


async def _run_in_thread(fn, *args):
    """Run a blocking function in a thread pool to avoid blocking the async event loop."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_thread_pool, functools.partial(fn, *args))


# ── Global pipeline (singleton) ───────────────────────────────────────────────
_pipeline: RAGPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load pipeline on startup, release on shutdown."""
    global _pipeline
    logger.info("Starting RAG pipeline (downloading models in background)...")
    _pipeline = await _run_in_thread(RAGPipeline)

    # Auto-ingest sample PDFs on first run — only if legal PDFs exist
    # sample_dir = Path("data/pdfs")
    # if sample_dir.exists() and _pipeline.list_documents() == []:
    #     # Only auto-ingest if the directory has the expected legal PDF files
    #     legal_pdfs = [
    #         sample_dir / "NDA-VendorX.pdf",
    #         sample_dir / "SLA-ProviderY.pdf",
    #         sample_dir / "IP-ContractorZ.pdf",
    #     ]
    #     has_legal_pdfs = any(p.exists() for p in legal_pdfs)
    #     if has_legal_pdfs:
    #         logger.info("Auto-ingesting legal sample PDFs from %s", sample_dir)
    #         results = _pipeline.ingest_directory(sample_dir)
    #         for r in results:
    #             logger.info("  Auto-ingested: %s (%d chunks)", r.document, r.chunks_created)
    #     else:
    #         logger.info(
    #             "Skipping auto-ingest: no legal PDFs found in %s. "
    #             "Run 'python generate_sample_pdfs.py' inside /backend to create them.",
    #             sample_dir,
    #         )

    logger.info("Pipeline ready. Documents: %s", _pipeline.list_documents())
    yield
    logger.info("Shutting down.")


# ── App ────────────────────────────────────────────────────────────────────────
settings = get_settings()
app = FastAPI(
    title="DocRAG API",
    description="Production-grade RAG pipeline — ask questions from any uploaded PDF",
    version="1.0.0",
    lifespan=lifespan,
)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://agentic-rag-full-stack-31x3rxy8y.vercel.app",
        "https://agentic-rag-full-stack.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_pipeline() -> RAGPipeline:
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialised yet.")
    return _pipeline


# ── Health ─────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
async def health() -> dict[str, Any]:
    pipeline = get_pipeline()
    return {
        "status": "healthy",
        "documents": pipeline.list_documents(),
        "total_chunks": pipeline._vector_store.count,
    }


# ── Query ──────────────────────────────────────────────────────────────────────
@app.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query(request: QueryRequest) -> QueryResponse:
    """
    Answer a question using the RAG pipeline.
    Returns answer, source citations (document + page + chunk), and confidence score.
    """
    pipeline = get_pipeline()
    try:
        response = pipeline.query(
            question=request.question,
            top_k=request.top_k,
            session_id=request.session_id,
        )
        return response
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Query failed: %s", exc)
        return QueryResponse(
            status="pipeline_error",
            answer="The query pipeline hit an internal error. The root cause was captured in server logs.",
            sources=[],
            confidence=0.0,
        )


# ── Ingest ─────────────────────────────────────────────────────────────────────
SUPPORTED_UPLOAD_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


@app.post("/ingest", response_model=IngestResponse, tags=["Documents"])
async def ingest_document(file: UploadFile = File(...)) -> IngestResponse:
    """Upload and ingest a PDF or image document into the vector store and graph."""
    pipeline = get_pipeline()

    suffix = Path(file.filename or "").suffix.lower()
    if not file.filename or suffix not in SUPPORTED_UPLOAD_SUFFIXES:
        raise HTTPException(status_code=400, detail="Only PDF and image files are accepted.")

    # Save to temp location
    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    document_path = upload_dir / file.filename

    content = await file.read()
    if len(content) > 50 * 1024 * 1024:  # 50 MB limit
        raise HTTPException(status_code=413, detail="File too large (max 50 MB).")

    document_path.write_bytes(content)
    logger.info("Received upload: %s (%d bytes)", file.filename, len(content))

    try:
        result = pipeline.ingest_document(document_path)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("Ingestion failed for %s: %s", file.filename, exc)
        raise HTTPException(status_code=500, detail="Ingestion failed.")

    # Move to permanent storage — overwrite if already exists (e.g. after reset)
    perm_path = Path("data/pdfs") / file.filename
    perm_path.parent.mkdir(parents=True, exist_ok=True)
    if perm_path.exists():
        perm_path.unlink()          # remove stale file before rename
    document_path.rename(perm_path)

    return result


# ── Documents ──────────────────────────────────────────────────────────────────
@app.get("/documents", response_model=list[str], tags=["Documents"])
async def list_documents() -> list[str]:
    """List all ingested document names."""
    return get_pipeline().list_documents()


@app.get("/documents/{document_name}", response_model=DocumentInfo, tags=["Documents"])
async def get_document(document_name: str) -> DocumentInfo:
    pipeline = get_pipeline()
    info = pipeline._vector_store.get_document_info(document_name)
    if not info:
        raise HTTPException(status_code=404, detail=f"Document '{document_name}' not found.")
    return DocumentInfo(**info)


@app.get("/entities", tags=["Graph"])
async def search_entities(q: str, limit: int = 10) -> list[dict[str, Any]]:
    """Search indexed entities by name or alias."""
    pipeline = get_pipeline()
    limit = min(max(limit, 1), 50)
    return [entity.model_dump() for entity in pipeline._graph_store.search_entities(q, limit=limit)]


@app.get("/entities/{entity_id}/neighbors", response_model=GraphNeighborsResponse, tags=["Graph"])
async def entity_neighbors(entity_id: str, depth: int = 1) -> GraphNeighborsResponse:
    """Return nearby entities and relationships from the document graph."""
    pipeline = get_pipeline()
    entity, neighbors, relationships = pipeline._graph_store.neighbors(entity_id, depth=depth)
    if entity is None:
        raise HTTPException(status_code=404, detail=f"Entity '{entity_id}' not found.")
    return GraphNeighborsResponse(entity=entity, neighbors=neighbors, relationships=relationships)


@app.delete("/documents/{document_name}", tags=["Documents"])
async def delete_document(document_name: str) -> dict[str, Any]:
    """Remove a document from the vector index AND delete its PDF from disk."""
    pipeline = get_pipeline()
    deleted = pipeline.delete_document(document_name)
    if deleted == 0:
        raise HTTPException(status_code=404, detail=f"Document '{document_name}' not found.")

    # Also delete PDF file from disk so re-upload works cleanly
    for search_dir in [Path("data/pdfs"), Path("data/uploads")]:
        candidate = search_dir / document_name
        if candidate.exists():
            candidate.unlink()
            logger.info("Deleted file from disk: %s", candidate)

    return {"document": document_name, "chunks_deleted": deleted}


@app.post("/reset", tags=["System"])
async def reset_all_documents() -> dict[str, Any]:
    """
    Delete ALL documents from the vector store AND from disk.
    After this, the same PDFs can be re-uploaded without any "file exists" errors.
    """
    pipeline = get_pipeline()
    docs = pipeline.list_documents()
    total_deleted = 0
    for doc in docs:
        total_deleted += pipeline.delete_document(doc)

    # Delete all PDF files from disk (both permanent and upload staging dirs)
    files_deleted: list[str] = []
    for search_dir in [Path("data/pdfs"), Path("data/uploads")]:
        if search_dir.exists():
            for f in search_dir.iterdir():
                if f.suffix.lower() in SUPPORTED_UPLOAD_SUFFIXES:
                    f.unlink()
                    files_deleted.append(f.name)
                    logger.info("Reset: deleted file %s", f)

    logger.info(
        "Reset complete: %d chunks, %d docs removed from index; %d PDF files deleted from disk",
        total_deleted, len(docs), len(files_deleted),
    )
    return {
        "status": "reset_complete",
        "documents_removed": docs,
        "chunks_deleted": total_deleted,
        "files_deleted_from_disk": files_deleted,
    }


# ── Cache endpoints (Features 6, 7) ──────────────────────────────────────────

@app.get("/cache/status", response_model=CacheStatus, tags=["Cache"])
async def cache_status() -> CacheStatus:
    """
    Get cache loading progress.

    After server restart, the system loads cached answers from Redis into
    the in-memory L2 layer.  This endpoint reports progress (0–100%)
    and whether the cache has reached the readiness threshold.

    Frontend can poll this to show "Cache loading..." and prevent
    answering until cache is ready.
    """
    pipeline = get_pipeline()
    status = pipeline._answer_cache.get_load_status()
    return CacheStatus(**status)


@app.get("/cache/entries", response_model=CacheEntriesResponse, tags=["Cache"])
async def cache_entries() -> CacheEntriesResponse:
    """
    Get all loaded cache entries and their remaining TTL.
    """
    pipeline = get_pipeline()
    entries = pipeline._answer_cache.get_all_entries()
    return CacheEntriesResponse(entries=entries)


@app.post("/cache/clear", response_model=ClearResponse, tags=["Cache"])
async def clear_cache(background_tasks: BackgroundTasks) -> ClearResponse:
    """
    Clear ALL caches:
      • Redis L1 cache
      • In-memory L2 cache
      • Semantic cache embeddings
      • Cache statistics

    Does NOT touch documents, vector store, or graph.
    """
    pipeline = get_pipeline()
    try:
        def _clear():
            pipeline._answer_cache.clear()
            pipeline._answer_cache.reset_stats()
        
        background_tasks.add_task(_clear)
        logger.info("Cache clear triggered via API (background)")
        return ClearResponse(success=True, detail="Cache clear started in background.")
    except Exception as exc:
        logger.exception("Cache clear failed to start: %s", exc)
        return ClearResponse(success=False, detail=f"Cache clear failed to start: {exc}")


# ── Chat clear endpoint (Feature 8) ──────────────────────────────────────────

@app.post("/chat/clear", response_model=ClearResponse, tags=["Chat"])
async def clear_chat() -> ClearResponse:
    """
    Clear ALL conversational memory:
      • Chat history (all sessions)
      • Session memory
      • Conversation context

    Does NOT delete:
      • Chroma database
      • Graph database
      • Uploaded documents

    Only conversational memory is removed.
    """
    pipeline = get_pipeline()
    try:
        pipeline._memory.clear_all()
        logger.info("Chat history cleared via API")
        return ClearResponse(success=True, detail="All chat history cleared successfully.")
    except Exception as exc:
        logger.exception("Chat clear failed: %s", exc)
        return ClearResponse(success=False, detail=f"Chat clear failed: {exc}")


# ── Debug / Diagnostics ───────────────────────────────────────────────────────
@app.post("/debug/query", tags=["Debug"])
async def debug_query(request: QueryRequest) -> dict:
    """
    Debug endpoint: shows routing decision, query classification,
    raw retrieval chunks + scores.
    Use this to diagnose why a question isn't being answered correctly.
    """
    from app.pipeline import _is_keyword_presence_question, _extract_keyword
    pipeline = get_pipeline()
    settings = get_settings()

    is_keyword_q = _is_keyword_presence_question(request.question)
    extracted_kw = _extract_keyword(request.question) if is_keyword_q else None

    # Query classification (if agent enabled)
    classification_info = None
    if pipeline._query_agent:
        classification = pipeline._query_agent.classify(request.question)
        classification_info = classification.model_dump()

    query_embedding = pipeline._embedding_model.embed_query(request.question)
    graph_results, query_entity_ids = pipeline._graph_retrieve(request.question)
    retrieved = pipeline._retriever.retrieve(
        query=request.question,
        query_embedding=query_embedding,
        vector_store=pipeline._vector_store,
        top_k_final=settings.rerank_top_k,
        graph_results=graph_results,
        query_entity_ids=query_entity_ids,
    )

    return {
        "question": request.question,
        "query_classification": classification_info,
        "routing": {
            "is_keyword_presence_question": is_keyword_q,
            "extracted_keyword": extracted_kw,
            "will_use_keyword_path": is_keyword_q and extracted_kw is not None,
        },
        "total_chunks_in_store": pipeline._vector_store.count,
        "documents_in_store": pipeline.list_documents(),
        "graph_retrieved_chunks": len(graph_results),
        "semantic_retrieved_chunks": [
            {
                "document": c.document,
                "page": c.page,
                "relevance_score": c.relevance_score,
                "chunk_preview": c.chunk[:200],
            }
            for c in retrieved
        ],
    }


# ── Evaluation ─────────────────────────────────────────────────────────────────
@app.post("/evaluate", response_model=EvalReport, tags=["Evaluation"])
async def run_auto_evaluation(n_pairs: int = 10) -> EvalReport:
    """
    AUTO evaluation — works with ANY uploaded PDF.

    Samples chunks from your ingested documents, uses the LLM to generate
    one factual question per chunk (with retry), then checks if the retriever
    finds the correct source chunk. This gives a real accuracy score for
    whatever documents you have uploaded right now.

    Args:
        n_pairs: Number of Q&A pairs to generate (default 10, max 20).
                 If LLM question-generation fails for some chunks, the report
                 will contain fewer items — this is expected.
    """
    from app.evaluation import auto_evaluate
    pipeline = get_pipeline()

    if not pipeline.list_documents():
        raise HTTPException(
            status_code=400,
            detail="No documents ingested. Upload at least one PDF before running evaluation.",
        )

    n_pairs = min(max(n_pairs, 3), 20)   # clamp to [3, 20]
    logger.info("Starting auto-evaluation with n_pairs=%d", n_pairs)
    try:
        report = await _run_in_thread(auto_evaluate, pipeline, n_pairs)
        logger.info(
            "Evaluation complete: %d/%d questions, Recall@3=%.1f%%",
            report.total_questions, n_pairs, report.recall_at_3 * 100,
        )
        return report
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Auto-evaluation failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {exc}")


@app.post("/evaluate/predefined", response_model=EvalReport, tags=["Evaluation"])
async def run_predefined_evaluation() -> EvalReport:
    """
    PREDEFINED evaluation — uses tests/eval_qa_pairs.json.
    Only gives accurate results when the expected documents are ingested.
    Use /evaluate (auto mode) for any uploaded PDF.
    """
    from app.evaluation import run_evaluation as _run_eval
    qa_path = Path("tests/eval_qa_pairs.json")
    if not qa_path.exists():
        raise HTTPException(status_code=404, detail="tests/eval_qa_pairs.json not found.")
    pipeline = get_pipeline()
    try:
        report = await _run_in_thread(_run_eval, qa_path, pipeline)
        return report
    except Exception as exc:
        logger.exception("Predefined evaluation failed: %s", exc)
        raise HTTPException(status_code=500, detail="Evaluation failed.")
