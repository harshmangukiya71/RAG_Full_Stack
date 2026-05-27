"""
pipeline.py — Main RAGPipeline class.

This is the canonical interface. All components are wired together here.
Usage:
    pipeline = RAGPipeline()
    result = pipeline.query("What is the notice period in the NDA with Vendor X?")
    result = pipeline.query("Is the keyword 'CPI' mentioned in this document?")
    # result: QueryResponse with .answer, .sources, .confidence
"""
from __future__ import annotations

import logging
import re
from hashlib import sha256
from pathlib import Path
from typing import Optional

import chromadb

from app.cache import AnswerCache
from app.config import get_settings
from app.embeddings import EmbeddingModel
from app.generation import generate_answer, generate_llm_fallback
from app.ingestion import parse_pdf
from app.memory import MemoryManager
from app.models import Chunk, DocumentInfo, IngestResponse, QueryResponse, SourceReference
from app.retrieval import HybridRetriever
from app.summarizer import DocumentSummarizer
from app.vectorstore import VectorStore

logger = logging.getLogger(__name__)

# ── Keyword-presence question detector ───────────────────────────────────────
# Matches: "is X mentioned?", "does the document contain X?", "is keyword X present?"
_KEYWORD_PRESENCE_RE = re.compile(
    r"\b(?:is|are|does|do|can|has|have)\b.{0,60}"
    r"\b(?:mention(?:ed)?|present|includ(?:ed)?|contain(?:ed)?|appear(?:s)?|found|exist(?:s)?|refer(?:red)?)\b",
    re.IGNORECASE,
)

# Patterns to extract the keyword from a keyword-presence question
_KEYWORD_EXTRACT_PATTERNS = [
    r"['\"\u201c\u201d]([^'\"\u201c\u201d]{1,40})['\"\u201c\u201d]",   # 'CPI' or "CPI"
    r"\bkeyword\s+['\"]?(\w[\w\-\.]*)['\"]?",                           # keyword CPI
    r"\bterm\s+['\"]?(\w[\w\-\.]*)['\"]?",                              # term XYZ
    r"\b(?:is|are|does)\s+(?:the\s+)?(?:word\s+)?(\w[\w\-\.]*)\s+"
    r"(?:mention|present|includ|contain|appear|found|exist|refer)",     # is CPI mention
    r"\b(?:contain|mention|include|refer\s+to)\s+['\"]?(\w[\w\-\.]*)['\"]?",
]

# ── General question detector ───────────────────────────────────────────────────────
# If question seems document-specific (has doc-referencing words), do NOT fall back to LLM.
_DOC_SPECIFIC_RE = re.compile(
    r"\b(?:document|contract|agreement|policy|clause|section|exhibit|appendix|"
    r"resume|cv|report|file|page|paragraph|excerpt|content|the\s+pdf|"
    r"this\s+document|provided|mentioned|ingested|uploaded)\b",
    re.IGNORECASE,
)

# LLM fallback threshold: if top retrieved chunk scores below this, attempt fallback
_FALLBACK_RELEVANCE_THRESHOLD = 0.15

_CONTEXTUAL_FOLLOWUP_RE = re.compile(
    r"\b(?:it|that|this|they|them|those|these|he|she|same|above|previous|earlier)\b",
    re.IGNORECASE,
)


def _is_general_question(question: str) -> bool:
    """Return True if question does NOT reference document-specific concepts."""
    return not bool(_DOC_SPECIFIC_RE.search(question))


def _is_keyword_presence_question(question: str) -> bool:
    """Return True if the question is asking whether a keyword exists in the document."""
    return bool(_KEYWORD_PRESENCE_RE.search(question))


def _extract_keyword(question: str) -> str | None:
    """Extract the keyword being searched for from a keyword-presence question."""
    for pattern in _KEYWORD_EXTRACT_PATTERNS:
        m = re.search(pattern, question, re.IGNORECASE)
        if m:
            kw = m.group(1).strip()
            if len(kw) >= 1:
                return kw
    return None


class RAGPipeline:
    """
    Production RAG pipeline for document question-answering.

    Lifecycle:
      1. Instantiate once (heavy models loaded on first use).
      2. Call ingest_document() to add PDFs.
      3. Call query() to answer questions.

    Supports any PDF type: legal, resume, story, technical, medical, etc.
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._vector_store = VectorStore()
        self._retriever = HybridRetriever()
        self._embedding_model = EmbeddingModel.get()
        self._summarizer = DocumentSummarizer()
        self._memory = MemoryManager()
        self._answer_cache = AnswerCache()

        # Bootstrap BM25 from existing ChromaDB data (if any)
        self._rebuild_bm25_from_store()
        logger.info("RAGPipeline initialised — %d chunks in store", self._vector_store.count)

    # ── Ingestion ────────────────────────────────────────────────────────────

    def ingest_document(self, pdf_path: Path | str) -> IngestResponse:
        """
        Ingest a single PDF document:
          parse → chunk → embed → upsert into ChromaDB → rebuild BM25.
        """
        pdf_path = Path(pdf_path)
        settings = self._settings

        logger.info("Ingesting: %s", pdf_path.name)

        # 1. Parse PDF into chunks
        chunks = parse_pdf(
            pdf_path,
            chunk_size_tokens=settings.chunk_size_tokens,
            chunk_overlap_tokens=settings.chunk_overlap_tokens,
        )
        if not chunks:
            raise ValueError(f"No text could be extracted from {pdf_path.name}")

        # 2. Summarize full extracted text before indexing it
        summary_text = ""
        try:
            summary_text = self._summarizer.summarize_chunks(pdf_path.name, chunks).summary
        except Exception as exc:
            logger.exception("Summary generation failed for %s: %s", pdf_path.name, exc)

        # 3. Embed all chunks
        texts = [c.text for c in chunks]
        embeddings = self._embedding_model.embed_documents(texts)

        # 4. Upsert into ChromaDB
        self._vector_store.upsert_chunks(chunks, embeddings.tolist(), document_summary=summary_text)

        # 5. Rebuild BM25 index
        self._rebuild_bm25_from_store()
        self._answer_cache.clear()

        pages = sorted({c.page for c in chunks})
        logger.info(
            "Ingested '%s': %d pages, %d chunks",
            pdf_path.name, len(pages), len(chunks),
        )

        return IngestResponse(
            document=pdf_path.name,
            pages_processed=len(pages),
            chunks_created=len(chunks),
            summary=summary_text or None,
        )

    def ingest_directory(self, directory: Path | str) -> list[IngestResponse]:
        """Ingest all PDFs in a directory."""
        from app.ingestion import iter_pdfs
        results: list[IngestResponse] = []
        for pdf_path in iter_pdfs(directory):
            try:
                result = self.ingest_document(pdf_path)
                results.append(result)
            except Exception as exc:
                logger.error("Failed to ingest %s: %s", pdf_path.name, exc)
        return results

    # ── Query ────────────────────────────────────────────────────────────────

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        session_id: str | None = None,
    ) -> QueryResponse:
        """
        Answer a question using the full RAG pipeline.

        Automatically routes to the right query strategy:
          • Keyword-presence questions ("is X mentioned?") → direct text search → YES/NO
          • All other questions → hybrid retrieve → re-rank → LLM generate
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        k = top_k or self._settings.final_context_k
        corpus_key = self._corpus_key()
        use_cache = self._should_use_answer_cache(question, session_id)

        # ── No documents guard ────────────────────────────────────────────────
        if self._vector_store.count == 0:
            response = QueryResponse(
                answer=(
                    "No documents have been uploaded yet. "
                    "Please upload a PDF first, then ask your question."
                ),
                sources=[],
                confidence=0.0,
            )
            self._memory.add_turn(session_id, question, response.answer)
            return response

        if use_cache:
            cached = self._answer_cache.get_exact(question, corpus_key, k)
            if cached:
                self._memory.add_turn(session_id, question, cached.answer)
                return cached

        # ── Route: keyword-presence question ─────────────────────────────────
        if _is_keyword_presence_question(question):
            keyword = _extract_keyword(question)
            if keyword:
                logger.info("Detected keyword-presence query. Keyword: %r", keyword)
                response = self._keyword_presence_query(question, keyword)
                if use_cache:
                    question_embedding = self._embedding_model.embed_query(question)
                    self._answer_cache.set(question, question_embedding, corpus_key, k, response)
                self._memory.add_turn(session_id, question, response.answer)
                return response

        # ── Route: standard semantic RAG query ────────────────────────────────
        response = self._semantic_query(
            question,
            top_k,
            session_id=session_id,
            corpus_key=corpus_key,
            use_cache=use_cache,
        )
        self._memory.add_turn(session_id, question, response.answer)
        return response

    # ── Private: keyword-presence query ──────────────────────────────────────

    def _keyword_presence_query(self, question: str, keyword: str) -> QueryResponse:
        """
        Direct text search across all ingested chunks.
        Returns YES (with citations) or NO — no LLM needed.

        This is far more accurate than semantic search for "is X mentioned?" questions
        because cross-encoders penalise keyword-presence queries (they score passage
        relevance, not keyword containment).
        """
        all_chunks = self._retriever._corpus_chunks

        # Case-insensitive exact keyword search across all chunks
        kw_lower = keyword.lower()
        matching: list[Chunk] = [c for c in all_chunks if kw_lower in c.text.lower()]

        if matching:
            # Sort by page number for cleaner citations
            matching.sort(key=lambda c: (c.page, c.chunk_index))
            # De-duplicate by page (one citation per page)
            seen_pages: set[tuple[str, int]] = set()
            sources: list[SourceReference] = []
            for c in matching:
                key = (c.document, c.page)
                if key not in seen_pages:
                    seen_pages.add(key)
                    sources.append(SourceReference(
                        document=c.document,
                        page=c.page,
                        chunk=c.text,
                    ))
                if len(sources) >= 3:   # cap at 3 citation pages
                    break

            pages_found = sorted({s.page for s in sources})
            page_list = ", ".join(f"page {p}" for p in pages_found)
            answer = (
                f'YES — the keyword "{keyword}" is mentioned in the document '
                f"(found on {page_list})."
            )
            logger.info("Keyword %r found in %d chunks", keyword, len(matching))
            return QueryResponse(answer=answer, sources=sources, confidence=0.99)

        else:
            # Keyword genuinely not found in any ingested chunk
            docs = self._vector_store.list_documents()
            doc_list = ", ".join(docs) if docs else "the ingested documents"
            answer = (
                f'NO — the keyword "{keyword}" was not found anywhere in {doc_list}.'
            )
            logger.info("Keyword %r not found in corpus (%d chunks searched)", keyword, len(all_chunks))
            return QueryResponse(answer=answer, sources=[], confidence=0.97)

    # ── Private: standard semantic RAG ───────────────────────────────────────

    def _semantic_query(
        self,
        question: str,
        top_k: Optional[int] = None,
        session_id: str | None = None,
        corpus_key: str | None = None,
        use_cache: bool = True,
    ) -> QueryResponse:
        """Full hybrid BM25 + dense + cross-encoder + LLM pipeline with fallback."""
        settings = self._settings
        k = top_k or settings.final_context_k

        logger.info("Semantic query: %r", question[:120])

        # 1. Embed query
        query_embedding = self._embedding_model.embed_query(question)

        if use_cache:
            cached = self._answer_cache.get_semantic(
                query_embedding,
                corpus_key or self._corpus_key(),
                k,
            )
            if cached:
                return cached

        # 2. Hybrid retrieval (BM25 + Dense → RRF → Cross-encoder re-rank)
        retrieve_k = max(k, settings.rerank_top_k)
        retrieved = self._retriever.retrieve(
            query=question,
            query_embedding=query_embedding,
            vector_store=self._vector_store,
            top_k_final=retrieve_k,
        )

        top_score = retrieved[0].relevance_score if retrieved else 0.0
        logger.info(
            "Retrieved %d chunks. Top score: %.4f",
            len(retrieved), top_score,
        )

        # 3. LLM Fallback: if retrieval confidence is very low AND question is general
        if top_score < _FALLBACK_RELEVANCE_THRESHOLD and _is_general_question(question):
            logger.info(
                "Low retrieval score (%.4f) + general question detected — using LLM fallback.",
                top_score,
            )
            try:
                return generate_llm_fallback(
                    question,
                    conversation_history=self._memory.format_history(session_id),
                )
            except Exception as exc:
                logger.warning("LLM fallback failed (%s), proceeding with RAG answer.", exc)

        # 4. Generate answer with hallucination mitigation (standard RAG path)
        response = generate_answer(
            question,
            retrieved,
            top_k_context=k,
            conversation_history=self._memory.format_history(session_id),
        )
        if use_cache:
            self._answer_cache.set(question, query_embedding, corpus_key or self._corpus_key(), k, response)
        return response

    # ── Document Management ──────────────────────────────────────────────────

    def list_documents(self) -> list[str]:
        return self._vector_store.list_documents()

    def get_document_info(self, document_name: str) -> DocumentInfo:
        info = self._vector_store.get_document_info(document_name)
        return DocumentInfo(**info)

    def delete_document(self, document_name: str) -> int:
        count = self._vector_store.delete_document(document_name)
        self._rebuild_bm25_from_store()
        self._answer_cache.clear()
        return count

    # ── Private helpers ──────────────────────────────────────────────────────

    def _corpus_key(self) -> str:
        docs = self._vector_store.list_documents()
        payload = "|".join(docs) + f"|chunks:{self._vector_store.count}"
        return sha256(payload.encode("utf-8")).hexdigest()[:16]

    def _should_use_answer_cache(self, question: str, session_id: str | None) -> bool:
        if not self._settings.cache_enabled:
            return False
        has_history = bool(self._memory.get_history(session_id))
        if has_history and _CONTEXTUAL_FOLLOWUP_RE.search(question):
            return False
        return True

    def _rebuild_bm25_from_store(self) -> None:
        """Load all chunks from ChromaDB and rebuild the in-memory BM25 index."""
        if self._vector_store.count == 0:
            return
        try:
            result = self._vector_store._collection.get(include=["documents", "metadatas"])
            chunks: list[Chunk] = []
            for text, meta in zip(result["documents"], result["metadatas"]):  # type: ignore
                chunks.append(Chunk(
                    document=meta["document"],
                    page=meta["page"],
                    chunk_index=meta["chunk_index"],
                    text=text,
                    token_count=meta.get("token_count", len(text) // 4),
                ))
            self._retriever.rebuild_bm25(chunks)
        except Exception as exc:
            logger.warning("BM25 rebuild failed (store may be empty): %s", exc)
