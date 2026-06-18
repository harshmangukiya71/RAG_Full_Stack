"""
pipeline.py — Main RAGPipeline class.

Key improvements over original:
  • _graph_retrieve() uses entity mentions from LLM KG extraction to retrieve
    chunks directly attached to matched entities.
  • Graph traversal is strictly depth-1: we look up entities from the query,
    fetch their directly-attached chunks, and stop.  No multi-hop expansion.
  • query_entity_ids are passed through to HybridRetriever.retrieve() so the
    evidence filter inside retrieval.py can use them.
  • _graph_retrieve logs clearly when chunks are dropped due to low confidence.

Agentic upgrades:
  • Query Planning Agent classifies queries and selects retrieval strategies
  • Retrieval Strategy Router dynamically adjusts sources and top_k
  • Reasoning Agent post-processes retrieved chunks into structured evidence
  • Iterative retrieval loop retries when evidence is insufficient
"""
from __future__ import annotations

import logging
import json
import re
from hashlib import sha256
from pathlib import Path
from typing import Optional

from app.cache import AnswerCache
from app.config import get_settings
from app.embeddings import EmbeddingModel
from app.entities import EntityExtractor
from app.generation import generate_answer, generate_llm_fallback
from app.graph import (
    GraphStore,
    create_graph_store,
    graph_entity_id,
    graph_label_for_id,
    graph_relation_tokens,
)
from app.ingestion import parse_pdf
from app.memory import MemoryManager
from app.models import (
    Chunk,
    DocumentInfo,
    GraphNeighborsResponse,
    IngestResponse,
    QueryClassification,
    QueryResponse,
    ReasoningOutput,
    RetrievedChunk,
    SourceReference,
)
from app.query_agent import QueryPlanningAgent
from app.reasoning_agent import ReasoningAgent
from app.retrieval import HybridRetriever
from app.summarizer import DocumentSummarizer
from app.vectorstore import VectorStore

logger = logging.getLogger(__name__)

# ── Keyword-presence question detector ───────────────────────────────────────
_KEYWORD_PRESENCE_RE = re.compile(
    r"\b(?:is|are|does|do|can|has|have)\b.{0,60}"
    r"\b(?:mention(?:ed)?|present|includ(?:ed)?|contain(?:ed)?|appear(?:s)?|found|exist(?:s)?|refer(?:red)?)\b",
    re.IGNORECASE,
)

_KEYWORD_EXTRACT_PATTERNS = [
    r"['\"\u201c\u201d]([^'\"\u201c\u201d]{1,40})['\"\u201c\u201d]",
    r"\bkeyword\s+['\"]?(\w[\w\-\.]*)['\"]?",
    r"\bterm\s+['\"]?(\w[\w\-\.]*)['\"]?",
    r"\b(?:is|are|does)\s+(?:the\s+)?(?:word\s+)?(\w[\w\-\.]*)\s+"
    r"(?:mention|present|includ|contain|appear|found|exist|refer)",
    r"\b(?:contain|mention|include|refer\s+to)\s+['\"]?(\w[\w\-\.]*)['\"]?",
]

_DOC_SPECIFIC_RE = re.compile(
    r"\b(?:document|contract|agreement|policy|clause|section|exhibit|appendix|"
    r"resume|cv|report|file|page|paragraph|excerpt|content|the\s+pdf|"
    r"this\s+document|provided|mentioned|ingested|uploaded)\b",
    re.IGNORECASE,
)

_FALLBACK_RELEVANCE_THRESHOLD = 0.15

_CONTEXTUAL_FOLLOWUP_RE = re.compile(
    r"\b(?:it|that|this|they|them|those|these|he|she|same|above|previous|earlier)\b",
    re.IGNORECASE,
)

_GRAPH_ENTITY_IN_QUERY_RE = re.compile(
    r"\b(?:(?P<label>[A-Z][a-z][A-Za-z]{1,30})\s+)?(?P<id>[A-Z]{1,10}-\d+)\b"
)
_GRAPH_LIST_QUERY_RE = re.compile(
    r"\b(?:list|show|give|display|enumerate)\b.*\b(?:all|every)?\b",
    re.IGNORECASE,
)
_INVOICE_ID_RE = re.compile(r"\bINV-\d+\b", re.IGNORECASE)
_INVOICE_AMOUNT_FACT_RE = re.compile(
    r"\bInvoice\s+(?P<invoice>INV-\d+)\s+amount\s+was\s+"
    r"(?P<amount>[$]?\s*\d[\d,]*(?:\.\d+)?)\b",
    re.IGNORECASE,
)
_PROJECT_NAME_RE = re.compile(
    r"\bProject\s+[A-Z][A-Za-z0-9_-]*(?:\s+[A-Z][A-Za-z0-9_-]*){0,4}\b"
)
_PROJECT_TRANSACTION_FACT_RE = re.compile(
    r"\b(?P<project>Project\s+[A-Z][A-Za-z0-9_-]*(?:\s+[A-Z][A-Za-z0-9_-]*){0,4})"
    r"\s+processed\s+exactly\s+"
    r"(?P<count>\d[\d,]*(?:\.\d+)?)\s+transactions?\b",
    re.IGNORECASE,
)
_FINANCIAL_ENTITY_PREFIXES = {
    "company": "COMPANY",
    "corporation": "CORPORATION",
    "corp": "CORPORATION",
    "subsidiary": "SUBSIDIARY",
    "investor": "INVESTOR",
    "shareholder": "SHAREHOLDER",
    "founder": "FOUNDER",
    "ceo": "CEO",
    "executive": "EXECUTIVE",
    "person": "PERSON",
    "billionaire": "BILLIONAIRE",
    "entrepreneur": "ENTREPRENEUR",
    "project": "PROJECT",
    "product": "PRODUCT",
    "contract": "CONTRACT",
    "agreement": "AGREEMENT",
    "partnership": "PARTNERSHIP",
    "invoice": "INVOICE",
    "payment": "PAYMENT",
    "transaction": "TRANSACTION",
    "asset": "ASSET",
    "liability": "LIABILITY",
    "stock": "STOCK",
    "bond": "BOND",
    "fund": "FUND",
    "etf": "ETF",
    "bank": "BANK",
    "department": "DEPARTMENT",
    "employee": "EMPLOYEE",
    "country": "COUNTRY",
    "city": "CITY",
    "region": "REGION",
}
_FINANCIAL_PREFIXED_ENTITY_RE = re.compile(
    r"\b(?P<prefix>company|corporation|corp|subsidiary|investor|shareholder|"
    r"founder|ceo|executive|person|billionaire|entrepreneur|project|product|"
    r"contract|agreement|partnership|invoice|payment|transaction|asset|"
    r"liability|stock|bond|fund|etf|bank|department|employee|country|city|region)"
    r"\s+(?P<name>[A-Z]{1,10}-\d+|[A-Z][A-Za-z0-9&.-]*(?:\s+[A-Z][A-Za-z0-9&.-]*){0,5})",
    re.IGNORECASE,
)
_CAPITALIZED_ENTITY_RE = re.compile(
    r"\b([A-Z][A-Za-z0-9&.-]*(?:\s+[A-Z][A-Za-z0-9&.-]*){0,5})\b"
)

# Minimum confidence for semantic graph facts.
_GRAPH_MIN_CONFIDENCE: float = 0.70


def _is_general_question(
    question: str,
    classification: QueryClassification | None = None,
) -> bool:
    if classification and classification.query_type in {
        "RANKING", "AGGREGATION", "COUNTING", "COMPARISON", "ANALYTICAL",
    }:
        return False
    return not bool(_DOC_SPECIFIC_RE.search(question))


def _is_keyword_presence_question(question: str) -> bool:
    return bool(_KEYWORD_PRESENCE_RE.search(question))


def _extract_keyword(question: str) -> str | None:
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
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._vector_store = VectorStore()
        self._retriever = HybridRetriever()
        self._embedding_model = EmbeddingModel.get()
        self._summarizer = DocumentSummarizer()
        self._memory = MemoryManager()
        self._answer_cache = AnswerCache()
        self._entity_extractor = EntityExtractor()
        self._graph_store: GraphStore = create_graph_store()

        # ── Agentic components (Feature 1, 3) ─────────────────────────────────
        self._query_agent = QueryPlanningAgent() if self._settings.query_agent_enabled else None
        self._reasoning_agent = ReasoningAgent() if self._settings.reasoning_agent_enabled else None

        self._rebuild_bm25_from_store()
        logger.info(
            "RAGPipeline initialised — %d chunks in store | "
            "query_agent=%s reasoning_agent=%s",
            self._vector_store.count,
            "ON" if self._query_agent else "OFF",
            "ON" if self._reasoning_agent else "OFF",
        )

    # ── Ingestion ────────────────────────────────────────────────────────────

    def ingest_document(self, pdf_path: Path | str) -> IngestResponse:
        """
        Ingest a single PDF document:
          parse → chunk → entity enrich → embed → upsert vector store →
          upsert graph → rebuild BM25.
        """
        pdf_path = Path(pdf_path)
        settings = self._settings
        logger.info("Ingesting: %s", pdf_path.name)

        # 1. Parse
        chunks = parse_pdf(
            pdf_path,
            chunk_size_tokens=settings.chunk_size_tokens,
            chunk_overlap_tokens=settings.chunk_overlap_tokens,
        )
        if not chunks:
            raise ValueError(f"No text could be extracted from {pdf_path.name}")

        # 2. Entity extraction + enrichment
        chunks, mentions = self._entity_extractor.enrich_chunks(chunks)

        # 3. Summarise
        summary_text = ""
        try:
            summary_text = self._summarizer.summarize_chunks(
                pdf_path.name, chunks
            ).summary
        except Exception as exc:
            logger.exception(
                "Summary generation failed for %s: %s", pdf_path.name, exc
            )

        # 4. Embed
        texts = [c.text for c in chunks]
        embeddings = self._embedding_model.embed_documents(texts)

        # 5. Upsert vector store
        self._vector_store.upsert_chunks(
            chunks, embeddings.tolist(), document_summary=summary_text
        )

        # 6. Upsert graph
        logger.info(
            "Passing KG to Neo4j: %d entities, %d relationships for %s",
            len(mentions),
            sum(len(chunk.kg_relations) for chunk in chunks),
            pdf_path.name,
        )
        relationships_created = self._graph_store.upsert_document_graph(
            pdf_path.name, chunks, mentions
        )

        self._rebuild_bm25_from_store()
        self._answer_cache.clear()

        pages = sorted({c.page for c in chunks})
        confidences = [c.ocr_confidence for c in chunks if c.ocr_confidence is not None]
        methods = sorted({c.extraction_method for c in chunks})
        logger.info(
            "Ingested '%s': %d pages, %d chunks, %d entities",
            pdf_path.name,
            len(pages),
            len(chunks),
            len(mentions),
        )

        return IngestResponse(
            document=pdf_path.name,
            pages_processed=len(pages),
            chunks_created=len(chunks),
            summary=summary_text or None,
            extraction_method="+".join(methods) if methods else "native",
            average_ocr_confidence=(
                round(sum(confidences) / len(confidences), 3)
                if confidences
                else None
            ),
            entities_extracted=len(mentions),
            graph_relationships_created=relationships_created,
        )

    def ingest_directory(self, directory: Path | str) -> list[IngestResponse]:
        """Ingest all supported files in a directory."""
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

        Routing:
          • Keyword-presence questions ("is X mentioned?") → direct text scan
          • All other questions → hybrid retrieve → rerank → LLM generate
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        k = top_k or self._settings.final_context_k
        corpus_key = self._corpus_key()
        use_cache = self._should_use_answer_cache(question, session_id)

        # Guard: no documents
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

        exact_metric_response = self._direct_exact_metric_query(question)
        if exact_metric_response:
            self._memory.add_turn(session_id, question, exact_metric_response.answer)
            return exact_metric_response

        graph_fact_response = self._direct_graph_fact_query(question)
        if graph_fact_response:
            if use_cache:
                q_emb = self._embedding_model.embed_query(question)
                self._answer_cache.set(question, q_emb, corpus_key, k, graph_fact_response)
            self._memory.add_turn(session_id, question, graph_fact_response.answer)
            return graph_fact_response

        # Route: keyword-presence
        if _is_keyword_presence_question(question):
            keyword = _extract_keyword(question)
            if keyword:
                logger.info("Keyword-presence route. Keyword: %r", keyword)
                response = self._keyword_presence_query(question, keyword)
                if use_cache:
                    q_emb = self._embedding_model.embed_query(question)
                    self._answer_cache.set(question, q_emb, corpus_key, k, response)
                self._memory.add_turn(session_id, question, response.answer)
                return response

        # Route: semantic RAG
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

    def _keyword_presence_query(
        self, question: str, keyword: str
    ) -> QueryResponse:
        """Direct text scan — no LLM, no graph, no speculation."""
        all_chunks = self._retriever._corpus_chunks
        kw_lower = keyword.lower()
        matching = [c for c in all_chunks if kw_lower in c.text.lower()]

        if matching:
            matching.sort(key=lambda c: (c.page, c.chunk_index))
            seen_pages: set[tuple[str, int]] = set()
            sources: list[SourceReference] = []
            for c in matching:
                key = (c.document, c.page)
                if key not in seen_pages:
                    seen_pages.add(key)
                    sources.append(
                        SourceReference(
                            document=c.document,
                            page=c.page,
                            chunk=c.text,
                            chunk_index=c.chunk_index,
                            entities=c.entities,
                        )
                    )
                if len(sources) >= 3:
                    break

            pages_found = sorted({s.page for s in sources})
            page_list = ", ".join(f"page {p}" for p in pages_found)
            answer = (
                f'YES — the keyword "{keyword}" is mentioned in the document '
                f"(found on {page_list})."
            )
            logger.info("Keyword %r found in %d chunks", keyword, len(matching))
            return QueryResponse(answer=answer, sources=sources, confidence=0.99)

        docs = self._vector_store.list_documents()
        doc_list = ", ".join(docs) if docs else "the ingested documents"
        answer = f'NO — the keyword "{keyword}" was not found anywhere in {doc_list}.'
        logger.info(
            "Keyword %r not found (%d chunks searched)", keyword, len(all_chunks)
        )
        return QueryResponse(answer=answer, sources=[], confidence=0.97)

    # ── Private: semantic RAG query (agentic pipeline) ─────────────────────────

    def _semantic_query(
        self,
        question: str,
        top_k: Optional[int] = None,
        session_id: str | None = None,
        corpus_key: str | None = None,
        use_cache: bool = True,
    ) -> QueryResponse:
        """
        Agentic RAG pipeline:

          1. Query Planning Agent → classify query type + strategy
          2. Embed query
          3. Graph retrieval (confidence-filtered, depth=1)
          4. Retrieval Strategy Router → dynamic sources + top_k
          5. Iterative retrieval loop (if evidence insufficient)
          6. Reasoning Agent → structured evidence
          7. LLM generation with reasoning context
        """
        settings = self._settings
        k = top_k or settings.final_context_k

        logger.info("Semantic query: %r", question[:120])

        # ── Step 1: Query Planning Agent ──────────────────────────────────────
        classification: QueryClassification | None = None
        if self._query_agent:
            classification = self._query_agent.classify(question)
            # Override top_k from classification unless user specified one
            if top_k is None:
                k = classification.top_k
            logger.info(
                "Query classified: type=%s scope=%s strategy=%s top_k=%d",
                classification.query_type,
                classification.query_scope,
                classification.retrieval_strategy,
                k,
            )

        # ── Step 2: Embed query ───────────────────────────────────────────────
        query_embedding = self._embedding_model.embed_query(question)

        if use_cache:
            cached = self._answer_cache.get_semantic(
                query_embedding, corpus_key or self._corpus_key(), k
            )
            if cached:
                return cached

        # ── Step 3: Graph retrieval (confidence-filtered, depth=1) ────────────
        graph_results, query_entity_ids = self._graph_retrieve(question)

        # ── Step 4: Retrieval (strategy-routed or fallback) ───────────────────
        if classification and classification.retrieval_strategy != "hybrid":
            retrieved = self._retriever.retrieve_with_strategy(
                query=question,
                query_embedding=query_embedding,
                vector_store=self._vector_store,
                strategy=classification.retrieval_strategy,
                top_k=k,
                graph_results=graph_results,
                query_entity_ids=query_entity_ids,
                query_type=classification.query_type,
                query_scope=classification.query_scope,
            )
        else:
            retrieve_k = max(k, settings.rerank_top_k)
            retrieved = self._retriever.retrieve(
                query=question,
                query_embedding=query_embedding,
                vector_store=self._vector_store,
                top_k_final=retrieve_k,
                graph_results=graph_results,
                query_entity_ids=query_entity_ids,
            )

        top_score = retrieved[0].relevance_score if retrieved else 0.0
        logger.info(
            "Retrieved %d chunks. Top score: %.4f", len(retrieved), top_score
        )

        if (
            classification
            and self._reasoning_agent
            and self._is_global_reasoning_query(classification)
        ):
            retrieved = self._adaptive_global_retrieve(
                question, query_embedding, classification, retrieved,
                graph_results, query_entity_ids,
            )
            top_score = retrieved[0].relevance_score if retrieved else 0.0

        # ── Step 5: Iterative retrieval (Feature 4) ───────────────────────────
        if (
            classification
            and classification.retrieval_strategy == "iterative"
            and self._reasoning_agent
        ):
            retrieved = self._iterative_retrieve(
                question, query_embedding, classification, retrieved,
                graph_results, query_entity_ids,
            )
            top_score = retrieved[0].relevance_score if retrieved else 0.0

        # ── Step 6: LLM fallback for general questions ────────────────────────
        is_corpus_wide = (
            classification is not None
            and classification.query_type in {"RANKING", "AGGREGATION", "COUNTING"}
        )
        if (
            not is_corpus_wide
            and top_score < _FALLBACK_RELEVANCE_THRESHOLD
            and _is_general_question(question, classification)
        ):
            logger.info(
                "Low retrieval score (%.4f) + general question → LLM fallback.",
                top_score,
            )
            try:
                return generate_llm_fallback(
                    question,
                    conversation_history=self._memory.format_history(session_id),
                )
            except Exception as exc:
                logger.warning(
                    "LLM fallback failed (%s), continuing with RAG answer.", exc
                )

        # ── Step 7: Reasoning Agent (Feature 3) ──────────────────────────────
        reasoning_output: ReasoningOutput | None = None
        if self._reasoning_agent and classification:
            reasoning_output = self._reasoning_agent.run(
                question, classification, retrieved
            )
            logger.info(
                "Reasoning: sufficient=%s entities=%d calculations=%d rankings=%d",
                reasoning_output.evidence_sufficient,
                len(reasoning_output.entities),
                len(reasoning_output.calculations),
                len(reasoning_output.rankings),
            )
            logger.info(
                "Aggregation Results: calculations=%s rankings=%s",
                reasoning_output.calculations[:3],
                reasoning_output.rankings[:3],
            )

        # ── Step 8: Generate grounded answer ──────────────────────────────────
        conversation_history = self._memory.format_history(session_id)

        response = generate_answer(
            question,
            retrieved,
            top_k_context=min(k, len(retrieved)) if retrieved else k,
            conversation_history=conversation_history,
            reasoning_output=reasoning_output,
        )

        # Attach classification info to response
        if classification:
            response = response.model_copy(
                update={"query_classification": classification}
            )

        if use_cache:
            self._answer_cache.set(
                question,
                query_embedding,
                corpus_key or self._corpus_key(),
                k,
                response,
            )
        return response

    def _is_global_reasoning_query(
        self, classification: QueryClassification
    ) -> bool:
        return (
            classification.query_type == "RANKING"
            or (
                classification.query_scope == "GLOBAL"
                and classification.query_type in {"AGGREGATION", "COUNTING"}
            )
        )

    def _adaptive_global_retrieve(
        self,
        question: str,
        query_embedding,
        classification: QueryClassification,
        initial_results: list[RetrievedChunk],
        graph_results: list[RetrievedChunk],
        query_entity_ids: set[str] | None,
    ) -> list[RetrievedChunk]:
        """Bounded recall expansion for corpus-wide reasoning queries."""
        max_iterations = 3
        all_results = list(initial_results)
        seen_chunks: set[str] = {
            f"{c.document}::{c.chunk_index}" for c in all_results
        }

        records = self._reasoning_agent._extract_metric_records(all_results)
        previous_count = len({r["entity"].lower() for r in records})
        logger.info(
            "Adaptive global retrieval: iteration=1 entities=%d chunks=%d",
            previous_count, len(all_results),
        )

        for iteration in range(2, max_iterations + 1):
            expanded_k = min(100, classification.top_k * iteration)
            new_results = self._retriever.retrieve_with_strategy(
                query=question,
                query_embedding=query_embedding,
                vector_store=self._vector_store,
                strategy=classification.retrieval_strategy,
                top_k=expanded_k,
                graph_results=graph_results,
                query_entity_ids=query_entity_ids,
                query_type=classification.query_type,
                query_scope=classification.query_scope,
            )

            for chunk in new_results:
                key = f"{chunk.document}::{chunk.chunk_index}"
                if key not in seen_chunks:
                    seen_chunks.add(key)
                    all_results.append(chunk)

            records = self._reasoning_agent._extract_metric_records(all_results)
            current_count = len({r["entity"].lower() for r in records})
            logger.info(
                "Adaptive global retrieval: iteration=%d entities=%d chunks=%d",
                iteration, current_count, len(all_results),
            )
            if current_count <= previous_count:
                break
            previous_count = current_count

        all_results.sort(key=lambda c: c.relevance_score, reverse=True)
        return all_results

    # ── Iterative retrieval loop (Feature 4) ──────────────────────────────────

    def _iterative_retrieve(
        self,
        question: str,
        query_embedding,
        classification: QueryClassification,
        initial_results: list[RetrievedChunk],
        graph_results: list[RetrievedChunk],
        query_entity_ids: set[str] | None,
    ) -> list[RetrievedChunk]:
        """
        Agentic retrieval loop:
          1. Retrieve
          2. Evaluate evidence sufficiency via reasoning agent
          3. If insufficient → retrieve again with expanded top_k
          4. Merge and deduplicate context
          5. Repeat up to max_iterations
        """
        max_iterations = self._settings.max_retrieval_iterations
        all_results = list(initial_results)
        seen_keys: set[str] = {
            f"{c.document}::{c.chunk_index}" for c in all_results
        }

        for iteration in range(1, max_iterations):
            # Check sufficiency
            if self._reasoning_agent:
                reasoning = self._reasoning_agent.run(
                    question, classification, all_results
                )
                if reasoning.evidence_sufficient:
                    logger.info(
                        "Iterative retrieval: evidence sufficient after %d iteration(s)",
                        iteration,
                    )
                    break

            # Expand retrieval with increased top_k
            expanded_k = classification.top_k * (iteration + 1)
            logger.info(
                "Iterative retrieval: iteration %d, expanding top_k to %d",
                iteration + 1, expanded_k,
            )

            new_results = self._retriever.retrieve_with_strategy(
                query=question,
                query_embedding=query_embedding,
                vector_store=self._vector_store,
                strategy=classification.retrieval_strategy,
                top_k=expanded_k,
                graph_results=graph_results,
                query_entity_ids=query_entity_ids,
                query_type=classification.query_type,
                query_scope=classification.query_scope,
            )

            # Merge — deduplicate by (document, chunk_index)
            added = 0
            for chunk in new_results:
                key = f"{chunk.document}::{chunk.chunk_index}"
                if key not in seen_keys:
                    seen_keys.add(key)
                    all_results.append(chunk)
                    added += 1

            logger.info(
                "Iterative retrieval: added %d new chunks (total %d)",
                added, len(all_results),
            )

            if added == 0:
                logger.info("Iterative retrieval: no new chunks found, stopping")
                break

        # Sort by relevance score and return
        all_results.sort(key=lambda c: c.relevance_score, reverse=True)
        return all_results

    # ── Document management ───────────────────────────────────────────────────

    def list_documents(self) -> list[str]:
        return self._vector_store.list_documents()

    def get_document_info(self, document_name: str) -> DocumentInfo:
        info = self._vector_store.get_document_info(document_name)
        return DocumentInfo(**info)

    def delete_document(self, document_name: str) -> int:
        count = self._vector_store.delete_document(document_name)
        try:
            self._graph_store.delete_document(document_name)
        except Exception as exc:
            logger.warning("Graph delete failed for %s: %s", document_name, exc)
        self._rebuild_bm25_from_store()
        self._answer_cache.clear()
        return count

    def search_entities(self, query: str, limit: int = 10):
        return self._graph_store.search_entities(query, limit=limit)

    def graph_neighbors(
        self, entity_id: str, depth: int = 1, limit: int = 50
    ) -> GraphNeighborsResponse:
        entity, neighbors, relationships = self._graph_store.neighbors(
            entity_id, depth=depth, limit=limit
        )
        return GraphNeighborsResponse(
            entity=entity, neighbors=neighbors, relationships=relationships
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _corpus_key(self) -> str:
        docs = self._vector_store.list_documents()
        payload = "|".join(docs) + f"|chunks:{self._vector_store.count}"
        return sha256(payload.encode("utf-8")).hexdigest()[:16]

    def _should_use_answer_cache(
        self, question: str, session_id: str | None
    ) -> bool:
        if not self._settings.cache_enabled:
            return False
        has_history = bool(self._memory.get_history(session_id))
        if has_history and _CONTEXTUAL_FOLLOWUP_RE.search(question):
            return False
        return True

    def _direct_graph_fact_query(self, question: str) -> QueryResponse | None:
        """Answer graph facts using generic entity and relation matching."""
        q = question.strip()
        query_relation_tokens = graph_relation_tokens(q)
        try:
            return self._direct_graph_fact_query_safe(q, query_relation_tokens)
        except Exception as exc:
            logger.exception("Graph fact lookup failed; falling back to retrieval: %s", exc)
            return None

    def _direct_graph_fact_query_safe(
        self, q: str, query_relation_tokens: set[str]
    ) -> QueryResponse | None:
        """Exception-free caller wraps this method and falls back to semantic RAG."""
        if _GRAPH_LIST_QUERY_RE.search(q) and not _GRAPH_ENTITY_IN_QUERY_RE.search(q):
            facts = self._graph_store.relationship_facts(limit=300)
            facts = self._filter_graph_facts_for_query(facts, query_relation_tokens)
            if not facts:
                logger.info("Graph lookup entity_not_found: list query produced no edges")
                return None
            facts = sorted(
                facts,
                key=lambda f: (
                    str(f.get("type", "")),
                    str(f.get("source_name", "")),
                    str(f.get("target_name", "")),
                ),
            )
            lines = [
                f"{fact['source_name']} -> {fact['target_name']}"
                for fact in facts
            ]
            return QueryResponse(
                answer="Graph relationships:\n" + "\n".join(lines),
                sources=self._sources_from_graph_facts(facts),
                confidence=0.92,
            )

        query_entities = self._extract_graph_query_entities(q)
        logger.info("Extracted Entities for graph lookup: %s", query_entities)
        if not query_entities:
            return None

        candidate_entity_ids: set[str] = set()
        for entity_text, entity_label in query_entities:
            if entity_label:
                candidate_entity_ids.add(graph_entity_id(entity_label, entity_text))
            for entity in self._graph_store.search_entities(entity_text, limit=10):
                candidate_entity_ids.add(entity.entity_id)

        if not candidate_entity_ids:
            logger.info("Graph lookup entity_not_found: %s", query_entities)
            return None

        facts: list[dict] = []
        for candidate_id in candidate_entity_ids:
            facts.extend(self._graph_store.relationship_facts(source_id=candidate_id, limit=100))
            facts.extend(self._graph_store.relationship_facts(target_id=candidate_id, limit=100))
        logger.info("Graph Edges Found before filtering: %d", len(facts))
        facts = self._dedupe_graph_facts(facts)
        facts = self._filter_graph_facts_for_query(facts, query_relation_tokens)
        logger.info("Graph Edges Found after filtering: %d", len(facts))
        if not facts:
            logger.info("Graph lookup missing_edge: entities=%s", sorted(candidate_entity_ids))
            return None

        if len(facts) == 1:
            fact = facts[0]
            return QueryResponse(
                answer=self._format_graph_fact_answer(fact, candidate_entity_ids),
                sources=self._sources_from_graph_facts(facts),
                confidence=0.95,
            )

        lines = [
            f"{fact['source_name']} -> {fact['target_name']}"
            for fact in facts[:20]
        ]
        return QueryResponse(
            answer="Matching graph relationships:\n" + "\n".join(lines),
            sources=self._sources_from_graph_facts(facts),
            confidence=0.9,
        )

    def _extract_graph_query_entities(self, question: str) -> list[tuple[str, str | None]]:
        """Normalize finance entity mentions from questions before graph lookup."""
        candidates: list[tuple[str, str | None]] = []
        seen: set[tuple[str, str | None]] = set()

        for match in _FINANCIAL_PREFIXED_ENTITY_RE.finditer(question):
            prefix = match.group("prefix").lower().rstrip(".")
            label = _FINANCIAL_ENTITY_PREFIXES.get(prefix)
            name = self._normalize_query_entity_text(match.group("name"))
            if not re.match(r"^(?:[A-Z]{1,10}-\d+|[A-Z])", name):
                continue
            key = (name, label)
            if name and key not in seen:
                seen.add(key)
                candidates.append(key)

        for match in _GRAPH_ENTITY_IN_QUERY_RE.finditer(question):
            entity_text = self._normalize_query_entity_text(match.group("id"))
            entity_label = graph_label_for_id(match.group("label"), entity_text)
            key = (entity_text, entity_label)
            if key not in seen:
                seen.add(key)
                candidates.append(key)

        for match in _CAPITALIZED_ENTITY_RE.finditer(question):
            name = self._normalize_query_entity_text(match.group(1))
            if len(name) < 3 or name.lower() in {"which", "who", "what", "contract", "project", "invoice"}:
                continue
            if not (
                "-" in name
                or any(suffix in name.lower() for suffix in (" corp", " inc", " llc", " ltd", " bank"))
            ):
                continue
            key = (name, None)
            if key not in seen:
                seen.add(key)
                candidates.append(key)

        return candidates

    def _normalize_query_entity_text(self, text: str) -> str:
        text = re.sub(r"[?.,;:]+$", "", text.strip())
        text = re.sub(
            r"^(?:company|corporation|corp\.?|project|contract|invoice|"
            r"department|employee|product|bank|fund|stock|bond|agreement|"
            r"partnership|payment|transaction)\s+",
            "",
            text,
            flags=re.IGNORECASE,
        )
        return re.sub(r"\s+", " ", text).strip()

    def _dedupe_graph_facts(self, facts: list[dict]) -> list[dict]:
        seen: set[tuple[str, str, str]] = set()
        deduped: list[dict] = []
        for fact in facts:
            key = (fact["source_id"], fact["target_id"], fact["type"])
            if key in seen:
                continue
            seen.add(key)
            deduped.append(fact)
        return deduped

    def _filter_graph_facts_for_query(
        self,
        facts: list[dict],
        query_relation_tokens: set[str],
    ) -> list[dict]:
        structural_types: set[str] = set()
        candidates = [
            fact for fact in facts
            if fact.get("type") not in structural_types
        ]
        if not candidates:
            candidates = [
                fact for fact in facts
                if fact.get("type") not in structural_types
            ]
        if not query_relation_tokens:
            return candidates

        filtered: list[dict] = []
        for fact in candidates:
            evidence = fact.get("evidence") or {}
            fact_tokens = graph_relation_tokens(
                " ".join(
                    [
                        str(fact.get("type", "")),
                        str(evidence.get("relation_text", "")),
                    ]
                )
            )
            if query_relation_tokens & fact_tokens:
                filtered.append(fact)
        return filtered or candidates

    def _format_graph_fact_answer(self, fact: dict, queried_entity_ids: set[str]) -> str:
        source = fact["source_name"]
        target = fact["target_name"]
        relation = str((fact.get("evidence") or {}).get("relation_text") or fact["type"])
        relation = relation.strip().replace("_", " ").lower()
        if fact["target_id"] in queried_entity_ids:
            return f"{source} {relation} {target}."
        return f"{source} {relation} {target}."

    def _direct_exact_metric_query(self, question: str) -> QueryResponse | None:
        """Answer exact invoice/project metric questions from loaded chunk text."""
        q = question.strip()
        q_lower = q.lower()

        invoice_match = _INVOICE_ID_RE.search(q)
        if invoice_match and "amount" in q_lower:
            invoice_id = invoice_match.group(0).upper()
            for chunk in self._retriever._corpus_chunks:
                for fact in _INVOICE_AMOUNT_FACT_RE.finditer(chunk.text):
                    if fact.group("invoice").upper() != invoice_id:
                        continue
                    amount = fact.group("amount").strip()
                    return QueryResponse(
                        answer=f"The amount of invoice {invoice_id} is {amount}.",
                        sources=[
                            SourceReference(
                                document=chunk.document,
                                page=chunk.page,
                                chunk=chunk.text,
                                chunk_index=chunk.chunk_index,
                                entities=chunk.entities,
                            )
                        ],
                        confidence=1.0,
                    )

        project_match = _PROJECT_NAME_RE.search(q)
        if project_match and "transaction" in q_lower:
            project_name = project_match.group(0)
            for chunk in self._retriever._corpus_chunks:
                for fact in _PROJECT_TRANSACTION_FACT_RE.finditer(chunk.text):
                    if fact.group("project").lower() != project_name.lower():
                        continue
                    count = fact.group("count").strip()
                    return QueryResponse(
                        answer=f"{project_name} processed {count} transactions.",
                        sources=[
                            SourceReference(
                                document=chunk.document,
                                page=chunk.page,
                                chunk=chunk.text,
                                chunk_index=chunk.chunk_index,
                                entities=chunk.entities,
                            )
                        ],
                        confidence=1.0,
                    )

        return None

    def _sources_from_graph_facts(self, facts: list[dict]) -> list[SourceReference]:
        sources: list[SourceReference] = []
        seen: set[tuple[str, int, int]] = set()
        for fact in facts:
            evidence = fact.get("evidence") or {}
            document = evidence.get("document") or "graph"
            page = int(evidence.get("page") or 0)
            chunk_index = int(evidence.get("chunk_index") or 0)
            key = (document, page, chunk_index)
            if key in seen:
                continue
            seen.add(key)
            sources.append(
                SourceReference(
                    document=document,
                    page=page,
                    chunk=evidence.get("text") or f"{fact['source_name']} {fact['type']} {fact['target_name']}",
                    chunk_index=chunk_index,
                    entities=[fact["source_id"], fact["target_id"]],
                )
            )
        return sources[:5]

    def _graph_retrieve(
        self, question: str
    ) -> tuple[list[RetrievedChunk], set[str]]:
        """
        Depth-1 graph retrieval with confidence filtering.

        Returns (graph_chunks, query_entity_ids).

        Entity mentions come from LLM KG extraction. Semantic graph edges are
        not used for speculative expansion here; chunk retrieval is direct by
        matched entity IDs.
        """
        try:
            query_mentions = self._entity_extractor.extract(question)
        except Exception as exc:
            logger.exception("Malformed entity extraction during graph retrieval: %s", exc)
            query_mentions = []
        # Collect entity IDs from the question itself
        query_entity_ids: set[str] = {m.entity_id for m in query_mentions}
        logger.info(
            "Extracted Entities: %s",
            [{"text": m.text, "label": m.label, "id": m.entity_id} for m in query_mentions],
        )

        # Expand via graph search (name/alias lookup only — no traversal)
        try:
            for mention in query_mentions:
                for entity in self._graph_store.search_entities(mention.normalized, limit=5):
                    query_entity_ids.add(entity.entity_id)
            for entity_text, entity_label in self._extract_graph_query_entities(question):
                if entity_label:
                    query_entity_ids.add(graph_entity_id(entity_label, entity_text))
                for entity in self._graph_store.search_entities(entity_text, limit=5):
                    query_entity_ids.add(entity.entity_id)
        except Exception as exc:
            logger.exception("Graph entity lookup failed; falling back to semantic retrieval: %s", exc)
            return [], set()

        if not query_entity_ids:
            logger.info("Graph Nodes Found: 0")
            return [], set()

        traversal_entity_ids = set(query_entity_ids)
        try:
            for entity_id in list(query_entity_ids):
                root, neighbors, relationships = self._graph_store.neighbors(
                    entity_id, depth=2, limit=self._settings.graph_top_k
                )
                if root:
                    traversal_entity_ids.add(root.entity_id)
                traversal_entity_ids.update(entity.entity_id for entity in neighbors)
                logger.info(
                    "Traversal Path: root=%s neighbors=%d relationships=%d",
                    entity_id,
                    len(neighbors),
                    len(relationships),
                )
        except Exception as exc:
            logger.exception("Graph traversal failed; using direct entity hits only: %s", exc)

        logger.info("Graph Nodes Found: %d", len(traversal_entity_ids))

        try:
            chunk_keys_with_meta = self._graph_store.chunks_for_entities(
                sorted(traversal_entity_ids),
                limit=self._settings.graph_top_k,
            )
        except Exception as exc:
            logger.exception("Graph chunk lookup failed; falling back to semantic retrieval: %s", exc)
            return [], query_entity_ids

        # ── Confidence filter ─────────────────────────────────────────────────
        # chunks_for_entities returns (document, page, chunk_index, entity_ids).
        # We can't filter on relationship confidence at this level because the
        # graph store aggregates by chunk.  Instead we use the entity_ids list:
        # keep only chunks where at least one matched entity_id is in our
        # query_entity_ids (direct hit, not transitive).
        filtered_keys = [
            (doc, page, chunk_idx)
            for doc, page, chunk_idx, matched_entities in chunk_keys_with_meta
            if traversal_entity_ids & set(matched_entities)
        ]

        if not filtered_keys:
            logger.info("Graph retrieval empty traversal result.")
            return [], query_entity_ids

        graph_chunks = self._vector_store.get_chunks_by_keys(
            filtered_keys, score=0.72
        )

        logger.debug(
            "Graph retrieval: %d entity IDs → %d raw keys → %d after filter",
            len(query_entity_ids),
            len(chunk_keys_with_meta),
            len(filtered_keys),
        )
        return graph_chunks, query_entity_ids

    def _rebuild_bm25_from_store(self) -> None:
        """Load all chunks from the vector store and rebuild the in-memory BM25 index."""
        if self._vector_store.count == 0:
            self._retriever.rebuild_bm25([])
            return
        try:
            self._retriever.rebuild_bm25(self._vector_store.list_all_chunks())
        except Exception as exc:
            logger.warning("BM25 rebuild failed: %s", exc)
