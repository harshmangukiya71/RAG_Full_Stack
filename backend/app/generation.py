"""
generation.py — LLM generation via NVIDIA NIM with hallucination mitigation.

Hallucination mitigation (multi-layered):
  1. Context-only system prompt: LLM is explicitly forbidden from using external knowledge.
  2. LLM Fallback: If no relevant chunks are retrieved and question seems general,
     fall back to a direct LLM answer (clearly labelled as such).
  3. Confidence scoring: Computed as weighted combination of:
       • Top chunk re-ranking score (semantic relevance)
       • Token overlap between answer and retrieved context (faithfulness proxy)
     If computed confidence < 0.35 we append a caveat to the answer.
  4. Source citation enforcement: sources are pulled from the retrieved chunks
     deterministically, not from LLM hallucination.

Works with: any uploaded PDF — legal, resume, story, technical, medical, etc.
"""
from __future__ import annotations

import logging
import re
from typing import Sequence

from openai import OpenAI

from app.config import get_settings
from app.models import QueryResponse, RetrievedChunk, SourceReference

logger = logging.getLogger(__name__)

# ── System prompt (RAG mode) ─────────────────────────────────────────────────
_SYSTEM_PROMPT = """You are a helpful, precise document assistant. Your job is to answer questions
based strictly on the provided document excerpts. Follow these rules:

RULES:
1. Answer ONLY from the provided context. Do NOT use external knowledge or make assumptions.
2. If the context does not contain the answer, say exactly: "The information requested was not found in the provided document excerpts."
3. Be specific and accurate: include exact names, numbers, dates, and details when present.
4. Match your tone to the document type — factual for reports, friendly for stories, precise for contracts.
5. Do NOT fabricate, infer, or extrapolate any information not explicitly present in the context.
"""

# ── System prompt (LLM fallback mode) ────────────────────────────────────────
_FALLBACK_SYSTEM_PROMPT = """You are a helpful, knowledgeable assistant. Answer the user's question
clearly and accurately using your general knowledge. Be concise, factual, and friendly.
If the question is ambiguous, state your assumptions briefly."""

_CONTEXT_TEMPLATE = """DOCUMENT EXCERPTS:
{context}

{history_block}

USER QUESTION: {question}

Answer based solely on the excerpts above. Use the conversation history only to
understand references in the question; do not treat it as document evidence."""

_HISTORY_TEMPLATE = """RECENT CONVERSATION HISTORY:
{history}"""


def _build_context(chunks: list[RetrievedChunk]) -> str:
    parts: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(
            f"[Excerpt {i}] Document: {chunk.document} | Page: {chunk.page}\n"
            f"{chunk.chunk}"
        )
    return "\n\n---\n\n".join(parts)


def _token_overlap_score(answer: str, context: str) -> float:
    """
    Compute token-level recall: fraction of unique answer tokens found in context.
    Serves as a faithfulness proxy — high overlap means answer is grounded in context.
    """
    def tokenize(text: str) -> set[str]:
        text = re.sub(r"[^\w\s]", " ", text.lower())
        tokens = set(text.split())
        # Remove stopwords (minimal list)
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "of", "in",
                     "to", "and", "or", "that", "this", "it", "for", "on", "with",
                     "as", "at", "by", "from", "be", "has", "have", "had", "not"}
        return tokens - stopwords

    answer_tokens = tokenize(answer)
    context_tokens = tokenize(context)

    if not answer_tokens:
        return 0.0
    overlap = len(answer_tokens & context_tokens)
    return overlap / len(answer_tokens)


def _compute_confidence(
    top_relevance_score: float,
    token_overlap: float,
    settings,
) -> float:
    """
    Confidence = weighted combination of retrieval quality + answer faithfulness.

      • top_relevance_score (0–1): how semantically relevant the top chunk is.
      • token_overlap (0–1): how much of the answer appears verbatim in context.

    The combined score is clipped to [0, 1].
    """
    raw = (0.6 * top_relevance_score) + (0.4 * token_overlap)
    scaled = min(1.0, raw * settings.confidence_scale_factor)
    return round(scaled, 3)


def generate_answer(
    question: str,
    retrieved_chunks: list[RetrievedChunk],
    top_k_context: int | None = None,
    conversation_history: str = "",
) -> QueryResponse:
    """
    Generate an answer grounded in retrieved_chunks using NVIDIA NIM.

    Steps:
      1. Check if top chunk relevance >= threshold; refuse if not.
      2. Build context from top-k retrieved chunks.
      3. Call NVIDIA NIM (OpenAI-compatible API).
      4. Compute confidence score.
      5. Return structured QueryResponse.
    """
    settings = get_settings()
    k = top_k_context or settings.final_context_k
    context_chunks = retrieved_chunks[:k]

    # ── Step 1: Refusal guard (only if zero chunks came through) ─────────────
    # Note: the empty-store case is handled upstream in pipeline.py before
    # generate_answer() is called. If we reach here, we always have some chunks.
    # Low relevance scores still get sent to the LLM — it will correctly say
    # "not found" if the context doesn't contain the answer.
    top_score = context_chunks[0].relevance_score if context_chunks else 0.0
    logger.info(
        "Generating answer: %d chunks, top relevance=%.4f",
        len(context_chunks), top_score,
    )
    if not context_chunks:
        # Should not normally reach here (pipeline guards against this)
        return QueryResponse(
            answer="No relevant content was retrieved. Please upload a document first.",
            sources=[],
            confidence=0.0,
        )


    # ── Step 2: Build context ─────────────────────────────────────────────
    context_text = _build_context(context_chunks)
    history_block = _HISTORY_TEMPLATE.format(history=conversation_history) if conversation_history else ""
    user_content = _CONTEXT_TEMPLATE.format(
        context=context_text,
        history_block=history_block,
        question=question,
    )

    # ── Step 3: Call NVIDIA NIM ───────────────────────────────────────────
    client = OpenAI(
        base_url=settings.nvidia_base_url,
        api_key=settings.nvidia_api_key,
    )

    try:
        response = client.chat.completions.create(
            model=settings.nvidia_model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            max_tokens=settings.nvidia_max_tokens,
            temperature=settings.nvidia_temperature,
            stream=False,
        )
        raw_answer = response.choices[0].message.content or ""
    except Exception as exc:
        logger.exception("NVIDIA NIM API call failed: %s", exc)
        raise

    # ── Step 4: Confidence scoring ────────────────────────────────────────
    top_relevance = context_chunks[0].relevance_score
    overlap = _token_overlap_score(raw_answer, context_text)
    confidence = _compute_confidence(top_relevance, overlap, settings)

    # Append caveat if confidence is low
    if confidence < 0.35 and "not found" not in raw_answer.lower():
        raw_answer += (
            "\n\n⚠️ *Low confidence: the retrieved context may not fully cover this question. "
            "Please verify against the original document.*"
        )

    # ── Step 5: Build structured response ────────────────────────────────
    sources = [
        SourceReference(
            document=chunk.document,
            page=chunk.page,
            chunk=chunk.chunk,
        )
        for chunk in context_chunks
    ]

    return QueryResponse(
        answer=raw_answer.strip(),
        sources=sources,
        confidence=confidence,
    )


def generate_llm_fallback(question: str, conversation_history: str = "") -> QueryResponse:
    """
    Answer a question directly from LLM general knowledge (no document context).

    Used when:
      - No documents are ingested, OR
      - Retrieved chunks have very low relevance AND question seems general.

    The response is clearly labelled so the user knows it's not grounded in their docs.
    """
    settings = get_settings()
    client = OpenAI(base_url=settings.nvidia_base_url, api_key=settings.nvidia_api_key)

    logger.info("LLM fallback: answering '%s' from general knowledge", question[:80])

    try:
        user_content = question
        if conversation_history:
            user_content = (
                "RECENT CONVERSATION HISTORY:\n"
                f"{conversation_history}\n\n"
                f"USER QUESTION: {question}"
            )

        response = client.chat.completions.create(
            model=settings.nvidia_model,
            messages=[
                {"role": "system", "content": _FALLBACK_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            max_tokens=settings.nvidia_max_tokens,
            temperature=0.4,   # slightly warmer for general Q&A
            stream=False,
        )
        raw_answer = response.choices[0].message.content or ""
    except Exception as exc:
        logger.exception("LLM fallback API call failed: %s", exc)
        raise

    # Prepend a clear indicator that this is NOT from the uploaded documents
    labelled = (
        "🌐 **General Knowledge Answer** *(No relevant content found in your documents — "
        "answering from general knowledge)*\n\n"
        + raw_answer.strip()
    )

    return QueryResponse(
        answer=labelled,
        sources=[],
        confidence=0.5,   # medium confidence — LLM-only, not grounded
    )
