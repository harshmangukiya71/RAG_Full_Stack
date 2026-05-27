"""
models.py — shared Pydantic schemas used across the API.
"""
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


# ── Internal data structures ────────────────────────────────────────────────

class Chunk(BaseModel):
    """A single text chunk with full provenance metadata."""
    document: str               # original filename (e.g. "NDA-VendorX.pdf")
    page: int                   # 1-indexed page number
    chunk_index: int            # position within the document
    text: str                   # the raw chunk text
    token_count: int            # approximate BPE token count


class RetrievedChunk(BaseModel):
    """A chunk augmented with a relevance score after retrieval/re-ranking."""
    document: str
    page: int
    chunk_index: int
    chunk: str                  # alias: text content
    relevance_score: float      # 0–1, higher = more relevant


# ── API request / response ──────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=5, max_length=2000,
                          description="The user's natural-language question.")
    top_k: Optional[int] = Field(None, ge=1, le=10,
                                 description="Override default context chunks (default=5).")
    session_id: Optional[str] = Field(None, max_length=128,
                                      description="Stable chat session id for conversational memory.")


class SourceReference(BaseModel):
    document: str               # filename / document title
    page: int                   # page number
    chunk: str                  # the retrieved text chunk used


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceReference]
    confidence: float = Field(..., ge=0.0, le=1.0)


# ── Ingestion ───────────────────────────────────────────────────────────────

class IngestResponse(BaseModel):
    document: str
    pages_processed: int
    chunks_created: int
    summary: Optional[str] = None
    status: str = "success"


class DocumentInfo(BaseModel):
    document: str
    total_chunks: int
    pages: list[int]
    summary: Optional[str] = None


# ── Evaluation ──────────────────────────────────────────────────────────────

class EvalPair(BaseModel):
    question: str
    expected_document: str
    expected_page: int
    answer_hint: str = ""


class EvalResult(BaseModel):
    question: str
    expected_document: str
    expected_page: int
    retrieved_top5: list[dict]          # top-5 retrieved chunks with scores
    hit_at_1: bool                      # correct chunk is rank 1
    hit_at_3: bool                      # correct chunk in top 3
    hit_at_5: bool                      # correct chunk in top 5
    rank: int                           # rank of correct result (0 = not found)
    reciprocal_rank: float              # 1/rank, or 0.0 if not found


class EvalReport(BaseModel):
    total_questions: int
    # ── Hit counts ──────────────────────────────────────────────────────────
    hits_at_1: int
    hits_at_3: int
    hits_at_5: int
    # ── Recall@K (fraction of questions where correct chunk is in top K) ────
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    # ── MRR (Mean Reciprocal Rank) ───────────────────────────────────────────
    mrr: float
    # ── Legacy alias kept for backward compat ───────────────────────────────
    precision_at_3: float               # = recall_at_3 for single-answer QA
    hits: int                           # = hits_at_3
    results: list[EvalResult]
