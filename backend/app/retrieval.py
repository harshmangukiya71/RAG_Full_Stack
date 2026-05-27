"""
retrieval.py — Hybrid BM25 + Dense Vector retrieval with Cross-Encoder Re-ranking.

Strategy (chosen for legal documents):
  1. BM25 (rank_bm25): Legal queries often contain exact legal terms like
     "limitation of liability", "force majeure", "indemnification". BM25 exact-match
     natively handles these without semantic approximation.
  2. Dense vector search (ChromaDB / BGE): Handles paraphrase ("notice period" ↔
     "termination notice"), synonyms, and cross-clause reasoning.
  3. Reciprocal Rank Fusion (RRF): Merges both ranked lists without requiring score
     calibration — robust and proven in IR literature.
  4. Cross-Encoder re-ranking (ms-marco-MiniLM-L-6-v2): Jointly scores (query, chunk)
     pairs — far more accurate than bi-encoder retrieval. Applied only to top-20 fused
     results to bound latency.
"""
from __future__ import annotations

import logging
import math
from typing import Sequence

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from app.config import get_settings
from app.models import Chunk, RetrievedChunk

logger = logging.getLogger(__name__)

# Cross-encoder model — loaded once (singleton-like via module-level cache)
_cross_encoder: CrossEncoder | None = None


def _get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        logger.info("Loading cross-encoder model...")
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
        logger.info("Cross-encoder loaded.")
    return _cross_encoder


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenisation for BM25 corpus indexing."""
    return text.lower().split()


def _tokenize_query_with_prefix(query: str, corpus_vocab: set[str]) -> list[str]:
    """
    Tokenise a query and expand partial tokens to matching corpus words.

    Example: query token 'contrac' expands to ['contract', 'contracts', 'contractual']
    if those words exist in the corpus vocabulary. This gives prefix/partial-match
    behaviour within BM25 without requiring a full-text search engine.

    Short tokens (≤ 2 chars) are NOT expanded to avoid noise.
    """
    tokens = query.lower().split()
    expanded: list[str] = []
    for tok in tokens:
        if len(tok) > 2 and tok not in corpus_vocab:
            # Find all corpus words that start with this token (prefix match)
            matches = [w for w in corpus_vocab if w.startswith(tok) and w != tok]
            if matches:
                expanded.extend(matches[:5])   # cap at 5 expansions per token
                logger.debug("Prefix expansion: %r → %s", tok, matches[:5])
            else:
                expanded.append(tok)
        else:
            expanded.append(tok)
    return expanded


def _reciprocal_rank_fusion(
    *ranked_lists: list[RetrievedChunk],
    k: int = 60,
) -> list[RetrievedChunk]:
    """
    Merge multiple ranked lists via Reciprocal Rank Fusion.

    RRF score = Σ 1 / (k + rank_i)   for each list i that contains the item.

    k=60 is the standard constant from the original RRF paper (Cormack 2009).
    """
    scores: dict[str, float] = {}
    chunk_map: dict[str, RetrievedChunk] = {}

    for ranked_list in ranked_lists:
        for rank, chunk in enumerate(ranked_list, start=1):
            uid = f"{chunk.document}::{chunk.chunk_index}"
            scores[uid] = scores.get(uid, 0.0) + 1.0 / (k + rank)
            chunk_map[uid] = chunk

    # Sort by descending RRF score
    sorted_uids = sorted(scores, key=lambda u: scores[u], reverse=True)

    fused: list[RetrievedChunk] = []
    for uid in sorted_uids:
        c = chunk_map[uid]
        fused.append(RetrievedChunk(
            document=c.document,
            page=c.page,
            chunk_index=c.chunk_index,
            chunk=c.chunk,
            relevance_score=round(scores[uid], 6),
        ))
    return fused


class HybridRetriever:
    """
    Hybrid retriever: BM25 + Dense + Cross-Encoder re-ranking.
    Must be refreshed (rebuild_bm25) after new documents are ingested.
    """

    def __init__(self) -> None:
        self._corpus_chunks: list[Chunk] = []
        self._bm25: BM25Okapi | None = None
        self._tokenized_corpus: list[list[str]] = []
        self._corpus_vocab: set[str] = set()  # all unique tokens, for prefix expansion

    def rebuild_bm25(self, all_chunks: list[Chunk]) -> None:
        """Rebuild BM25 index and vocabulary from all stored chunks."""
        self._corpus_chunks = all_chunks
        self._tokenized_corpus = [_tokenize(c.text) for c in all_chunks]
        self._bm25 = BM25Okapi(self._tokenized_corpus)
        # Build vocabulary for prefix expansion
        self._corpus_vocab = {tok for tokens in self._tokenized_corpus for tok in tokens}
        logger.info(
            "BM25 index rebuilt with %d chunks, vocab size: %d",
            len(all_chunks), len(self._corpus_vocab),
        )

    def bm25_retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """BM25 retrieval with prefix-aware query expansion."""
        if self._bm25 is None or not self._corpus_chunks:
            return []

        # Use prefix-aware tokenisation if vocab is available
        if self._corpus_vocab:
            tokenized_query = _tokenize_query_with_prefix(query, self._corpus_vocab)
        else:
            tokenized_query = _tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results: list[RetrievedChunk] = []
        max_score = float(scores[top_indices[0]]) if len(top_indices) > 0 else 1.0
        if max_score == 0.0:
            max_score = 1.0

        for idx in top_indices:
            chunk = self._corpus_chunks[idx]
            norm_score = float(scores[idx]) / max_score
            if norm_score > 0.001:   # skip zero-score results
                results.append(RetrievedChunk(
                    document=chunk.document,
                    page=chunk.page,
                    chunk_index=chunk.chunk_index,
                    chunk=chunk.text,
                    relevance_score=round(norm_score, 4),
                ))
        return results

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """
        Cross-encoder re-ranking: jointly scores (query, chunk) pairs.
        Much more accurate than bi-encoder but O(N) forward passes — cap N at 20.
        """
        if not candidates:
            return []

        encoder = _get_cross_encoder()
        pairs = [(query, c.chunk) for c in candidates]
        ce_scores = encoder.predict(pairs)

        # Pair scores with chunks and sort
        scored = sorted(zip(ce_scores, candidates), key=lambda x: x[0], reverse=True)

        # Normalise cross-encoder scores to [0, 1] via sigmoid
        reranked: list[RetrievedChunk] = []
        for ce_score, chunk in scored[:top_k]:
            sigmoid_score = 1.0 / (1.0 + math.exp(-float(ce_score)))
            reranked.append(RetrievedChunk(
                document=chunk.document,
                page=chunk.page,
                chunk_index=chunk.chunk_index,
                chunk=chunk.chunk,
                relevance_score=round(sigmoid_score, 4),
            ))
        return reranked

    def retrieve(
        self,
        query: str,
        query_embedding: np.ndarray,
        vector_store,           # VectorStore — avoid circular import
        top_k_final: int = 5,
    ) -> list[RetrievedChunk]:
        """
        Full hybrid retrieval pipeline:
          BM25(top-20) + Dense(top-20) → RRF → Cross-encoder re-rank(top-5).
        """
        settings = get_settings()

        # 1. BM25 retrieval
        bm25_results = self.bm25_retrieve(query, top_k=settings.bm25_top_k)
        logger.debug("BM25 returned %d candidates", len(bm25_results))

        # 2. Dense retrieval
        dense_results = vector_store.query(
            query_embedding=query_embedding.tolist(),
            top_k=settings.dense_top_k,
        )
        logger.debug("Dense retrieval returned %d candidates", len(dense_results))

        # 3. Reciprocal Rank Fusion
        fused = _reciprocal_rank_fusion(bm25_results, dense_results)
        logger.debug("RRF fusion produced %d candidates", len(fused))

        # 4. Cross-encoder re-ranking
        reranked = self.rerank(query, fused[:20], top_k=settings.rerank_top_k)
        logger.debug("Cross-encoder reranked to %d results", len(reranked))

        return reranked[:top_k_final]
