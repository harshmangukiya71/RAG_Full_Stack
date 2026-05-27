"""
cache.py - exact and semantic answer cache for RAG responses.

The cache is scoped by corpus fingerprint, so answers are reused only when the
same documents are indexed. It uses TTL expiry plus LRU eviction to keep memory
bounded and lookup fast for development/single-process deployments.
"""
from __future__ import annotations

import logging
import re
import time
from collections import OrderedDict
from dataclasses import dataclass
from hashlib import sha256

import numpy as np

from app.config import get_settings
from app.models import QueryResponse

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    cache_id: str
    corpus_key: str
    normalized_question: str
    top_k: int
    embedding: np.ndarray
    response: QueryResponse
    created_at: float
    last_accessed_at: float
    hits: int = 0


class AnswerCache:
    """Bounded exact + semantic cache with TTL and LRU eviction."""

    def __init__(self) -> None:
        settings = get_settings()
        self._enabled = settings.cache_enabled
        self._ttl_seconds = settings.cache_ttl_seconds
        self._max_entries = max(1, settings.cache_max_entries)
        self._semantic_threshold = settings.semantic_cache_threshold
        self._entries: OrderedDict[str, CacheEntry] = OrderedDict()
        self._exact_index: dict[tuple[str, int, str], str] = {}

    def get_exact(self, question: str, corpus_key: str, top_k: int) -> QueryResponse | None:
        if not self._enabled:
            return None
        self._evict_expired()
        normalized = self._normalize_question(question)
        cache_id = self._exact_index.get((corpus_key, top_k, normalized))
        if not cache_id:
            return None
        entry = self._entries.get(cache_id)
        if not entry:
            return None
        self._touch(entry)
        logger.info("Answer cache exact hit: %s", cache_id)
        return entry.response

    def get_semantic(
        self,
        question_embedding: np.ndarray,
        corpus_key: str,
        top_k: int,
    ) -> QueryResponse | None:
        if not self._enabled or not self._entries:
            return None
        self._evict_expired()

        best_entry: CacheEntry | None = None
        best_score = -1.0
        for entry in self._entries.values():
            if entry.corpus_key != corpus_key or entry.top_k != top_k:
                continue
            score = float(np.dot(question_embedding, entry.embedding))
            if score > best_score:
                best_score = score
                best_entry = entry

        if best_entry and best_score >= self._semantic_threshold:
            self._touch(best_entry)
            logger.info(
                "Answer cache semantic hit: %s similarity=%.4f",
                best_entry.cache_id,
                best_score,
            )
            return best_entry.response
        return None

    def set(
        self,
        question: str,
        question_embedding: np.ndarray,
        corpus_key: str,
        top_k: int,
        response: QueryResponse,
    ) -> None:
        if not self._enabled:
            return
        self._evict_expired()
        normalized = self._normalize_question(question)
        question_hash = sha256(normalized.encode("utf-8")).hexdigest()[:16]
        cache_id = f"{corpus_key}:{top_k}:{question_hash}"
        now = time.time()
        entry = CacheEntry(
            cache_id=cache_id,
            corpus_key=corpus_key,
            normalized_question=normalized,
            top_k=top_k,
            embedding=question_embedding,
            response=response,
            created_at=now,
            last_accessed_at=now,
        )
        self._entries[cache_id] = entry
        self._entries.move_to_end(cache_id)
        self._exact_index[(corpus_key, top_k, normalized)] = cache_id
        self._evict_lru()

    def invalidate_corpus(self, corpus_key: str) -> None:
        """Remove entries for a corpus after document changes."""
        for cache_id, entry in list(self._entries.items()):
            if entry.corpus_key == corpus_key:
                self._delete(cache_id, entry)

    def clear(self) -> None:
        """Clear all cached answers after index-wide document changes."""
        self._entries.clear()
        self._exact_index.clear()

    @staticmethod
    def _normalize_question(question: str) -> str:
        normalized = question.strip().lower()
        normalized = re.sub(r"\s+", " ", normalized)
        normalized = re.sub(r"[^\w\s]", "", normalized)
        return normalized

    def _touch(self, entry: CacheEntry) -> None:
        entry.hits += 1
        entry.last_accessed_at = time.time()
        self._entries.move_to_end(entry.cache_id)

    def _evict_expired(self) -> None:
        if self._ttl_seconds <= 0:
            return
        cutoff = time.time() - self._ttl_seconds
        for cache_id, entry in list(self._entries.items()):
            if entry.created_at < cutoff:
                self._delete(cache_id, entry)

    def _evict_lru(self) -> None:
        while len(self._entries) > self._max_entries:
            cache_id, entry = self._entries.popitem(last=False)
            self._exact_index.pop((entry.corpus_key, entry.top_k, entry.normalized_question), None)
            logger.debug("Evicted LRU answer cache entry: %s", cache_id)

    def _delete(self, cache_id: str, entry: CacheEntry) -> None:
        self._entries.pop(cache_id, None)
        self._exact_index.pop((entry.corpus_key, entry.top_k, entry.normalized_question), None)
