"""
cache.py — Two-tier answer cache: Redis (L1) + In-Memory (L2 fallback).

Architecture:
─────────────
        Query arrives
             │
             ▼
    ┌─────────────────┐
    │  L1: Redis       │  ← persistent, survives restarts, shared across
    │  (if available)  │    workers, TTL managed by Redis natively
    └────────┬────────┘
             │ miss / Redis down
             ▼
    ┌─────────────────┐
    │  L2: In-Memory   │  ← always available, OrderedDict + LRU eviction,
    │  (always on)     │    numpy matmul for semantic lookup
    └─────────────────┘

Each tier supports:
  • Exact lookup  — normalised question string hash → O(1)
  • Semantic lookup — L2-normalised embedding cosine similarity via matmul

Redis storage format:
  • Exact key:    "rag:exact:{corpus_key}:{top_k}:{question_hash}"
  • Semantic key: "rag:sem:{corpus_key}:{entry_id}"
  • Value:        JSON-serialised QueryResponse + base64 embedding
  • TTL:          cache_ttl_seconds (set per-key on Redis)

Fallback behaviour:
  • Redis unavailable at startup → logs warning, runs in-memory only
  • Redis drops mid-run → caught per-call, falls through to in-memory
  • Redis comes back → next set() call re-connects automatically (redis-py
    connection pool handles reconnection)

Config keys used (already in your config.py):
  cache_enabled, cache_ttl_seconds, cache_max_entries,
  semantic_cache_threshold, redis_url
"""
from __future__ import annotations

import base64
import json
import logging
import pickle
import re
import time
from collections import OrderedDict
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from hashlib import sha256
from typing import Optional

import numpy as np

from app.config import get_settings
from app.models import QueryResponse, SourceReference

logger = logging.getLogger(__name__)

# Redis key prefix — change if you run multiple RAG instances on one Redis
_KEY_PREFIX = "rag"

# Semantic threshold hard cap — never go above 0.85 even if config says so
_MAX_SEMANTIC_THRESHOLD = 0.85


# ── Serialisation helpers ─────────────────────────────────────────────────────

def _response_to_dict(response: QueryResponse) -> dict:
    """Convert QueryResponse to a plain dict for JSON serialisation."""
    return {
        "answer": response.answer,
        "confidence": response.confidence,
        "sources": [
            {
                "document": s.document,
                "page": s.page,
                "chunk": s.chunk,
                "chunk_index": s.chunk_index,
                "entities": s.entities,
            }
            for s in response.sources
        ],
    }


def _dict_to_response(d: dict) -> QueryResponse:
    """Reconstruct QueryResponse from a plain dict."""
    return QueryResponse(
        answer=d["answer"],
        confidence=d["confidence"],
        sources=[
            SourceReference(
                document=s["document"],
                page=s["page"],
                chunk=s["chunk"],
                chunk_index=s.get("chunk_index"),
                entities=s.get("entities", []),
            )
            for s in d.get("sources", [])
        ],
    )


def _embedding_to_b64(v: np.ndarray) -> str:
    """Encode float32 numpy array as base64 string for Redis storage."""
    return base64.b64encode(v.astype(np.float32).tobytes()).decode("ascii")


def _b64_to_embedding(s: str) -> np.ndarray:
    """Decode base64 string back to float32 numpy array."""
    return np.frombuffer(base64.b64decode(s), dtype=np.float32)


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    """L2-normalise a vector. Returns v unchanged if norm is zero."""
    norm = float(np.linalg.norm(v))
    if norm < 1e-10:
        return v.astype(np.float32)
    return (v / norm).astype(np.float32)


def _normalize_question(question: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation."""
    q = question.strip().lower()
    q = re.sub(r"\s+", " ", q)
    q = re.sub(r"[^\w\s]", "", q)
    return q


def _question_hash(normalized: str) -> str:
    return sha256(normalized.encode("utf-8")).hexdigest()[:16]


# ── Redis client (lazy, optional) ─────────────────────────────────────────────

class _RedisClient:
    """
    Thin wrapper around redis-py that:
      • Connects lazily on first use
      • Silently degrades to None if redis package not installed
      • Catches all Redis errors per-call so the app never crashes on Redis issues
    """

    def __init__(self, url: str, ttl: int) -> None:
        self._url = url
        self._ttl = ttl
        self._client = None
        self._available = False
        self._connect()

    def _connect(self) -> None:
        if not self._url:
            logger.info("Redis cache URL is not configured; using in-memory cache only.")
            return
        try:
            import redis  # type: ignore
            client = redis.Redis.from_url(
                self._url,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
                retry_on_timeout=False,
            )
            client.ping()
            self._client = client
            self._available = True
            logger.info("Redis cache connected: %s", self._url)
        except ImportError:
            logger.warning(
                "redis-py not installed — running without Redis cache. "
                "Install with: pip install redis"
            )
        except Exception as exc:
            logger.warning(
                "Redis not available at %s (%s) — falling back to in-memory cache only.",
                self._url, exc,
            )

    @property
    def available(self) -> bool:
        return self._available and self._client is not None

    def get(self, key: str) -> str | None:
        if not self.available:
            return None
        try:
            return self._client.get(key)  # type: ignore
        except Exception as exc:
            logger.debug("Redis GET failed (%s): %s", key, exc)
            self._available = False
            return None

    def set(self, key: str, value: str) -> None:
        if not self.available:
            return
        try:
            self._client.setex(key, self._ttl, value)  # type: ignore
        except Exception as exc:
            logger.debug("Redis SET failed (%s): %s", key, exc)
            self._available = False

    def delete(self, key: str) -> None:
        if not self.available:
            return
        try:
            self._client.delete(key)  # type: ignore
        except Exception as exc:
            logger.debug("Redis DELETE failed (%s): %s", key, exc)

    def keys(self, pattern: str) -> list[str]:
        if not self.available:
            return []
        try:
            return self._client.keys(pattern)  # type: ignore
        except Exception as exc:
            logger.debug("Redis KEYS failed (%s): %s", pattern, exc)
            return []

    def ttl(self, key: str) -> int:
        if not self.available:
            return -1
        try:
            return self._client.ttl(key)  # type: ignore
        except Exception as exc:
            logger.debug("Redis TTL failed (%s): %s", key, exc)
            return -1

    def flushprefix(self, prefix: str) -> int:
        """Delete all keys matching prefix:*"""
        keys = self.keys(f"{prefix}:*")
        if not keys:
            return 0
        try:
            return self._client.delete(*keys)  # type: ignore
        except Exception as exc:
            logger.debug("Redis FLUSHPREFIX failed: %s", exc)
            return 0

    def ping(self) -> bool:
        """Re-check connection — used for auto-reconnect."""
        try:
            if self._client:
                self._client.ping()  # type: ignore
                self._available = True
                return True
        except Exception:
            pass
        return False


# ── In-memory cache entry ─────────────────────────────────────────────────────

@dataclass
class _MemEntry:
    cache_id: str
    corpus_key: str
    normalized_question: str
    top_k: int
    embedding: np.ndarray          # L2-normalised float32
    response: QueryResponse
    created_at: float
    last_accessed_at: float
    hits: int = 0


# ── Main AnswerCache ──────────────────────────────────────────────────────────

class AnswerCache:
    """
    Two-tier answer cache:
      L1 = Redis  (persistent, shared, survives restarts)
      L2 = In-Memory (always available, numpy matmul semantic search)

    Public API is identical to the previous single-tier cache so pipeline.py
    needs zero changes.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._enabled = settings.cache_enabled
        self._ttl = settings.cache_ttl_seconds
        self._max_entries = max(1, settings.cache_max_entries)
        self._threshold = min(settings.semantic_cache_threshold, _MAX_SEMANTIC_THRESHOLD)

        # L1: Redis
        self._redis = _RedisClient(settings.redis_url, self._ttl)

        # L2: In-memory
        self._entries: OrderedDict[str, _MemEntry] = OrderedDict()
        self._exact_index: dict[tuple[str, int, str], str] = {}
        self._matrix: Optional[np.ndarray] = None
        self._matrix_ids: list[str] = []
        self._matrix_dirty = False

        # Stats
        self._redis_exact_hits = 0
        self._redis_semantic_hits = 0
        self._mem_exact_hits = 0
        self._mem_semantic_hits = 0
        self._misses = 0
        self._sets = 0
        self._evictions = 0

        # Local disk fallback file
        self._cache_file = Path("data/local_cache.pkl")
        self._cache_file.parent.mkdir(parents=True, exist_ok=True)
        self._load_from_disk()

        logger.info(
            "AnswerCache ready — Redis: %s | In-memory max: %d | threshold: %.2f",
            "✅ connected" if self._redis.available else "❌ offline (in-memory only)",
            self._max_entries,
            self._threshold,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def get_exact(
        self, question: str, corpus_key: str, top_k: int
    ) -> QueryResponse | None:
        """
        Exact cache lookup. Called BEFORE embed_query() to save inference cost.
        Checks Redis first, then in-memory.
        """
        if not self._enabled:
            return None
        self._evict_expired()

        normalized = _normalize_question(question)
        qhash = _question_hash(normalized)

        # ── L1: Redis exact ───────────────────────────────────────────────────
        redis_key = f"{_KEY_PREFIX}:exact:{corpus_key}:{top_k}:{qhash}"
        raw = self._redis.get(redis_key)
        if raw:
            try:
                data = json.loads(raw)
                response = _dict_to_response(data["response"])
                self._redis_exact_hits += 1
                logger.info(
                    "Cache L1-EXACT hit key=%s redis_exact=%d",
                    redis_key, self._redis_exact_hits,
                )
                # Warm L2 so next hit is even faster
                if data.get("embedding"):
                    emb = _b64_to_embedding(data["embedding"])
                    self._mem_set(normalized, qhash, corpus_key, top_k, emb, response)
                return response
            except Exception as exc:
                logger.debug("Redis exact deserialise failed: %s", exc)

        # ── L2: In-memory exact ───────────────────────────────────────────────
        cache_id = self._exact_index.get((corpus_key, top_k, normalized))
        if cache_id:
            entry = self._entries.get(cache_id)
            if entry:
                self._touch(entry)
                self._mem_exact_hits += 1
                logger.info(
                    "Cache L2-EXACT hit id=%s mem_exact=%d",
                    cache_id, self._mem_exact_hits,
                )
                return entry.response

        self._misses += 1
        return None

    def get_semantic(
        self,
        question_embedding: np.ndarray,
        corpus_key: str,
        top_k: int,
    ) -> QueryResponse | None:
        """
        Semantic cache lookup using cosine similarity.
        Checks Redis index first, then in-memory numpy matmul.
        """
        if not self._enabled or not self._entries and not self._redis.available:
            return None
        self._evict_expired()

        q_norm = _l2_normalize(question_embedding)

        # ── L1: Redis semantic ────────────────────────────────────────────────
        # Redis stores all semantic entries under rag:sem:{corpus_key}:*
        # We fetch all embeddings for this corpus_key and do matmul locally
        # (Redis doesn't do vector math natively without RediSearch)
        sem_keys = self._redis.keys(f"{_KEY_PREFIX}:sem:{corpus_key}:*")
        if sem_keys:
            best_score = -1.0
            best_raw = None
            best_key = None
            candidate_embeddings = []
            candidate_keys = []
            candidate_raws = []

            for k in sem_keys:
                raw = self._redis.get(k)
                if not raw:
                    continue
                try:
                    data = json.loads(raw)
                    emb = _b64_to_embedding(data["embedding"])
                    if emb.shape != q_norm.shape:
                        continue
                    candidate_embeddings.append(emb)
                    candidate_keys.append(k)
                    candidate_raws.append(data)
                except Exception:
                    continue

            if candidate_embeddings:
                matrix = np.stack(candidate_embeddings, axis=0)  # (M, D)
                sims = matrix @ q_norm                            # (M,) vectorized
                best_idx = int(np.argmax(sims))
                best_score = float(sims[best_idx])

                if best_score >= self._threshold:
                    data = candidate_raws[best_idx]
                    try:
                        response = _dict_to_response(data["response"])
                        self._redis_semantic_hits += 1
                        logger.info(
                            "Cache L1-SEMANTIC hit sim=%.4f key=%s redis_sem=%d",
                            best_score, candidate_keys[best_idx],
                            self._redis_semantic_hits,
                        )
                        # Warm L2
                        normalized = data.get("normalized_question", "")
                        qhash = _question_hash(normalized) if normalized else "warm"
                        self._mem_set(
                            normalized, qhash, corpus_key, top_k,
                            candidate_embeddings[best_idx], response,
                        )
                        return response
                    except Exception as exc:
                        logger.debug("Redis semantic deserialise failed: %s", exc)

        # ── L2: In-memory semantic (numpy matmul) ─────────────────────────────
        if not self._entries:
            self._misses += 1
            return None

        if self._matrix_dirty or self._matrix is None:
            self._rebuild_matrix(expected_dim=q_norm.shape[0])

        if self._matrix is None or not self._matrix_ids:
            self._misses += 1
            return None

        valid = [
            i for i, cid in enumerate(self._matrix_ids)
            if cid in self._entries
            and self._entries[cid].corpus_key == corpus_key
            and self._entries[cid].embedding.shape == q_norm.shape
        ]
        if not valid:
            self._misses += 1
            return None

        idx_arr = np.array(valid, dtype=np.int32)
        sub = self._matrix[idx_arr]        # (M, D)
        sims = sub @ q_norm                # (M,) vectorized
        best_local = int(np.argmax(sims))
        best_score = float(sims[best_local])
        best_id = self._matrix_ids[valid[best_local]]

        if best_score >= self._threshold and best_id in self._entries:
            entry = self._entries[best_id]
            self._touch(entry)
            self._mem_semantic_hits += 1
            logger.info(
                "Cache L2-SEMANTIC hit sim=%.4f id=%s mem_sem=%d",
                best_score, best_id, self._mem_semantic_hits,
            )
            return entry.response

        self._misses += 1
        logger.debug("Cache MISS best_sim=%.4f threshold=%.2f", best_score, self._threshold)
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

        normalized = _normalize_question(question)
        qhash = _question_hash(normalized)
        emb_norm = _l2_normalize(question_embedding)

        # ── L1: Write to Redis ────────────────────────────────────────────────
        payload = json.dumps({
            "response": _response_to_dict(response),
            "embedding": _embedding_to_b64(emb_norm),
            "normalized_question": normalized,
            "corpus_key": corpus_key,
            "top_k": top_k,
            "created_at": time.time(),
        })
        # Exact key
        exact_key = f"{_KEY_PREFIX}:exact:{corpus_key}:{top_k}:{qhash}"
        self._redis.set(exact_key, payload)
        # Semantic key (separate namespace for bulk key scan)
        sem_key = f"{_KEY_PREFIX}:sem:{corpus_key}:{qhash}"
        self._redis.set(sem_key, payload)

        # ── L2: Write to in-memory ────────────────────────────────────────────
        self._mem_set(normalized, qhash, corpus_key, top_k, emb_norm, response)

        self._sets += 1
        logger.debug(
            "Cache SET normalized=%r corpus=%s redis=%s mem_size=%d",
            normalized[:40], corpus_key,
            "✅" if self._redis.available else "❌",
            len(self._entries),
        )

    def invalidate_corpus(self, corpus_key: str) -> None:
        """Remove all entries for a corpus (called after document changes)."""
        # L1
        deleted = self._redis.flushprefix(f"{_KEY_PREFIX}:exact:{corpus_key}")
        deleted += self._redis.flushprefix(f"{_KEY_PREFIX}:sem:{corpus_key}")
        if deleted:
            logger.info("Redis: deleted %d keys for corpus %s", deleted, corpus_key)
        # L2
        for cache_id, entry in list(self._entries.items()):
            if entry.corpus_key == corpus_key:
                self._mem_delete(cache_id, entry)

    def clear(self) -> None:
        """Clear all cached answers (called after index-wide document changes)."""
        deleted = self._redis.flushprefix(_KEY_PREFIX)
        logger.info(
            "Cache CLEARED — Redis keys deleted: %d | "
            "Stats: redis_exact=%d redis_sem=%d mem_exact=%d mem_sem=%d "
            "misses=%d sets=%d evictions=%d",
            deleted,
            self._redis_exact_hits, self._redis_semantic_hits,
            self._mem_exact_hits, self._mem_semantic_hits,
            self._misses, self._sets, self._evictions,
        )
        self._entries.clear()
        self._exact_index.clear()
        self._matrix = None
        self._matrix_ids = []
        self._matrix_dirty = False
        self._save_to_disk()

    def stats(self) -> dict:
        total = (
            self._redis_exact_hits + self._redis_semantic_hits
            + self._mem_exact_hits + self._mem_semantic_hits
            + self._misses
        )
        hits = (
            self._redis_exact_hits + self._redis_semantic_hits
            + self._mem_exact_hits + self._mem_semantic_hits
        )
        return {
            "redis_available": self._redis.available,
            "mem_size": len(self._entries),
            "redis_exact_hits": self._redis_exact_hits,
            "redis_semantic_hits": self._redis_semantic_hits,
            "mem_exact_hits": self._mem_exact_hits,
            "mem_semantic_hits": self._mem_semantic_hits,
            "misses": self._misses,
            "sets": self._sets,
            "evictions": self._evictions,
            "hit_rate": round(hits / total, 4) if total else 0.0,
        }

    # ── Private: in-memory helpers ────────────────────────────────────────────

    def _mem_set(
        self,
        normalized: str,
        qhash: str,
        corpus_key: str,
        top_k: int,
        emb_norm: np.ndarray,
        response: QueryResponse,
    ) -> None:
        cache_id = f"{corpus_key}:{top_k}:{qhash}"
        now = time.time()
        entry = _MemEntry(
            cache_id=cache_id,
            corpus_key=corpus_key,
            normalized_question=normalized,
            top_k=top_k,
            embedding=emb_norm,
            response=response,
            created_at=now,
            last_accessed_at=now,
        )
        self._entries[cache_id] = entry
        self._entries.move_to_end(cache_id)
        self._exact_index[(corpus_key, top_k, normalized)] = cache_id
        self._matrix_dirty = True
        self._evict_lru()
        self._save_to_disk()

    def _rebuild_matrix(self, expected_dim: int | None = None) -> None:
        if not self._entries:
            self._matrix = None
            self._matrix_ids = []
            self._matrix_dirty = False
            return
        ids = [
            cache_id
            for cache_id, entry in self._entries.items()
            if expected_dim is None or entry.embedding.shape == (expected_dim,)
        ]
        if not ids:
            self._matrix = None
            self._matrix_ids = []
            self._matrix_dirty = False
            return
        arrays = [self._entries[cid].embedding for cid in ids]
        self._matrix = np.stack(arrays, axis=0)
        self._matrix_ids = ids
        self._matrix_dirty = False
        logger.debug("L2 matrix rebuilt shape=%s", self._matrix.shape)

    def _touch(self, entry: _MemEntry) -> None:
        entry.hits += 1
        entry.last_accessed_at = time.time()
        self._entries.move_to_end(entry.cache_id)

    def _evict_expired(self) -> None:
        if self._ttl <= 0:
            return
        cutoff = time.time() - self._ttl
        for cache_id, entry in list(self._entries.items()):
            if entry.created_at < cutoff:
                self._mem_delete(cache_id, entry)
                self._evictions += 1

    def _evict_lru(self) -> None:
        while len(self._entries) > self._max_entries:
            cache_id, entry = self._entries.popitem(last=False)
            self._exact_index.pop(
                (entry.corpus_key, entry.top_k, entry.normalized_question), None
            )
            self._matrix_dirty = True
            self._evictions += 1
            logger.debug("L2 LRU evict id=%s", cache_id)

    def _mem_delete(self, cache_id: str, entry: _MemEntry) -> None:
        self._entries.pop(cache_id, None)
        self._exact_index.pop(
            (entry.corpus_key, entry.top_k, entry.normalized_question), None
        )
        self._matrix_dirty = True
        self._save_to_disk()

    # ── Disk persistence helpers ──────────────────────────────────────────────

    def _load_from_disk(self) -> None:
        if not self._cache_file.exists():
            return
        try:
            with open(self._cache_file, "rb") as f:
                saved_entries = pickle.load(f)
            
            now = time.time()
            for cache_id, entry in saved_entries.items():
                # Check TTL on load
                if int(entry.created_at + self._ttl - now) > 0:
                    self._entries[cache_id] = entry
                    self._exact_index[(entry.corpus_key, entry.top_k, entry.normalized_question)] = cache_id
            
            if self._entries:
                self._matrix_dirty = True
                logger.info("Loaded %d active cache entries from local disk fallback.", len(self._entries))
        except Exception as exc:
            logger.warning("Failed to load local cache from disk: %s", exc)

    def _save_to_disk(self) -> None:
        try:
            temp_file = self._cache_file.with_suffix('.tmp')
            with open(temp_file, "wb") as f:
                pickle.dump(self._entries, f)
            temp_file.replace(self._cache_file)
        except Exception as exc:
            logger.debug("Failed to save local cache to disk: %s", exc)

    # ── Cache load progress (Feature 6) ───────────────────────────────────────

    def get_load_status(self) -> dict:
        """
        Report cache loading progress after server restart.

        Returns percentage of Redis entries that have been warmed into
        the in-memory L2 layer.
        """
        redis_count = 0
        if self._redis.available:
            redis_count = len(self._redis.keys(f"{_KEY_PREFIX}:*"))

        mem_count = len(self._entries)

        if redis_count == 0 and mem_count == 0:
            percent = 100  # nothing to load → fully ready
        elif redis_count == 0:
            percent = 100
        else:
            percent = min(100, int((mem_count / max(1, redis_count)) * 100))

        return {
            "cache_loaded_percent": percent,
            "is_ready": percent >= get_settings().cache_ready_percent,
            "redis_available": self._redis.available,
            "total_entries": redis_count,
            "loaded_entries": mem_count,
        }

    def get_all_entries(self) -> list[dict]:
        """Return a list of all currently loaded cache entries with TTL."""
        import time
        results = []
        now = time.time()
        
        # We fetch from L2 memory so we have the parsed QueryResponse 
        for entry in list(self._entries.values()):
            # Calculate TTL purely based on creation time to avoid network overhead per item
            ttl = int(entry.created_at + self._ttl - now)
            
            if ttl <= 0:
                continue
                
            results.append({
                "question": entry.normalized_question,
                "answer_preview": entry.response.answer[:150] + "..." if len(entry.response.answer) > 150 else entry.response.answer,
                "ttl_seconds": ttl,
                "hits": entry.hits,
            })
            
        # Sort by most recently accessed
        results.sort(key=lambda x: x["ttl_seconds"], reverse=True)
        return results

    def reset_stats(self) -> None:
        """Reset all cache statistics counters."""
        self._redis_exact_hits = 0
        self._redis_semantic_hits = 0
        self._mem_exact_hits = 0
        self._mem_semantic_hits = 0
        self._misses = 0
        self._sets = 0
        self._evictions = 0
