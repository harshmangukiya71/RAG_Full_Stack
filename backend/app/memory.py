"""
memory.py - session-based conversational memory for RAG answers.

Retrieval still uses only the current question. Memory is injected later into
the LLM prompt so follow-up questions can be understood in conversation.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from threading import Lock
from typing import Literal

from app.config import get_settings

logger = logging.getLogger(__name__)

Role = Literal["user", "assistant"]


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str


class MemoryManager:
    """Redis-backed memory with in-memory fallback for local development."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._max_turns = max(1, self._settings.memory_max_turns)
        self._lock = Lock()
        self._memory: dict[str, list[ChatMessage]] = {}
        self._redis = self._connect_redis()

    def get_history(self, session_id: str | None) -> list[ChatMessage]:
        if not session_id:
            return []
        if self._redis:
            raw = self._redis.get(self._key(session_id))
            if not raw:
                return []
            try:
                payload = json.loads(raw)
                return [ChatMessage(role=item["role"], content=item["content"]) for item in payload]
            except Exception as exc:
                logger.warning("Failed to decode Redis memory for session %s: %s", session_id, exc)
                return []

        with self._lock:
            return list(self._memory.get(session_id, []))

    def add_turn(self, session_id: str | None, question: str, answer: str) -> None:
        if not session_id:
            return
        history = self.get_history(session_id)
        history.extend([
            ChatMessage(role="user", content=question),
            ChatMessage(role="assistant", content=answer),
        ])
        history = history[-self._max_turns * 2:]

        if self._redis:
            payload = json.dumps([asdict(message) for message in history])
            self._redis.setex(self._key(session_id), self._settings.memory_ttl_seconds, payload)
            return

        with self._lock:
            self._memory[session_id] = history

    def format_history(self, session_id: str | None) -> str:
        history = self.get_history(session_id)
        if not history:
            return ""
        lines = []
        for message in history:
            speaker = "User" if message.role == "user" else "Assistant"
            lines.append(f"{speaker}: {message.content}")
        return "\n".join(lines)

    def _connect_redis(self):
        if self._settings.memory_backend == "memory":
            logger.info("Using in-memory chat memory")
            return None
        if not self._settings.redis_url:
            logger.info("Redis URL is not configured; using in-memory chat memory")
            return None
        try:
            import redis

            client = redis.Redis.from_url(
                self._settings.redis_url,
                decode_responses=True,
                socket_connect_timeout=0.5,
                socket_timeout=0.5,
            )
            client.ping()
            logger.info("Using Redis chat memory at %s", self._settings.redis_url)
            return client
        except Exception as exc:
            if self._settings.memory_backend == "redis":
                logger.warning("Redis memory requested but unavailable: %s", exc)
            else:
                logger.info("Redis unavailable; using in-memory chat memory")
            return None

    @staticmethod
    def _key(session_id: str) -> str:
        return f"rag:memory:{session_id}"

    def clear_session(self, session_id: str) -> None:
        """Clear conversational memory for a single session."""
        if self._redis:
            try:
                self._redis.delete(self._key(session_id))
            except Exception as exc:
                logger.warning("Redis clear session failed for %s: %s", session_id, exc)
        with self._lock:
            self._memory.pop(session_id, None)
        logger.info("Cleared memory for session: %s", session_id)

    def clear_all(self) -> None:
        """
        Clear ALL conversational memory across all sessions.

        Only removes chat history — does NOT touch document indexes,
        Pinecone database, graph store, or uploaded documents.
        """
        if self._redis:
            try:
                keys = self._redis.keys("rag:memory:*")
                if keys:
                    self._redis.delete(*keys)
                    logger.info("Cleared %d Redis memory keys", len(keys))
            except Exception as exc:
                logger.warning("Redis clear all memory failed: %s", exc)
        with self._lock:
            session_count = len(self._memory)
            self._memory.clear()
        logger.info("Cleared in-memory chat history (%d sessions)", session_count)

