"""
embeddings.py — singleton embedding model wrapper.

Choice: BAAI/bge-large-en-v1.5 via sentence-transformers.

Rationale:
  • Top-ranked on MTEB retrieval benchmark (beats OpenAI ada-002 on legal/financial text).
  • 1024-dim dense vectors with strong instruction-following via query prefix.
  • Fully local — no API calls, no per-token cost, confidential documents stay on-prem.
  • Supports batch encoding for efficient ingestion.
"""
from __future__ import annotations

import logging
import threading
from functools import lru_cache
from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import get_settings

logger = logging.getLogger(__name__)

# BGE models expect this prefix for queries (not for documents).
_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class EmbeddingModel:
    """Thread-safe singleton wrapper around sentence-transformers."""

    _instance: EmbeddingModel | None = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        settings = get_settings()
        logger.info("Loading embedding model: %s (device=%s)", settings.embedding_model, settings.embedding_device)
        self._model = SentenceTransformer(
            settings.embedding_model,
            device=settings.embedding_device,
        )
        self._dim = self._model.get_sentence_embedding_dimension()
        logger.info("Embedding model loaded — dim=%d", self._dim)

    @classmethod
    def get(cls) -> "EmbeddingModel":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @property
    def dim(self) -> int:
        return self._dim

    def embed_documents(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        """
        Embed a list of document chunks.
        Returns shape (N, dim) float32 array, L2-normalised.
        """
        embeddings = self._model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=len(texts) > 50,
            normalize_embeddings=True,      # cosine sim = dot product after L2-norm
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a user query.
        Uses BGE query prefix for asymmetric retrieval.
        Returns shape (dim,) float32 array, L2-normalised.
        """
        embedding = self._model.encode(
            _QUERY_PREFIX + query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embedding.astype(np.float32)

    def similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Cosine similarity between two L2-normalised vectors."""
        return float(np.dot(vec_a, vec_b))
