"""
embeddings.py - singleton NVIDIA embedding API wrapper.

The public interface intentionally remains unchanged:
  - embed_documents(texts)
  - embed_query(query)

Changing embedding providers changes vector dimensionality for most models.
Existing ChromaDB/Neo4j vector data must be re-embedded and reindexed before
serving queries with the new NVIDIA embedding model.
"""
from __future__ import annotations

import logging
import threading
from typing import Sequence

import numpy as np
from openai import OpenAI

from app.config import get_settings

logger = logging.getLogger(__name__)


class EmbeddingModel:
    _instance: "EmbeddingModel | None" = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        settings = get_settings()
        self._client = OpenAI(
            api_key=settings.nvidia_api_key,
            base_url=settings.nvidia_base_url,
        )
        self._model = settings.nvidia_embedding_model
        self._batch_size = max(1, settings.nvidia_embedding_batch_size)
        self._dim: int | None = None
        logger.info("Using NVIDIA embedding model: %s", self._model)

    @classmethod
    def get(cls) -> "EmbeddingModel":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @property
    def dimension(self) -> int | None:
        return self._dim

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start:start + self._batch_size])
            vectors.extend(self._embed(batch, input_type="passage"))
        return self._as_float32(vectors)

    def embed_query(self, query: str) -> np.ndarray:
        vectors = self._embed([query], input_type="query")
        return self._as_float32(vectors)[0]

    def similarity(self, vec_a, vec_b) -> float:
        return float(np.dot(vec_a, vec_b))

    def _embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        response = self._client.embeddings.create(
            model=self._model,
            input=texts,
            encoding_format="float",
            extra_body={
                "input_type": input_type,
                "truncate": "NONE",
            },
        )
        return [item.embedding for item in response.data]

    def _as_float32(self, vectors: list[list[float]]) -> np.ndarray:
        array = np.asarray(vectors, dtype=np.float32)
        if array.ndim != 2:
            raise ValueError("NVIDIA embedding API returned an invalid vector shape")
        if self._dim is None:
            self._dim = int(array.shape[1])
            logger.info("NVIDIA embedding dimension detected: %d", self._dim)
        elif array.shape[1] != self._dim:
            raise ValueError(
                f"Embedding dimension changed from {self._dim} to {array.shape[1]}. "
                "Re-embed and reindex the document store before serving traffic."
            )
        return array
