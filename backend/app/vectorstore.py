"""
vectorstore.py — ChromaDB persistent vector store wrapper.

Choice: ChromaDB (persistent, local)

Rationale:
  • Native metadata filtering: filter by document, page, date — critical for legal
    multi-tenant retrieval where you query within a specific contract.
  • Persistent on-disk storage, zero additional infrastructure.
  • Python-native, Docker-friendly.
  • Supports cosine-similarity search on float32 embeddings.

At 50k+ docs, this would be replaced by Qdrant (distributed) or
Elasticsearch with dense-vector plugin.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import get_settings
from app.models import Chunk, RetrievedChunk

logger = logging.getLogger(__name__)


class VectorStore:
    """Thin wrapper around ChromaDB collection with upsert and query support."""

    def __init__(self) -> None:
        settings = get_settings()
        persist_dir = Path(settings.chroma_persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=settings.collection_name,
            metadata={"hnsw:space": "cosine"},   # cosine distance
        )
        logger.info(
            "ChromaDB collection '%s' ready — %d chunks stored",
            settings.collection_name,
            self._collection.count(),
        )

    @property
    def count(self) -> int:
        return self._collection.count()

    def upsert_chunks(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        document_summary: str | None = None,
    ) -> None:
        """
        Upsert chunks with their embeddings.
        Uses chunk_index + document as deterministic ID so re-ingestion is idempotent.
        """
        if not chunks:
            return

        ids = [f"{c.document}::{c.chunk_index}" for c in chunks]
        metadatas: list[dict[str, Any]] = [
            {
                "document": c.document,
                "page": c.page,
                "chunk_index": c.chunk_index,
                "token_count": c.token_count,
                "summary": document_summary or "",
            }
            for c in chunks
        ]
        documents = [c.text for c in chunks]

        # ChromaDB upsert is idempotent
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
        )
        logger.debug("Upserted %d chunks", len(chunks))

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 20,
        where: dict | None = None,
    ) -> list[RetrievedChunk]:
        """
        Dense vector search.
        Returns up to top_k RetrievedChunk objects ordered by cosine similarity (desc).
        """
        count = self._collection.count()
        if count == 0:
            logger.warning("Vector store is empty — no documents ingested yet.")
            return []

        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k, count),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = self._collection.query(**kwargs)

        retrieved: list[RetrievedChunk] = []
        docs = results["documents"][0]       # type: ignore
        metas = results["metadatas"][0]      # type: ignore
        distances = results["distances"][0]  # type: ignore

        for doc_text, meta, dist in zip(docs, metas, distances):
            # ChromaDB cosine distance ∈ [0, 2]; convert to similarity [0, 1]
            similarity = max(0.0, 1.0 - dist / 2.0)
            retrieved.append(RetrievedChunk(
                document=meta["document"],
                page=meta["page"],
                chunk_index=meta["chunk_index"],
                chunk=doc_text,
                relevance_score=round(similarity, 4),
            ))

        return retrieved


    def list_documents(self) -> list[str]:
        """Return unique document names in the collection."""
        if self._collection.count() == 0:
            return []
        result = self._collection.get(include=["metadatas"])
        docs = {m["document"] for m in result["metadatas"]}  # type: ignore
        return sorted(docs)

    def delete_document(self, document_name: str) -> int:
        """Delete all chunks for a specific document. Returns deleted count."""
        result = self._collection.get(
            where={"document": document_name},
            include=["metadatas"],
        )
        ids = result["ids"]
        if ids:
            self._collection.delete(ids=ids)
        return len(ids)

    def get_document_info(self, document_name: str) -> dict:
        """Return metadata summary for a specific document."""
        result = self._collection.get(
            where={"document": document_name},
            include=["metadatas"],
        )
        if not result["metadatas"]:
            return {}
        pages = sorted({m["page"] for m in result["metadatas"]})  # type: ignore
        summaries = [m.get("summary", "") for m in result["metadatas"] if m.get("summary")]  # type: ignore
        return {
            "document": document_name,
            "total_chunks": len(result["ids"]),
            "pages": pages,
            "summary": summaries[0] if summaries else None,
        }
