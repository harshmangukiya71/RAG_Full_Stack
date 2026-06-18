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
import json
import re
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import get_settings
from app.models import Chunk, RetrievedChunk

logger = logging.getLogger(__name__)

_BAD_SUMMARY_MARKERS = (
    "partial summary 1",
    "please provide",
    "provided \"partial summary",
    "provided partial summary",
    "is incomplete",
    "i apologize",
)


def _is_bad_summary(summary: str | None) -> bool:
    text = (summary or "").strip().lower()
    if not text:
        return True
    return any(marker in text for marker in _BAD_SUMMARY_MARKERS)


def _extractive_summary(documents: list[str]) -> str | None:
    text = re.sub(r"\s+", " ", " ".join(documents)).strip()
    if not text:
        return None
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if len(sentence.strip()) > 20
    ]
    if not sentences:
        return text[:1200]
    overview = " ".join(sentences[:2]).strip()
    bullets = "\n".join(f"- {sentence}" for sentence in (sentences[2:8] or sentences[:6]))
    return f"{overview}\n\nKey points:\n{bullets}".strip()


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

        Embedding model migrations require a full re-embed/reindex. ChromaDB
        fixes collection dimensionality from stored vectors, so NVIDIA vectors
        cannot be mixed with older local embedding vectors.
        """
        if not chunks:
            return
        self._validate_embedding_dimensions(embeddings)

        ids = [f"{c.document}::{c.chunk_index}" for c in chunks]
        metadatas: list[dict[str, Any]] = [
            {
                "document": c.document,
                "page": c.page,
                "chunk_index": c.chunk_index,
                "token_count": c.token_count,
                "summary": document_summary or "",
                "section_title": c.section_title or "",
                "entities": json.dumps(c.entities),
                "ocr_confidence": c.ocr_confidence if c.ocr_confidence is not None else 1.0,
                "extraction_method": c.extraction_method,
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

        self._validate_embedding_dimensions([query_embedding])

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
                entity_matches=json.loads(meta.get("entities", "[]") or "[]"),
            ))

        return retrieved

    def _validate_embedding_dimensions(self, embeddings: list[list[float]]) -> None:
        if not embeddings:
            return
        incoming_dim = len(embeddings[0])
        if incoming_dim == 0 or any(len(embedding) != incoming_dim for embedding in embeddings):
            raise ValueError("All embeddings must be non-empty and have the same dimension")
        if self._collection.count() == 0:
            return

        sample = self._collection.get(limit=1, include=["embeddings"])
        stored_embeddings = sample.get("embeddings")
        if stored_embeddings is None or len(stored_embeddings) == 0:
            return
        stored_dim = len(stored_embeddings[0])
        if stored_dim != incoming_dim:
            raise ValueError(
                f"Embedding dimension mismatch: stored={stored_dim}, incoming={incoming_dim}. "
                "Re-embed and reindex documents after changing the embedding model."
            )

    def get_chunks_by_keys(self, keys: list[tuple[str, int, int]], score: float = 0.7) -> list[RetrievedChunk]:
        """Fetch chunks by (document, page, chunk_index), used by graph retrieval."""
        if not keys:
            return []
        docs = sorted({document for document, _, _ in keys})
        wanted_entities: dict[tuple[str, int, int], list[str]] = {}
        for key in keys:
            wanted_entities[(key[0], key[1], key[2])] = []

        retrieved: list[RetrievedChunk] = []
        for document in docs:
            result = self._collection.get(where={"document": document}, include=["documents", "metadatas"])
            for doc_text, meta in zip(result["documents"], result["metadatas"]):  # type: ignore
                key = (meta["document"], int(meta["page"]), int(meta["chunk_index"]))
                if key not in wanted_entities:
                    continue
                try:
                    entity_matches = json.loads(meta.get("entities", "[]") or "[]")
                except Exception:
                    entity_matches = []
                retrieved.append(RetrievedChunk(
                    document=key[0],
                    page=key[1],
                    chunk_index=key[2],
                    chunk=doc_text,
                    relevance_score=score,
                    retrieval_source="graph",
                    entity_matches=entity_matches,
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
            include=["documents", "metadatas"],
        )
        if not result["metadatas"]:
            return {}
        pages = sorted({m["page"] for m in result["metadatas"]})  # type: ignore
        summaries = [m.get("summary", "") for m in result["metadatas"] if m.get("summary")]  # type: ignore
        summary = summaries[0] if summaries else None
        if _is_bad_summary(summary):
            summary = _extractive_summary(result.get("documents", []) or [])  # type: ignore
        confidences = [float(m.get("ocr_confidence", 1.0)) for m in result["metadatas"]]  # type: ignore
        entities: set[str] = set()
        for meta in result["metadatas"]:  # type: ignore
            try:
                entities.update(json.loads(meta.get("entities", "[]") or "[]"))
            except Exception:
                pass
        return {
            "document": document_name,
            "total_chunks": len(result["ids"]),
            "pages": pages,
            "summary": summary,
            "average_ocr_confidence": round(sum(confidences) / len(confidences), 3) if confidences else None,
            "entities": sorted(entities),
        }
