import os
import re

file_path = r"C:\Users\os\Desktop\RAG\backend\app\vectorstore.py"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

new_content = """\"\"\"Vector store backends for Qdrant Cloud and local ChromaDB.\"\"\"
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any
import uuid

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, MatchAny
except ImportError:
    QdrantClient = None

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
except ImportError:
    chromadb = None

from app.config import get_settings
from app.models import Chunk, RetrievedChunk

logger = logging.getLogger(__name__)

_BAD_SUMMARY_MARKERS = (
    "partial summary 1",
    "please provide",
    "provided \\"partial summary",
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
    text = re.sub(r"\\s+", " ", " ".join(documents)).strip()
    if not text:
        return None
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\\s+", text)
        if len(sentence.strip()) > 20
    ]
    if not sentences:
        return text[:1200]
    overview = " ".join(sentences[:2]).strip()
    bullets = "\\n".join(f"- {sentence}" for sentence in (sentences[2:8] or sentences[:6]))
    return f"{overview}\\n\\nKey points:\\n{bullets}".strip()


class VectorStore:
    \"\"\"Thin wrapper around the configured vector database.\"\"\"

    def __init__(self) -> None:
        settings = get_settings()
        backend = settings.vector_db.strip().lower()
        if backend == "qdrant":
            self._init_qdrant(settings)
            return
        if backend not in {"chroma", "chromadb"}:
            raise ValueError("VECTOR_DB must be either 'chroma' or 'qdrant'")
        self._init_chroma(settings)

    def _init_chroma(self, settings: Any) -> None:
        self._backend = "chroma"
        if chromadb is None:
            raise ImportError("ChromaDB support requires the 'chromadb' package.")
        persist_dir = Path(settings.chroma_persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=settings.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaDB collection '%s' ready - %d chunks stored",
            settings.collection_name,
            self._collection.count(),
        )

    def _init_qdrant(self, settings: Any) -> None:
        self._backend = "qdrant"
        if QdrantClient is None:
            raise ImportError("Qdrant support requires the 'qdrant-client' package. Run: pip install qdrant-client")
        if not settings.qdrant_url.strip():
            raise ValueError("QDRANT_URL must be set when VECTOR_DB=qdrant")
        if not settings.qdrant_api_key.strip():
            raise ValueError("QDRANT_API_KEY must be set when VECTOR_DB=qdrant")
        if not settings.qdrant_collection.strip():
            raise ValueError("QDRANT_COLLECTION must be set when VECTOR_DB=qdrant")

        self._collection_name = settings.qdrant_collection.strip()
        self._client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
        )
        
        # Verify collection exists, if not create it
        if not self._client.collection_exists(self._collection_name):
            logger.info("Creating Qdrant collection: %s", self._collection_name)
            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(size=4096, distance=Distance.COSINE),
                sparse_vectors_config=self._client.get_fastembed_sparse_vector_params() if hasattr(self._client, 'get_fastembed_sparse_vector_params') else None
            )
        
        logger.info(
            "Using vector database: Qdrant\\n"
            "Collection: %s\\n"
            "Embedding dimension: 4096\\n"
            "Qdrant connection successful",
            self._collection_name
        )

    @property
    def count(self) -> int:
        if self._backend == "qdrant":
            collection_info = self._client.get_collection(self._collection_name)
            return collection_info.points_count
        return self._collection.count()

    def upsert_chunks(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        document_summary: str | None = None,
    ) -> None:
        \"\"\"Upsert chunks with their embeddings.\"\"\"
        if not chunks:
            return
        self._validate_embedding_dimensions(embeddings)

        if self._backend == "qdrant":
            points = []
            for c, emb in zip(chunks, embeddings):
                id_str = f"{c.document}::{c.chunk_index}"
                # Qdrant requires UUIDs or unsigned integers for IDs. Using UUID derived from string.
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, id_str))
                payload = {
                    "document": c.document,
                    "page": c.page,
                    "chunk_index": c.chunk_index,
                    "token_count": c.token_count,
                    "summary": document_summary or "",
                    "section_title": c.section_title or "",
                    "entities": json.dumps(c.entities),
                    "ocr_confidence": c.ocr_confidence if c.ocr_confidence is not None else 1.0,
                    "extraction_method": c.extraction_method,
                    "text": c.text,
                    "chunk_id_str": id_str  # Keep original ID as string for exact matches
                }
                points.append(PointStruct(id=point_id, vector=emb, payload=payload))

            for start in range(0, len(points), 100):
                self._client.upsert(
                    collection_name=self._collection_name,
                    points=points[start:start + 100]
                )
        else:
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
        \"\"\"Dense vector search ordered by cosine similarity descending.\"\"\"
        count = self.count
        if count == 0:
            logger.warning("Vector store is empty - no documents ingested yet.")
            return []

        self._validate_embedding_dimensions([query_embedding])

        if self._backend == "qdrant":
            search_result = self._client.search(
                collection_name=self._collection_name,
                query_vector=query_embedding,
                limit=min(top_k, count),
                query_filter=_qdrant_filter(where),
                with_payload=True
            )
            
            retrieved: list[RetrievedChunk] = []
            for hit in search_result:
                meta = hit.payload or {}
                # Qdrant cosine similarity returns values where higher is more similar
                similarity = hit.score 
                retrieved.append(_retrieved_from_metadata(meta, similarity))
            return retrieved

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
            similarity = max(0.0, 1.0 - dist / 2.0)
            retrieved.append(
                RetrievedChunk(
                    document=meta["document"],
                    page=meta["page"],
                    chunk_index=meta["chunk_index"],
                    chunk=doc_text,
                    relevance_score=round(similarity, 4),
                    entity_matches=json.loads(meta.get("entities", "[]") or "[]"),
                )
            )

        return retrieved

    def _validate_embedding_dimensions(self, embeddings: list[list[float]]) -> None:
        if not embeddings:
            return
        incoming_dim = len(embeddings[0])
        if incoming_dim == 0 or any(len(embedding) != incoming_dim for embedding in embeddings):
            raise ValueError("All embeddings must be non-empty and have the same dimension")
        if self.count == 0:
            return

        if self._backend == "qdrant":
            # Assume 4096 based on requirements, but check configuration if possible
            collection_info = self._client.get_collection(self._collection_name)
            stored_dim = collection_info.config.params.vectors.size if collection_info.config.params.vectors else None
            if stored_dim is None:
                return
            if stored_dim != incoming_dim:
                raise ValueError(
                    f"Embedding dimension mismatch: stored={stored_dim}, incoming={incoming_dim}. "
                    "Re-embed and reindex documents after changing the embedding model."
                )
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
        \"\"\"Fetch chunks by (document, page, chunk_index), used by graph retrieval.\"\"\"
        if not keys:
            return []

        if self._backend == "qdrant":
            wanted = {(document, int(page), int(chunk_index)) for document, page, chunk_index in keys}
            
            # Since Qdrant uses UUIDs, it's easier to search by filter
            docs = list(set(k[0] for k in keys))
            
            conditions = []
            for doc in docs:
                conditions.append(FieldCondition(key="document", match=MatchValue(value=doc)))
            
            records, _ = self._client.scroll(
                collection_name=self._collection_name,
                scroll_filter=Filter(should=conditions),
                limit=10000,
                with_payload=True
            )
            
            retrieved: list[RetrievedChunk] = []
            for record in records:
                meta = record.payload or {}
                key = (
                    str(meta.get("document", "")),
                    int(meta.get("page", 0)),
                    int(meta.get("chunk_index", 0)),
                )
                if key in wanted:
                    retrieved.append(_retrieved_from_metadata(meta, score, retrieval_source="graph"))
            return retrieved

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
                retrieved.append(
                    RetrievedChunk(
                        document=key[0],
                        page=key[1],
                        chunk_index=key[2],
                        chunk=doc_text,
                        relevance_score=score,
                        retrieval_source="graph",
                        entity_matches=entity_matches,
                    )
                )
        return retrieved

    def list_documents(self) -> list[str]:
        \"\"\"Return unique document names in the collection.\"\"\"
        if self.count == 0:
            return []
        if self._backend == "qdrant":
            return sorted({chunk.document for chunk in self.list_all_chunks() if chunk.document})

        result = self._collection.get(include=["metadatas"])
        docs = {m["document"] for m in result["metadatas"]}  # type: ignore
        return sorted(docs)

    def delete_document(self, document_name: str) -> int:
        \"\"\"Delete all chunks for a specific document. Returns deleted count.\"\"\"
        if self._backend == "qdrant":
            # First count how many points we will delete
            count_result = self._client.count(
                collection_name=self._collection_name,
                count_filter=Filter(
                    must=[FieldCondition(key="document", match=MatchValue(value=document_name))]
                )
            )
            count = count_result.count
            
            # Then delete them
            self._client.delete(
                collection_name=self._collection_name,
                points_selector=Filter(
                    must=[FieldCondition(key="document", match=MatchValue(value=document_name))]
                )
            )
            return count

        result = self._collection.get(
            where={"document": document_name},
            include=["metadatas"],
        )
        ids = result["ids"]
        if ids:
            self._collection.delete(ids=ids)
        return len(ids)

    def get_document_info(self, document_name: str) -> dict:
        \"\"\"Return metadata summary for a specific document.\"\"\"
        if self._backend == "qdrant":
            records, _ = self._client.scroll(
                collection_name=self._collection_name,
                scroll_filter=Filter(
                    must=[FieldCondition(key="document", match=MatchValue(value=document_name))]
                ),
                limit=10000,
                with_payload=True
            )
            if not records:
                return {}
            
            chunks = [_chunk_from_metadata(r.payload or {}) for r in records]
            pages = sorted({chunk.page for chunk in chunks})
            summaries = [str((r.payload or {}).get("summary", "")) for r in records if (r.payload or {}).get("summary")]
            summary = summaries[0] if summaries else None
            if _is_bad_summary(summary):
                summary = _extractive_summary([chunk.text for chunk in chunks])
            confidences = [
                float(chunk.ocr_confidence if chunk.ocr_confidence is not None else 1.0)
                for chunk in chunks
            ]
            entities: set[str] = set()
            for chunk in chunks:
                entities.update(chunk.entities)
            return {
                "document": document_name,
                "total_chunks": len(records),
                "pages": pages,
                "summary": summary,
                "average_ocr_confidence": round(sum(confidences) / len(confidences), 3) if confidences else None,
                "entities": sorted(entities),
            }

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

    def list_all_chunks(self) -> list[Chunk]:
        \"\"\"Load all chunks from the active vector store.\"\"\"
        if self.count == 0:
            return []

        if self._backend == "qdrant":
            chunks: list[Chunk] = []
            offset = None
            while True:
                records, offset = self._client.scroll(
                    collection_name=self._collection_name,
                    limit=1000,
                    offset=offset,
                    with_payload=True
                )
                for r in records:
                    chunks.append(_chunk_from_metadata(r.payload or {}))
                if offset is None:
                    break
            chunks.sort(key=lambda chunk: (chunk.document, chunk.page, chunk.chunk_index))
            return chunks

        result = self._collection.get(include=["documents", "metadatas"])
        chunks: list[Chunk] = []
        for text, meta in zip(result["documents"], result["metadatas"]):  # type: ignore
            chunks.append(_chunk_from_metadata(meta, text=text))
        return chunks


def _qdrant_filter(where: dict | None) -> Filter | None:
    if not where:
        return None
    conditions = []
    for key, value in where.items():
        if isinstance(value, dict):
            # Basic handling of simple ops if needed, falling back to MatchValue
            # e.g., if there's a specific requirement for $in or $eq
            pass
        else:
            conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
    return Filter(must=conditions)


def _chunk_from_metadata(meta: dict[str, Any], text: str | None = None) -> Chunk:
    chunk_text = text if text is not None else str(meta.get("text", ""))
    entities = []
    try:
        entities = json.loads(meta.get("entities", "[]") or "[]")
    except Exception:
        pass
    return Chunk(
        document=str(meta.get("document", "")),
        page=int(meta.get("page", 0)),
        chunk_index=int(meta.get("chunk_index", 0)),
        text=chunk_text,
        token_count=int(meta.get("token_count", 0) or len(chunk_text) // 4),
        section_title=meta.get("section_title") or None,
        entities=entities,
        ocr_confidence=meta.get("ocr_confidence"),
        extraction_method=str(meta.get("extraction_method", "native")),
    )


def _retrieved_from_metadata(
    meta: dict[str, Any],
    score: float,
    retrieval_source: str = "dense",
) -> RetrievedChunk:
    entity_matches = []
    try:
        entity_matches = json.loads(meta.get("entities", "[]") or "[]")
    except Exception:
        pass
    return RetrievedChunk(
        document=str(meta.get("document", "")),
        page=int(meta.get("page", 0)),
        chunk_index=int(meta.get("chunk_index", 0)),
        chunk=str(meta.get("text", "")),
        relevance_score=round(max(0.0, min(1.0, score)), 4),
        retrieval_source=retrieval_source,
        entity_matches=entity_matches,
    )
"""

with open(file_path, "w", encoding="utf-8") as f:
    f.write(new_content)
