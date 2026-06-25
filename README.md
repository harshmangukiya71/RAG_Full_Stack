# DocRAG - Hybrid Agentic RAG Assistant

## Live Demo

Frontend:
[https://agentic-rag-full-stack-31x3rxy8y.vercel.app/](https://agentic-rag-full-stack.vercel.app/)

Backend API:
https://agenticragfullstack-production.up.railway.app/docs



## Architecture

```text
                                        ┌─────────────────────────┐
                                        │      User Query         │
                                        └────────────┬────────────┘
                                                     │
                                                     ▼
                                        ┌─────────────────────────┐
                                        │     FastAPI Backend     │
                                        └────────────┬────────────┘
                                                     │
                                                     ▼
                                        ┌─────────────────────────┐
                                        │ Query Classification    │
                                        │                         │
                                        │ • Entity Lookup         │
                                        │ • Semantic Search       │
                                        │ • Counting / Ranking    │
                                        │ • Comparison            │
                                        │ • Temporal Queries      │
                                        │ • Multi-Hop Reasoning   │
                                        └────────────┬────────────┘
                                                     │
                                                     ▼
                    ┌─────────────────────────────────────────────────────────┐
                    │                Hybrid Retrieval Engine                  │
                    │                                                         │
                    │ • ChromaDB Dense Vector Search                          │
                    │ • BM25 Keyword Search                                   │
                    │ • Neo4j Graph Retrieval                                 │
                    └───────────────┬─────────────────────────────────────────┘
                                    │
                                    ▼
                    ┌──────────────────────────────────────────┐
                    │     Reciprocal Rank Fusion (RRF)         │
                    │       Aggregates Retrieval Results       │
                    └────────────────┬─────────────────────────┘
                                     │
                                     ▼
                    ┌──────────────────────────────────────────┐
                    │ Graph Boosting & Evidence Filtering      │
                    └────────────────┬─────────────────────────┘
                                     │
                                     ▼
                    ┌──────────────────────────────────────────┐
                    │ Cross Encoder Re-Ranker                  │
                    │ ms-marco-MiniLM-L-6-v2                   │
                    └────────────────┬─────────────────────────┘
                                     │
                                     ▼
                    ┌──────────────────────────────────────────┐
                    │ Top Relevant Chunks + Graph Context      │
                    └────────────────┬─────────────────────────┘
                                     │
                                     ▼
                    ┌──────────────────────────────────────────┐
                    │            Reasoning Agent               │
                    │                                          │
                    │ • Aggregation                            │
                    │ • Counting                               │
                    │ • Ranking                                │
                    │ • Comparisons                            │
                    │ • Evidence Validation                    │
                    │ • Sufficiency Checks                     │
                    └────────────────┬─────────────────────────┘
                                     │
                                     ▼
                    ┌──────────────────────────────────────────┐
                    │ NVIDIA Llama 3.3 70B                    │
                    │ Grounded Answer Generation               │
                    └────────────────┬─────────────────────────┘
                                     │
                                     ▼
                    ┌──────────────────────────────────────────┐
                    │ Confidence & Faithfulness Evaluation     │
                    └────────────────┬─────────────────────────┘
                                     │
                                     ▼
                    ┌──────────────────────────────────────────┐
                    │ Final Response                           │
                    │                                          │
                    │ • Generated Answer                       │
                    │ • Source Citations                       │
                    │ • Confidence Score                       │
                    │ • Query Classification                   │
                    └──────────────────────────────────────────┘



=================================================================================
                               DOCUMENT INGESTION PIPELINE
=================================================================================


                    ┌───────────────────────────┐
                    │ PDF / Image / Screenshot  │
                    └─────────────┬─────────────┘
                                  │
                                  ▼
                    ┌───────────────────────────┐
                    │ PyMuPDF Text Extraction   │
                    └─────────────┬─────────────┘
                                  │
                          Enough Native Text?
                              ┌───┴───┐
                              │       │
                             Yes      No
                              │       │
                              ▼       ▼
                    ┌──────────────┐  ┌─────────────────────┐
                    │ Clean Text   │  │ OCR Pipeline        │
                    │ Processing   │  │ PaddleOCR           │
                    └──────┬───────┘  │ Tesseract Fallback  │
                           │          └──────────┬──────────┘
                           └──────────┬──────────┘
                                      │
                                      ▼
                    ┌───────────────────────────┐
                    │ Semantic Chunking         │
                    │ 512 Tokens + Overlap      │
                    └─────────────┬─────────────┘
                                  │
                                  ▼
                    ┌───────────────────────────┐
                    │ Metadata Enrichment       │
                    │                           │
                    │ • Page Number             │
                    │ • Section                 │
                    │ • OCR Confidence          │
                    │ • Extraction Method       │
                    └─────────────┬─────────────┘
                                  │
                                  ▼
                    ┌───────────────────────────┐
                    │ Entity & Relation         │
                    │ Extraction using LLM      │
                    └─────────────┬─────────────┘
                                  │
                                  ▼
                    ┌───────────────────────────┐
                    │ Entity Normalization      │
                    │ & Deduplication           │
                    └───────┬─────────┬─────────┘
                            │         │
                            ▼         ▼
                    ┌────────────┐ ┌────────────┐
                    │ Embeddings │ │ Neo4j      │
                    │ BGE Large  │ │ Knowledge  │
                    │            │ │ Graph      │
                    └─────┬──────┘ └─────┬──────┘
                          │              │
                          ▼              ▼
                    ┌────────────┐ ┌────────────┐
                    │ ChromaDB   │ │ Entity     │
                    │ Vector DB  │ │ Relations  │
                    └─────┬──────┘ └─────┬──────┘
                          │              │
                          └──────┬───────┘
                                 ▼
                    ┌───────────────────────────┐
                    │ BM25 Index Construction   │
                    └─────────────┬─────────────┘
                                  │
                                  ▼
                    ┌───────────────────────────┐
                    │ Hybrid Retrieval Ready    │
                    │ Knowledge Corpus          │
                    └───────────────────────────┘

## What This Project Uses

| Area | Technology |
| --- | --- |
| Backend API | FastAPI, Pydantic |
| Frontend | Next.js 14, React 18, TypeScript |
| LLM provider | NVIDIA OpenAI-compatible API |
| Default LLM | `meta/llama-3.3-70b-instruct` |
| Embedding model | `BAAI/bge-large-en-v1.5` via `sentence-transformers` |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Vector database | Qdrant Cloud (Cosine, 4096 dim) or ChromaDB |
| Graph database | Neo4j |
| Keyword search | BM25 using `rank-bm25` |
| OCR | Native PDF extraction with PyMuPDF, PaddleOCR fallback, Tesseract fallback |
| Cache | Redis L1 cache, in-memory/disk fallback L2 cache |
| Memory | Redis-backed chat memory with in-memory fallback |
| Evaluation | Auto-generated verified QA pairs, Recall@1/@3/@5, MRR, grounding metrics |
| Deployment | Docker, Docker Compose |

## Core Features

- Upload PDFs and image documents through the API/frontend.
- Extract text from native PDFs and OCR scanned pages/images.
- Build semantic chunks with page, chunk, section, extraction method, and OCR confidence metadata.
- Extract financial/business entities and relationships into Neo4j.
- Store local embeddings in ChromaDB.
- Retrieve using a hybrid approach: BM25 + dense vector search + Neo4j entity graph.
- Classify queries into lookup, structured reasoning, relationship, temporal, multi-hop, aggregation, counting, ranking, comparison, and analytical query types.
- Rerank retrieved evidence with a cross-encoder.
- Run a reasoning agent before generation for calculations, rankings, counts, and evidence sufficiency.
- Generate grounded answers with source citations and confidence scores.
- Cache repeated and semantically similar answers.
- Evaluate retrieval and grounding quality from uploaded documents.

## Architecture

```text
frontend/Next.js
    |
    | REST API
    v
backend/FastAPI
    |
    |-- ingestion.py + ocr.py        -> document parsing and chunking
    |-- entities.py                  -> entity and relation abstraction
    |-- embeddings.py                -> local embedding model
    |-- vectorstore.py               -> ChromaDB persistence
    |-- graph.py                     -> Neo4j entity graph
    |-- retrieval.py                 -> BM25 + dense + graph + rerank
    |-- query_agent.py               -> query classifier and strategy router
    |-- reasoning_agent.py           -> structured reasoning over evidence
    |-- generation.py                -> grounded LLM answer generation
    |-- evaluation.py                -> RAG quality evaluation
    |-- cache.py / memory.py         -> answer cache and chat memory
```

## Query Classification

The classifier agent is lightweight and runs without an LLM call. It maps a query to a retrieval strategy and `top_k` depth.

Important query families include:

- Lookup: `ENTITY_LOOKUP`, `SEMANTIC_LOOKUP`, `LOOKUP`
- Structured reasoning: `NUMERICAL_FILTER`, `COUNTING`, `AGGREGATION`, `RANKING`
- Relationship reasoning: `COMPARISON`, `RELATIONSHIP`, `MULTI_HOP`, `TEMPORAL`, `ANALYTICAL`

These classifications control whether the system leans more on exact metadata, dense search, graph retrieval, wider corpus retrieval, or iterative retrieval.

## Main API Endpoints

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Check server health and indexed document count |
| `POST` | `/ingest` | Upload and ingest a PDF/image document |
| `POST` | `/query` | Ask a question over uploaded documents |
| `GET` | `/documents` | List indexed documents |
| `GET` | `/documents/{document_name}` | Get document metadata and summary |
| `DELETE` | `/documents/{document_name}` | Remove one document from indexes and disk |
| `POST` | `/reset` | Remove all documents |
| `GET` | `/entities` | Search graph entities |
| `GET` | `/entities/{entity_id}/neighbors` | Inspect graph neighbors |
| `POST` | `/evaluate` | Auto-generate verified QA pairs and evaluate |
| `POST` | `/evaluate/predefined` | Run predefined evaluation pairs |
| `GET` | `/cache/status` | Check cache readiness |
| `POST` | `/cache/clear` | Clear answer cache |
| `POST` | `/chat/clear` | Clear conversational memory |
| `POST` | `/debug/query` | Inspect classification and retrieval results |

## Environment Variables

Create `backend/.env` from `backend/.env.example`.

```env
NVIDIA_API_KEY=nvapi-------
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
NVIDIA_MODEL=meta/llama-3.3-70b-instruct
NVIDIA_EMBEDDING_MODEL=nvidia/nv-embed-v1
NVIDIA_MAX_TOKENS=1024
NVIDIA_TEMPERATURE=0.1

GRAPH_BACKEND=neo4j

VECTOR_DB=qdrant
QDRANT_URL=https://your-qdrant-cluster-url.cloud.qdrant.io
QDRANT_API_KEY=your-api-key
QDRANT_COLLECTION=agentic-rag
NEO4J_URI=neo4j+s://52f6fec6.databases.neo4j.io
NEO4J_USERNAME=52f6fec6
NEO4J_PASSWORD=7XcwEdmn1upF-Is02B7GiUgLVeq4lltV8Me33TTjkUc
NEO4J_DATABASE=52f6fec6
AURA_INSTANCEID=52f6fec6
AURA_INSTANCENAME=Free instance


# Agentic RAG settings
QUERY_AGENT_ENABLED=true
REASONING_AGENT_ENABLED=true
MAX_RETRIEVAL_ITERATIONS=3
CACHE_LOAD_MONITORING=true
CACHE_READY_PERCENT=100

#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
```

Qdrant Cloud setup:
Create a Qdrant Cloud cluster and get your URL and API Key. The app will automatically create the `agentic-rag` collection configured for 4096 dimensions with Cosine distance and BM25 sparse vectors enabled.

Neo4j is required because `GRAPH_BACKEND` is configured for Neo4j only. Redis is optional; if Redis is unavailable, the app falls back to in-memory and local disk caching where supported.

```

## Docker

```bash
docker-compose up --build
```

The compose file starts the FastAPI backend and Next.js frontend. Ensure required external services such as Neo4j, and optionally Redis, are available or configured for your environment.

## Typical Usage

1. Start Neo4j.
2. Configure `backend/.env` with your NVIDIA API key and model settings.
3. Start the backend and frontend.
4. Upload a PDF, scanned PDF, screenshot, or image.
5. Wait for ingestion to complete: OCR/text extraction, chunking, entity abstraction, embeddings, Qdrant/ChromaDB insert, Neo4j graph insert, and BM25 rebuild.
6. Ask questions from the frontend or `/query`.
7. Review answer sources, confidence, and query classification.
8. Run `/evaluate` to measure retrieval and grounding quality on the current corpus.

## Evaluation

The project includes an automatic evaluation pipeline. It samples indexed chunks, asks the configured LLM to generate factual, relational, and comparative questions, verifies each question against the source chunk, and then evaluates retrieval quality.

Metrics include:

- Recall@1, Recall@3, Recall@5
- MRR
- answer faithfulness
- evidence coverage
- unsupported answer rate
- speculative graph edge rate

## Repository Structure

```text
RAG/
|-- backend/
|   |-- app/
|   |   |-- main.py
|   |   |-- pipeline.py
|   |   |-- ingestion.py
|   |   |-- ocr.py
|   |   |-- entities.py
|   |   |-- graph.py
|   |   |-- embeddings.py
|   |   |-- vectorstore.py
|   |   |-- retrieval.py
|   |   |-- query_agent.py
|   |   |-- reasoning_agent.py
|   |   |-- generation.py
|   |   |-- evaluation.py
|   |   |-- cache.py
|   |   |-- memory.py
|   |   `-- models.py
|   |-- data/
|   |-- requirements.txt
|   |-- Dockerfile
|   `-- .env.example
|-- frontend/
|   |-- src/
|   |-- package.json
|   |-- Dockerfile
|   `-- next.config.mjs
|-- docker-compose.yml
`-- README.md
```

DocRAG is a full-stack document question-answering system for PDFs, scanned documents, screenshots, and image files. It combines dense vector search, keyword search, and graph-based entity retrieval so answers can use both semantic similarity and structured entity relationships.

## Notes

- The LLM is used for grounded answer generation, document summaries, entity/relation extraction, and evaluation question generation.
- The embedding model runs locally through Sentence Transformers.
- ChromaDB persists vectors under `backend/data/chroma_db` by default.
- Neo4j stores entities, document mentions, and `RELATED` relationships extracted from document chunks.
- The final answer is produced only after retrieval, reranking, reasoning, and confidence scoring.
