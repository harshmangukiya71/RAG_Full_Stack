# LexRAG — Production-Grade Legal Document RAG Pipeline

> **Stack**: FastAPI · ChromaDB · BGE-large · BM25 + Dense Hybrid · Cross-Encoder Re-ranking · NVIDIA NIM (Llama 3.1 70B) · Next.js 14

## Quick Start

### 1. Set your NVIDIA API key
```bash
cp .env.example .env
# Edit .env and replace nvapi-your-key-here with your actual key
```

### 2. Run locally without Docker

**Backend:**
```bash
cd backend
pip install -r requirements.txt

# Copy and fill in your API key
cp .env.example .env

# Generate the 3 sample PDFs
python generate_sample_pdfs.py

# Start the API server
uvicorn app.main:app --reload --port 8000
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev   # → http://localhost:3000
```

### 3. Run with Docker Compose
```bash
cp .env.example .env   # fill in NVIDIA_API_KEY
docker-compose up --build
# Frontend: http://localhost:3000
# Backend API: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

---

## Run the Evaluation Harness

```bash
cd backend

# Option A: via Python (skip ingestion if already done)
python -m app.evaluation --skip-ingest

# Option B: via API (click "Run Evaluation" button in UI, or:)
curl -X POST http://localhost:8000/evaluate | python -m json.tool
```

Expected output:
```
============================================================
EVALUATION REPORT — Precision@3
============================================================
  ✓ What is the notice period in the NDA with Vendor X?
  ✓ How long do the confidentiality obligations last...
  ✓ What is the limitation of liability cap in the NDA...
  ...
------------------------------------------------------------
  Total questions : 10
  Hits            : 9
  Precision@3     : 90.0%
============================================================
```

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `GET /health` | GET | Health check + document list |
| `POST /query` | POST | Query with `{"question": "..."}` |
| `POST /ingest` | POST | Upload PDF (multipart/form-data) |
| `GET /documents` | GET | List all ingested documents |
| `DELETE /documents/{name}` | DELETE | Remove a document |
| `POST /evaluate` | POST | Run Precision@3 harness |
| `GET /docs` | GET | Interactive Swagger UI |

---

## Architecture

See [DESIGN.md](backend/DESIGN.md) for full trade-off reasoning on:
- Chunking strategy (semantic + legal-aware)
- Embedding model (BGE-large vs OpenAI)
- Vector store (ChromaDB vs FAISS vs Pinecone)
- Retrieval (BM25 + Dense → RRF → Cross-Encoder)
- Hallucination mitigation (4 layers)
- Scaling to 50,000 documents

---

## Project Structure

```
rag-legal/
├── backend/
│   ├── app/
│   │   ├── config.py        # Settings (pydantic-settings)
│   │   ├── models.py        # Pydantic schemas
│   │   ├── ingestion.py     # PDF parsing + semantic chunking
│   │   ├── embeddings.py    # BGE-large embedding singleton
│   │   ├── vectorstore.py   # ChromaDB wrapper
│   │   ├── retrieval.py     # BM25 + Dense → RRF → Cross-encoder
│   │   ├── generation.py    # NVIDIA NIM + hallucination mitigation
│   │   ├── pipeline.py      # RAGPipeline (main interface)
│   │   ├── evaluation.py    # Precision@3 harness
│   │   └── main.py          # FastAPI application
│   ├── data/pdfs/           # Stored PDFs
│   ├── data/chroma_db/      # Persistent vector store
│   ├── tests/
│   │   └── eval_qa_pairs.json
│   ├── generate_sample_pdfs.py
│   ├── DESIGN.md
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   └── src/
│       ├── app/page.tsx     # Main chat UI
│       ├── components/      # UploadZone, SourceCard, ConfidenceBadge, EvalModal
│       └── lib/api.ts       # Typed API client
├── docker-compose.yml
└── README.md
```
