# RAG Full Stack AI Assistant

A production-ready **Retrieval-Augmented Generation (RAG)** system built using **Python, LangChain, OpenAI API, Hugging Face Embeddings, FAISS/ChromaDB, FastAPI, and Streamlit**.

This project enables users to upload and query custom documents such as PDFs, notes, reports, or text files. Instead of relying only on a language model’s pre-trained knowledge, the system retrieves relevant information from user-provided documents and uses it to generate accurate, grounded, and context-aware responses.

---

# Project Overview

Retrieval-Augmented Generation (RAG) combines the power of:

- Information Retrieval
- Vector Search
- Large Language Models (LLMs)

The goal of this project is to build an intelligent document-based question-answering system where users can interact with their own knowledge base.

When a user asks a question:

1. The system searches through indexed documents.
2. Retrieves the most relevant text chunks.
3. Sends those chunks to the language model.
4. Generates a contextual and accurate answer.

This approach significantly improves:
- factual accuracy
- hallucination reduction
- domain-specific answering
- explainability

---

# Features

- Upload and process PDFs and text documents
- Semantic text chunking
- Embedding generation using Hugging Face/OpenAI
- Vector similarity search using FAISS/ChromaDB
- Context-aware answer generation
- FastAPI backend APIs
- Streamlit/Frontend UI integration
- Modular and scalable architecture
- Conversational retrieval pipeline
- Memory-aware responses
- Evaluation pipeline for RAG quality
- Document summarization support
- Docker support
- Environment-based configuration
- Extensible pipeline for future improvements

---

# How RAG Works

## 1. Document Loading
The system loads external documents such as:
- PDFs
- text files
- notes
- reports
- knowledge base documents

---

## 2. Text Splitting
Large documents are split into smaller chunks to:
- improve retrieval quality
- reduce token size
- preserve semantic meaning

---

## 3. Embedding Generation
Each chunk is converted into a dense vector representation using:
- Hugging Face Embeddings
- OpenAI Embeddings

Embeddings capture semantic meaning instead of simple keywords.

---

## 4. Vector Storage
Embeddings are stored inside:
- FAISS
- ChromaDB

This allows efficient similarity search over large document collections.

---

## 5. Query Processing
When a user asks a question:
- the query is converted into an embedding vector
- semantic similarity search is performed

---

## 6. Retrieval
The most relevant chunks are retrieved based on vector similarity.

---

## 7. Answer Generation
The retrieved chunks are passed to the language model as context.

The language model generates:
- grounded
- contextual
- accurate responses

---

# Project Architecture

```text
                ┌──────────────────┐
                │   User Query     │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │ Query Embedding  │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │ Vector Database  │
                │ (FAISS/ChromaDB) │
                └────────┬─────────┘
                         │
             Retrieve Relevant Chunks
                         │
                         ▼
                ┌──────────────────┐
                │ Retrieved Context│
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │ Language Model   │
                │ (OpenAI/LLM)     │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │ Final Response   │
                └──────────────────┘
```

---

# Tech Stack

## Backend
- Python
- FastAPI
- LangChain

## AI / NLP
- OpenAI API
- Hugging Face Transformers
- Sentence Transformers

## Vector Database
- FAISS
- ChromaDB

## Frontend
- Streamlit
- React / Next.js

## Deployment & DevOps
- Docker
- Docker Compose

---

# Folder Structure

```text
RAG/
│
├── backend/
│   │
│   ├── app/
│   │   ├── __init__.py
│   │   ├── cache.py
│   │   ├── config.py
│   │   ├── embeddings.py
│   │   ├── evaluation.py
│   │   ├── generation.py
│   │   ├── ingestion.py
│   │   ├── main.py
│   │   ├── memory.py
│   │   ├── models.py
│   │   ├── pipeline.py
│   │   ├── retrieval.py
│   │   ├── summarizer.py
│   │   └── vectorstore.py
│   │
│   ├── data/
│   ├── tests/
│   ├── .env
│   ├── .env.example
│   ├── DESIGN.md
│   ├── Dockerfile
│   ├── fix_data.py
│   ├── generate_sample_pdfs.py
│   ├── requirements.txt
│   └── setup_and_run.ps1
│
├── frontend/
│   ├── src/
│   ├── .env.local
│   ├── Dockerfile
│   ├── next-env.d.ts
│   ├── next.config.mjs
│   ├── package.json
│   ├── package-lock.json
│   └── tsconfig.json
│
├── docker-compose.yml
├── README.md
└── .gitignore
```

---

# Installation Steps

## 1. Clone Repository

```bash
git clone https://github.com/your-username/rag-project.git
cd rag-project
```

---

## 2. Create Virtual Environment

### Windows
```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux / Mac
```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r backend/requirements.txt
```

---

# Environment Variables

Create a `.env` file inside the backend directory.

```env
OPENAI_API_KEY=your_openai_api_key

EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

VECTOR_DB=chroma

CHROMA_DB_DIR=./chroma_db

MODEL_NAME=gpt-4o-mini
```

---

# How to Run the Project

## Run Backend

```bash
cd backend
uvicorn app.main:app --reload
```

Backend server:
```text
http://localhost:8000
```

---

## Run Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend server:
```text
http://localhost:3000
```

---

# Docker Setup

## Build and Run

```bash
docker-compose up --build
```

---

# Usage Instructions

1. Upload PDF or document
2. Documents are processed and chunked
3. Embeddings are generated
4. Chunks are stored in vector database
5. Ask questions through the UI/API
6. System retrieves relevant chunks
7. LLM generates final contextual response

---

# Example Query Flow

## User Question

```text
What are the key responsibilities mentioned in the contract?
```

---

## System Flow

1. Query converted to embedding
2. Similar chunks retrieved from vector DB
3. Retrieved context passed to LLM
4. Final answer generated

---

## Example Response

```text
The contract specifies responsibilities related to project delivery,
quality assurance, reporting, and compliance requirements.
```

---

# Benefits of Using RAG

- Reduces hallucinations
- Uses real-time/custom data
- Improves factual accuracy
- Supports domain-specific applications
- Scales across large document collections
- Better explainability
- Personalized AI assistant capability

---

# Limitations

- Retrieval quality depends on chunking strategy
- Large vector databases may require optimization
- Embedding quality impacts accuracy
- LLM inference can be expensive
- Context window limitations exist

---

# Future Improvements

- Hybrid search (BM25 + vector search)
- Multi-modal RAG
- OCR support for scanned PDFs
- Streaming responses
- Authentication system
- Cloud deployment (AWS/Azure/GCP)
- Kubernetes deployment
- Real-time document indexing
- Citation generation
- Conversational memory optimization
- Fine-tuned domain-specific models

---

# Conclusion

This project demonstrates how Retrieval-Augmented Generation (RAG) can significantly improve the quality and reliability of AI-generated responses by integrating external knowledge retrieval with powerful language models.

The system is modular, scalable, and production-oriented, making it suitable for:
- AI assistants
- enterprise knowledge bases
- document search systems
- research assistants
- customer support automation
- internal company tools

By combining vector databases, embeddings, retrieval pipelines, and LLMs, this project provides a strong foundation for building advanced AI-powered applications.

---

# Author

Harsh Mangukiya

GitHub: https://github.com/harshmangukiya71/RAG_Full_Stack
