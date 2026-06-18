FROM python:3.11-slim

WORKDIR /app

# System deps for PyMuPDF
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmupdf-dev \
    tesseract-ocr \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# NOTE: make sure requirements.txt includes `redis>=5.0` for L1 cache to work.
# If redis-py is missing the app silently falls back to in-memory only.
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create data directories
RUN mkdir -p data/pdfs data/chroma_db data/uploads data/graph tests

# Pre-download the unchanged cross-encoder reranker at build time.
# Embeddings are served by NVIDIA's API and do not download local model files.
RUN python -c "from sentence_transformers import CrossEncoder; \
    CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# Generate sample PDFs
RUN python generate_sample_pdfs.py

EXPOSE 8000

# Single worker — Redis handles shared cache across scaled replicas
CMD sh -c "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"