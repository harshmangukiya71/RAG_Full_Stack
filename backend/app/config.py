"""
config.py — centralised settings via pydantic-settings.
All values can be overridden via environment variables or .env file.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ── NVIDIA NIM (OpenAI-compatible) ──────────────────────────────────────
    nvidia_api_key: str = "nvapi-xxxx"
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    nvidia_model: str = "meta/llama-3.1-70b-instruct"
    nvidia_max_tokens: int = 1024
    nvidia_temperature: float = 0.1          # low temp → factual, deterministic

    # ── Embedding ────────────────────────────────────────────────────────────
    # Summarization
    summary_chunk_size_tokens: int = 1200
    summary_chunk_overlap_tokens: int = 120
    summary_max_tokens_per_chunk: int = 350
    summary_final_max_tokens: int = 700

    embedding_model: str = "BAAI/bge-large-en-v1.5"
    embedding_device: str = "cpu"            # "cuda" if GPU available

    # ── Vector Store (ChromaDB) ──────────────────────────────────────────────
    chroma_persist_dir: str = "./data/chroma_db"
    collection_name: str = "legal_docs"

    # ── Chunking ─────────────────────────────────────────────────────────────
    chunk_size_tokens: int = 512
    chunk_overlap_tokens: int = 64

    # ── Retrieval ───────────────────────────────────────────────────
    bm25_top_k: int = 20
    dense_top_k: int = 20
    rerank_top_k: int = 5
    final_context_k: int = 5        # bumped from 3 → 5 for richer context

    # ── Hallucination mitigation thresholds ──────────────────────────
    min_relevance_score: float = 0.05   # lowered 0.10 → 0.05: cross-encoder sigmoid
                                        # scores legal chunks at 0.4-0.8; even 0.05
                                        # is sufficient signal to attempt an answer
    min_answer_coverage: float = 0.20        # token overlap fraction
    confidence_scale_factor: float = 1.6     # scalar applied to raw score

    # ── API ───────────────────────────────────────────────────────────────────
    # Conversational memory
    memory_backend: str = "auto"             # "auto", "redis", or "memory"
    redis_url: str = "redis://localhost:6379/0"
    memory_ttl_seconds: int = 60 * 60 * 24
    memory_max_turns: int = 4

    # Answer cache
    cache_enabled: bool = True
    cache_ttl_seconds: int = 60 * 60
    cache_max_entries: int = 512
    semantic_cache_threshold: float = 0.92

    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:3001"]


@lru_cache
def get_settings() -> Settings:
    return Settings()
