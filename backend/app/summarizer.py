"""
summarizer.py - map-reduce PDF summarization service.

The service is separate from retrieval. It runs after PDF text extraction
during ingestion and stores the final summary as document metadata.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from openai import OpenAI

from app.config import get_settings
from app.models import Chunk

logger = logging.getLogger(__name__)

_CHARS_PER_TOKEN = 4

_MAP_SYSTEM_PROMPT = """You summarize document chunks for a RAG system.
Extract the key facts, entities, dates, obligations, numbers, and conclusions.
Stay faithful to the chunk. Do not add outside knowledge."""

_REDUCE_SYSTEM_PROMPT = """You combine partial document summaries into one final summary.
Write a concise, complete overview of the whole document. Preserve important
facts, names, dates, numbers, risks, and conclusions. Do not add outside knowledge."""


@dataclass(frozen=True)
class DocumentSummary:
    document: str
    summary: str
    chunk_summaries: list[str]


class DocumentSummarizer:
    """Map-reduce summarizer backed by the configured OpenAI-compatible LLM."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._client = OpenAI(
            base_url=self._settings.nvidia_base_url,
            api_key=self._settings.nvidia_api_key,
        )

    def summarize_chunks(self, document_name: str, chunks: list[Chunk]) -> DocumentSummary:
        if not chunks:
            return DocumentSummary(document=document_name, summary="", chunk_summaries=[])

        full_text = "\n\n".join(chunk.text for chunk in chunks if chunk.text.strip())
        text_chunks = self._split_text(
            full_text,
            max_tokens=self._settings.summary_chunk_size_tokens,
            overlap_tokens=self._settings.summary_chunk_overlap_tokens,
        )
        if not text_chunks:
            return DocumentSummary(document=document_name, summary="", chunk_summaries=[])

        logger.info("Summarizing '%s' with %d map chunks", document_name, len(text_chunks))
        partials = [self._summarize_single_chunk(text, i + 1, len(text_chunks)) for i, text in enumerate(text_chunks)]
        final_summary = self._combine_summaries(document_name, partials)
        return DocumentSummary(document=document_name, summary=final_summary, chunk_summaries=partials)

    def _split_text(self, text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
        max_chars = max(1000, max_tokens * _CHARS_PER_TOKEN)
        overlap_chars = max(0, overlap_tokens * _CHARS_PER_TOKEN)
        paragraphs = [p.strip() for p in text.splitlines() if p.strip()]

        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        for paragraph in paragraphs:
            paragraph_len = len(paragraph)
            if current and current_len + paragraph_len + 2 > max_chars:
                chunk = "\n\n".join(current).strip()
                chunks.append(chunk)
                current = [chunk[-overlap_chars:]] if overlap_chars and len(chunk) > overlap_chars else []
                current_len = sum(len(p) for p in current)

            if paragraph_len > max_chars:
                step = max_chars - overlap_chars or max_chars
                for start in range(0, paragraph_len, step):
                    part = paragraph[start:start + max_chars].strip()
                    if part:
                        chunks.append(part)
                current = []
                current_len = 0
            else:
                current.append(paragraph)
                current_len += paragraph_len + 2

        if current:
            chunks.append("\n\n".join(current).strip())

        return chunks

    def _chat(self, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        response = self._client.chat.completions.create(
            model=self._settings.nvidia_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=self._settings.nvidia_temperature,
            stream=False,
        )
        return (response.choices[0].message.content or "").strip()

    def _summarize_single_chunk(self, text: str, index: int, total: int) -> str:
        prompt = (
            f"Summarize chunk {index} of {total}. Focus on durable document facts.\n\n"
            f"CHUNK:\n{text}"
        )
        return self._chat(_MAP_SYSTEM_PROMPT, prompt, self._settings.summary_max_tokens_per_chunk)

    def _combine_summaries(self, document_name: str, summaries: list[str]) -> str:
        joined = "\n\n".join(f"Partial summary {i + 1}:\n{s}" for i, s in enumerate(summaries))
        prompt = (
            f"Document: {document_name}\n\n"
            "Combine these partial summaries into a final document summary with:\n"
            "- 1 short overview paragraph\n"
            "- key points as bullets\n"
            "- notable dates/numbers/parties when present\n\n"
            f"{joined}"
        )
        return self._chat(_REDUCE_SYSTEM_PROMPT, prompt, self._settings.summary_final_max_tokens)
