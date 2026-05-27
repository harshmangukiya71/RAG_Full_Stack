"""
ingestion.py — PDF parsing and semantic chunking for any document type.

Strategy:
  1. Use PyMuPDF (fitz) to extract text per-page with bounding-box awareness.
  2. Split text into sentences / paragraphs using universal punctuation rules.
  3. Group sentences into token-bounded chunks with overlap so that no idea
     is silently cut in half.
  4. Each chunk carries full provenance: filename, page number, chunk index.

Works with: legal contracts, resumes, stories, reports, textbooks — any PDF.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Generator

import fitz  # PyMuPDF

from app.models import Chunk

logger = logging.getLogger(__name__)

# ── Universal sentence / paragraph splitter ────────────────────────────────
# Works for any document type: legal, medical, narrative, technical, etc.
_SENTENCE_SPLIT_RE = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z\"\u2018\u2019])"   # sentence boundary (any domain)
    r"|(?<=\n)\s*\n+",                          # paragraph / blank-line break
    re.UNICODE,
)

# Rough chars-per-token approximation (BPE tokeniser average ~4 chars/token)
_CHARS_PER_TOKEN = 4


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _split_into_sentences(text: str) -> list[str]:
    """Split page text into semantic units (sentences / paragraphs).

    Works for any document domain — legal, resume, narrative, technical.
    """
    # Normalise excessive whitespace while preserving paragraph breaks
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    parts = _SENTENCE_SPLIT_RE.split(text)
    sentences: list[str] = []
    for part in parts:
        part = part.strip()
        if part:
            sentences.append(part)
    return sentences


def _build_chunks(
    sentences: list[str],
    document: str,
    page: int,
    chunk_size_tokens: int,
    chunk_overlap_tokens: int,
    start_chunk_index: int,
) -> list[Chunk]:
    """Pack sentences into token-bounded chunks with overlap."""
    chunks: list[Chunk] = []
    current_sentences: list[str] = []
    current_tokens: int = 0
    chunk_idx = start_chunk_index

    i = 0
    while i < len(sentences):
        sent = sentences[i]
        sent_tokens = _approx_tokens(sent)

        if current_tokens + sent_tokens <= chunk_size_tokens:
            current_sentences.append(sent)
            current_tokens += sent_tokens
            i += 1
        else:
            # Flush current chunk
            if current_sentences:
                text = " ".join(current_sentences).strip()
                if text:
                    chunks.append(Chunk(
                        document=document,
                        page=page,
                        chunk_index=chunk_idx,
                        text=text,
                        token_count=current_tokens,
                    ))
                    chunk_idx += 1

                # Overlap: keep last N tokens worth of sentences
                overlap_sentences: list[str] = []
                overlap_tokens = 0
                for s in reversed(current_sentences):
                    t = _approx_tokens(s)
                    if overlap_tokens + t <= chunk_overlap_tokens:
                        overlap_sentences.insert(0, s)
                        overlap_tokens += t
                    else:
                        break

                current_sentences = overlap_sentences
                current_tokens = overlap_tokens
            else:
                # Single sentence larger than chunk size — keep as-is
                chunks.append(Chunk(
                    document=document,
                    page=page,
                    chunk_index=chunk_idx,
                    text=sent.strip(),
                    token_count=sent_tokens,
                ))
                chunk_idx += 1
                i += 1

    # Flush tail
    if current_sentences:
        text = " ".join(current_sentences).strip()
        if text:
            chunks.append(Chunk(
                document=document,
                page=page,
                chunk_index=chunk_idx,
                text=text,
                token_count=current_tokens,
            ))

    return chunks


def parse_pdf(
    pdf_path: Path | str,
    chunk_size_tokens: int = 512,
    chunk_overlap_tokens: int = 64,
) -> list[Chunk]:
    """
    Parse a PDF file and return a list of Chunk objects with full provenance.

    Args:
        pdf_path: Absolute or relative path to the PDF.
        chunk_size_tokens: Target maximum tokens per chunk.
        chunk_overlap_tokens: Tokens of overlap between consecutive chunks.

    Returns:
        List of Chunk objects ordered by page then chunk index.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    document_name = pdf_path.name
    all_chunks: list[Chunk] = []
    global_chunk_idx = 0

    logger.info("Parsing PDF: %s", document_name)

    with fitz.open(str(pdf_path)) as doc:
        for page_num, page in enumerate(doc, start=1):
            raw_text = page.get_text("text")  # type: ignore[attr-defined]
            if not raw_text or not raw_text.strip():
                logger.debug("Page %d is empty — skipping", page_num)
                continue

            sentences = _split_into_sentences(raw_text)
            if not sentences:
                continue

            page_chunks = _build_chunks(
                sentences=sentences,
                document=document_name,
                page=page_num,
                chunk_size_tokens=chunk_size_tokens,
                chunk_overlap_tokens=chunk_overlap_tokens,
                start_chunk_index=global_chunk_idx,
            )
            all_chunks.extend(page_chunks)
            global_chunk_idx += len(page_chunks)

    logger.info(
        "Parsed '%s' → %d pages → %d chunks",
        document_name,
        len(doc) if False else page_num,  # type: ignore
        len(all_chunks),
    )
    return all_chunks


def iter_pdfs(directory: Path | str) -> Generator[Path, None, None]:
    """Yield all PDF paths in a directory (non-recursive)."""
    directory = Path(directory)
    for f in sorted(directory.iterdir()):
        if f.suffix.lower() == ".pdf":
            yield f
