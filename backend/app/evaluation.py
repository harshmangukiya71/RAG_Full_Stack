"""
evaluation.py — Retrieval evaluation harness with Precision@K, Recall@K, and MRR.

Two evaluation modes:

  1. AUTO (default, recommended):
     Samples chunks from the actually-uploaded documents, uses the LLM to
     generate one factual question per chunk, then measures whether the
     retriever finds that chunk when asked the generated question.
     Works with ANY uploaded PDF — no hardcoded document names.

  2. PREDEFINED (legacy):
     Reads Q&A pairs from tests/eval_qa_pairs.json.
     Only useful when the exact expected documents are also ingested.

Metrics computed for both modes:
  • Recall@1  — correct chunk is the #1 retrieved result
  • Recall@3  — correct chunk appears in top 3
  • Recall@5  — correct chunk appears in top 5
  • MRR       — Mean Reciprocal Rank: mean of 1/rank_i across all questions
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from openai import OpenAI

from app.config import get_settings
from app.models import Chunk, EvalPair, EvalReport, EvalResult

if TYPE_CHECKING:
    from app.pipeline import RAGPipeline

logger = logging.getLogger(__name__)

# How many chunks to retrieve per eval question (must be >= max K we care about)
_EVAL_TOP_K = 5


# ── Question generation ───────────────────────────────────────────────────────

def _generate_question_from_chunk(chunk_text: str, max_retries: int = 3) -> str | None:
    """
    Ask the LLM to generate ONE factual question answerable from this chunk.
    Returns None if generation fails or the output is not a valid question.
    Retries up to max_retries times on transient API errors.

    Key fixes:
      - max_tokens=200 so questions are never truncated before the '?'
      - Auto-repair: if LLM output is missing '?' but looks like a question, append it
    """
    settings = get_settings()
    client = OpenAI(base_url=settings.nvidia_base_url, api_key=settings.nvidia_api_key)

    system_msg = (
        "You are an evaluation dataset generator. "
        "Your only job is to output a single, complete question ending with '?'. "
        "Do NOT include any explanation, prefix, or quotation marks. "
        "Always end your output with a question mark."
    )
    user_msg = (
        "Generate ONE specific factual question that:\n"
        "- Can be answered directly and precisely from the excerpt below\n"
        "- Is NOT a yes/no question\n"
        "- Asks about a specific fact, number, name, date, or detail present in the text\n"
        "- Is self-contained (does not say 'in the excerpt' or 'according to the text')\n"
        "- MUST end with a question mark '?'\n\n"
        f"Excerpt:\n{chunk_text[:700]}\n\n"
        "Question:"
    )

    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=settings.nvidia_model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=200,   # increased from 80 → prevents truncation before '?'
                temperature=0.3,
            )
            q = resp.choices[0].message.content or ""
            q = q.strip().strip('"\' ').strip()

            # Auto-repair: if question looks valid but is missing '?', append it
            # This handles cases where the LLM forgets the trailing punctuation
            if q and not q.endswith("?") and len(q) > 10:
                # Check if it starts with a question word — strong signal it's a question
                question_starters = (
                    "what", "who", "when", "where", "which", "how", "why",
                    "whose", "whom", "is", "are", "was", "were", "does",
                    "did", "can", "could", "would", "will",
                )
                if q.lower().split()[0] in question_starters:
                    q = q + "?"
                    logger.debug("Auto-appended '?' to question: %r", q[:60])

            # Validate: must end with '?', reasonable length
            if q.endswith("?") and 10 < len(q) < 400:
                return q

            logger.warning(
                "Question generation attempt %d/%d: invalid output %r",
                attempt, max_retries, q[:80],
            )
        except Exception as exc:
            logger.warning(
                "Question generation attempt %d/%d failed: %s",
                attempt, max_retries, exc,
            )
    return None


# ── Shared metric helpers ─────────────────────────────────────────────────────

def _find_rank(
    retrieved: list[dict],
    expected_document: str,
    expected_page: int,
) -> int:
    """
    Return the 1-based rank of the first correct result, or 0 if not found.
    Correct = document name matches (case-insensitive) AND page within ±1.
    """
    for rank, source in enumerate(retrieved, start=1):
        doc_match = source["document"].lower() == expected_document.lower()
        page_match = abs(source["page"] - expected_page) <= 1
        if doc_match and page_match:
            return rank
    return 0


def _compute_report(
    pairs: list[EvalPair],
    results: list[EvalResult],
    hits_at_1: int,
    hits_at_3: int,
    hits_at_5: int,
    reciprocal_rank_sum: float,
) -> EvalReport:
    n = max(len(pairs), 1)
    recall_at_1 = round(hits_at_1 / n, 4)
    recall_at_3 = round(hits_at_3 / n, 4)
    recall_at_5 = round(hits_at_5 / n, 4)
    mrr = round(reciprocal_rank_sum / n, 4)

    return EvalReport(
        total_questions=len(pairs),
        hits_at_1=hits_at_1,
        hits_at_3=hits_at_3,
        hits_at_5=hits_at_5,
        recall_at_1=recall_at_1,
        recall_at_3=recall_at_3,
        recall_at_5=recall_at_5,
        mrr=mrr,
        precision_at_3=recall_at_3,   # legacy alias
        hits=hits_at_3,               # legacy alias
        results=results,
    )


def _run_retrieval_eval(
    pairs: list[EvalPair],
    pipeline: "RAGPipeline",
) -> EvalReport:
    """Core evaluation loop used by both auto and predefined modes."""
    results: list[EvalResult] = []
    hits_at_1 = hits_at_3 = hits_at_5 = 0
    reciprocal_rank_sum = 0.0

    for i, pair in enumerate(pairs, start=1):
        logger.info("[%d/%d] Q: %s", i, len(pairs), pair.question[:80])

        try:
            response = pipeline.query(pair.question, top_k=_EVAL_TOP_K)
        except Exception as exc:
            logger.error("Query failed for Q%d: %s", i, exc)
            results.append(EvalResult(
                question=pair.question,
                expected_document=pair.expected_document,
                expected_page=pair.expected_page,
                retrieved_top5=[],
                hit_at_1=False, hit_at_3=False, hit_at_5=False,
                rank=0, reciprocal_rank=0.0,
            ))
            continue

        retrieved_top5 = [
            {
                "document": s.document,
                "page": s.page,
                "chunk_preview": s.chunk[:150],
            }
            for s in response.sources[:_EVAL_TOP_K]
        ]

        rank = _find_rank(retrieved_top5, pair.expected_document, pair.expected_page)
        rr = (1.0 / rank) if rank > 0 else 0.0

        h1 = rank == 1
        h3 = 0 < rank <= 3
        h5 = 0 < rank <= 5

        if h1: hits_at_1 += 1
        if h3: hits_at_3 += 1
        if h5: hits_at_5 += 1
        reciprocal_rank_sum += rr

        results.append(EvalResult(
            question=pair.question,
            expected_document=pair.expected_document,
            expected_page=pair.expected_page,
            retrieved_top5=retrieved_top5,
            hit_at_1=h1, hit_at_3=h3, hit_at_5=h5,
            rank=rank,
            reciprocal_rank=round(rr, 4),
        ))

        status = f"✓ RANK-{rank}" if rank > 0 else "✗ MISS"
        logger.info(
            "  %s  RR=%.3f  (expected %s p.%d)",
            status, rr, pair.expected_document, pair.expected_page,
        )

    report = _compute_report(pairs, results, hits_at_1, hits_at_3, hits_at_5, reciprocal_rank_sum)
    _print_report(report)
    return report


# ── Mode 1: AUTO evaluation ───────────────────────────────────────────────────

def auto_evaluate(pipeline: "RAGPipeline", n_pairs: int = 10) -> EvalReport:
    """
    Auto-generate Q&A pairs from ingested documents and evaluate retrieval.

    Algorithm:
      1. Get all chunks currently in the BM25 index (= all ingested text).
      2. Sample n_pairs chunks spread evenly across the corpus.
      3. For each sampled chunk, call the LLM to generate one factual question.
      4. Run retrieval for each question.
      5. A "hit" = the retriever returned the SOURCE CHUNK (same doc + page ±1)
         in the top-K results.

    This works for ANY uploaded PDF without requiring predefined Q&A pairs.
    """
    from app.pipeline import RAGPipeline as _RAGPipeline  # noqa: F401

    all_chunks: list[Chunk] = pipeline._retriever._corpus_chunks
    if not all_chunks:
        raise ValueError("No documents are ingested. Upload at least one PDF first.")

    logger.info(
        "Auto-eval: %d total chunks across %d documents. Sampling %d.",
        len(all_chunks), len({c.document for c in all_chunks}), n_pairs,
    )

    # ── Sample chunks to evaluate ─────────────────────────────────────────────
    # Use ALL chunks — no length filter. Resume bullet points are short but
    # the LLM can still generate questions from them (e.g. "What is Harsh's CPI?")
    pool = all_chunks

    logger.info(
        "Pool: %d total chunks available for evaluation (no length filter).",
        len(pool),
    )

    # If pool is smaller than requested n_pairs, cap and warn
    if len(pool) < n_pairs:
        logger.warning(
            "Only %d chunks available but n_pairs=%d was requested. "
            "Capping evaluation to %d questions.",
            len(pool), n_pairs, len(pool),
        )
        n_pairs = len(pool)

    if len(pool) <= n_pairs:
        sampled = pool
    else:
        # Spread evenly across the corpus (not just the beginning)
        indices = [int(i * len(pool) / n_pairs) for i in range(n_pairs)]
        sampled = [pool[i] for i in indices]

    logger.info(
        "Sampling %d/%d chunks for evaluation.",
        len(sampled), len(pool),
    )

    # Generate one question per sampled chunk
    pairs: list[EvalPair] = []
    logger.info("Generating %d questions via LLM (this takes ~10–30 seconds)...", len(sampled))

    for i, chunk in enumerate(sampled, start=1):
        logger.info("  Generating Q %d/%d from %s p.%d", i, len(sampled), chunk.document, chunk.page)
        question = _generate_question_from_chunk(chunk.text)
        if question:
            pairs.append(EvalPair(
                question=question,
                expected_document=chunk.document,
                expected_page=chunk.page,
                answer_hint="",
            ))
            logger.info("    → %s", question[:80])
        else:
            logger.warning("    → Question generation failed for this chunk, skipping.")

    if not pairs:
        raise ValueError(
            "Could not generate any questions. "
            "Check your NVIDIA API key and model connection."
        )

    logger.info("Generated %d/%d questions. Running retrieval evaluation...", len(pairs), len(sampled))
    return _run_retrieval_eval(pairs, pipeline)


# ── Mode 2: PREDEFINED evaluation ────────────────────────────────────────────

def run_evaluation(qa_path: Path, pipeline: "RAGPipeline") -> EvalReport:
    """
    Run evaluation from a predefined Q&A JSON file.
    Only accurate when the expected documents are actually ingested.
    """
    with open(qa_path, encoding="utf-8") as f:
        raw_pairs = json.load(f)

    pairs = [EvalPair(**p) for p in raw_pairs]
    logger.info("Predefined evaluation: %d questions from %s", len(pairs), qa_path)
    return _run_retrieval_eval(pairs, pipeline)


# ── Pretty-print ──────────────────────────────────────────────────────────────

def _print_report(report: EvalReport) -> None:
    n = report.total_questions
    print("\n" + "=" * 65)
    print("  EVALUATION REPORT")
    print("=" * 65)
    print(f"  {'Metric':<28}  {'Score':>8}  {'Hits':>8}")
    print("-" * 65)
    print(f"  {'Recall@1 (correct is #1 result)':<28}  {report.recall_at_1:>7.1%}  {report.hits_at_1:>5}/{n}")
    print(f"  {'Recall@3 (correct in top 3)':<28}  {report.recall_at_3:>7.1%}  {report.hits_at_3:>5}/{n}")
    print(f"  {'Recall@5 (correct in top 5)':<28}  {report.recall_at_5:>7.1%}  {report.hits_at_5:>5}/{n}")
    print(f"  {'MRR (Mean Reciprocal Rank)':<28}  {report.mrr:>8.4f}")
    print("=" * 65)
    print("\n  Per-question results:")
    for r in report.results:
        rank_str = f"rank {r.rank}" if r.rank > 0 else "not found"
        icon = "✓" if r.hit_at_3 else "✗"
        print(f"    {icon} [{rank_str:>9s}]  {r.question[:60]}")
    print("=" * 65 + "\n")


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

    parser = argparse.ArgumentParser(description="Run RAG retrieval evaluation")
    parser.add_argument("--qa-path", type=Path, default=None,
                        help="Path to predefined Q&A JSON (omit to use auto mode)")
    parser.add_argument("--ingest-dir", type=Path, default=Path("data/pdfs"))
    parser.add_argument("--skip-ingest", action="store_true")
    parser.add_argument("--n-pairs", type=int, default=10,
                        help="Number of auto-generated Q&A pairs (auto mode only)")
    args = parser.parse_args()

    from app.pipeline import RAGPipeline
    pipeline = RAGPipeline()

    if not args.skip_ingest:
        print(f"Ingesting PDFs from {args.ingest_dir}...")
        for r in pipeline.ingest_directory(args.ingest_dir):
            print(f"  Ingested: {r.document} ({r.chunks_created} chunks)")

    if args.qa_path:
        report = run_evaluation(args.qa_path, pipeline)
    else:
        print("Using AUTO mode — generating questions from uploaded documents...")
        report = auto_evaluate(pipeline, n_pairs=args.n_pairs)

    report_path = Path("tests/eval_report.json")
    report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    print(f"Full report saved to {report_path}")

    sys.exit(0 if report.recall_at_3 >= 0.7 else 1)
