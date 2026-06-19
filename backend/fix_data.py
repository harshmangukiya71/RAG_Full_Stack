"""
fix_data.py — One-shot fix script for LexRAG.

Run this from inside the backend directory:
    cd C:\\Users\\os\\Desktop\\rag-legal\\backend
    .venv\\Scripts\\python.exe fix_data.py

What this does:
1. Generates the 3 legal sample PDFs (NDA, SLA, IP Assignment)
2. Removes non-legal PDFs (resumes) from data/pdfs
3. Re-ingests the 3 legal PDFs into Qdrant and reports chunk counts
"""
import sys
from pathlib import Path

# ── Step 1: Generate PDFs ────────────────────────────────────────────────────
print("=" * 60)
print("[1/4] Generating legal sample PDFs...")
print("=" * 60)

try:
    # Ensure reportlab is available
    from reportlab.lib.pagesizes import A4  # noqa: F401
except ImportError:
    print("ERROR: reportlab not installed. Run: pip install reportlab")
    sys.exit(1)

# Inline import of generator
sys.path.insert(0, str(Path(__file__).parent))
import generate_sample_pdfs as gen

gen.create_nda()
gen.create_sla()
gen.create_ip_assignment()
print()

# ── Step 2: Remove non-legal PDFs ────────────────────────────────────────────
print("[2/4] Removing non-legal PDFs from data/pdfs...")
pdfs_dir = Path("data/pdfs")
legal_names = {"NDA-VendorX.pdf", "SLA-ProviderY.pdf", "IP-ContractorZ.pdf"}
if pdfs_dir.exists():
    for f in list(pdfs_dir.iterdir()):
        if f.suffix.lower() == ".pdf" and f.name not in legal_names:
            print(f"  Deleting: {f.name}")
            f.unlink()
print("  Remaining PDFs:", [f.name for f in pdfs_dir.glob("*.pdf")])
print()

# ── Step 3: Ingest legal PDFs into Qdrant ──────────────────────────────────
print("[3/4] Ingesting legal PDFs into Qdrant...")
print("      (This will download the embedding model on first run — ~1.3 GB)")
print()

try:
    from app.pipeline import RAGPipeline
    pipeline = RAGPipeline()
    results = pipeline.ingest_directory(pdfs_dir)
    print()
    print("-" * 60)
    for r in results:
        print(f"  ✓ {r.document}: {r.pages_processed} pages, {r.chunks_created} chunks")
    print("-" * 60)
    total_chunks = sum(r.chunks_created for r in results)
    print(f"  Total: {len(results)} documents, {total_chunks} chunks")
    print()
    print("=" * 60)
    print("  SUCCESS! Run the backend with:")
    print("  .venv\\Scripts\\uvicorn.exe app.main:app --reload --port 8000")
    print("=" * 60)
except Exception as e:
    print(f"\nERROR during ingestion: {e}")
    print("You can still start the server — it will ingest on startup.")
    sys.exit(1)
