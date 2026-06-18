# setup_and_run.ps1 — Full setup script for LexRAG backend
# Run from: C:\Users\os\Desktop\rag-legal\backend\
# Usage:    .\setup_and_run.ps1

Set-Location $PSScriptRoot

Write-Host "`n======================================" -ForegroundColor Cyan
Write-Host " LexRAG Backend Setup" -ForegroundColor Cyan
Write-Host "======================================`n" -ForegroundColor Cyan

# Step 1: Generate the legal sample PDFs
Write-Host "[1/4] Generating legal sample PDFs..." -ForegroundColor Yellow
.venv\Scripts\python.exe generate_sample_pdfs.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to generate PDFs. Is reportlab installed?" -ForegroundColor Red
    exit 1
}

# Step 2: Remove resume PDFs from data/pdfs (they are not legal docs)
Write-Host "`n[2/4] Cleaning up non-legal PDFs from data/pdfs..." -ForegroundColor Yellow
Get-ChildItem -Path "data\pdfs" -Filter "*.pdf" | Where-Object {
    $_.Name -notmatch "^(NDA-VendorX|SLA-ProviderY|IP-ContractorZ)\.pdf$"
} | ForEach-Object {
    Write-Host "  Removing: $($_.Name)" -ForegroundColor DarkYellow
    Remove-Item $_.FullName -Force
}
Write-Host "  PDFs in data/pdfs:" -ForegroundColor Green
Get-ChildItem -Path "data\pdfs" -Filter "*.pdf" | ForEach-Object { Write-Host "    - $($_.Name)" }

# Step 3: Clear the ChromaDB so stale resume embeddings are gone
Write-Host "`n[3/4] Clearing old ChromaDB data..." -ForegroundColor Yellow
if (Test-Path "data\chroma_db") {
    Remove-Item -Recurse -Force "data\chroma_db"
    Write-Host "  ChromaDB cleared." -ForegroundColor Green
} else {
    Write-Host "  No ChromaDB found (fresh start)." -ForegroundColor Green
}

# Step 4: Start the FastAPI backend (it will auto-ingest the legal PDFs on startup)
Write-Host "`n[4/4] Starting FastAPI backend on the configured Uvicorn bind address..." -ForegroundColor Yellow
Write-Host "      The server will auto-ingest the 3 legal PDFs on first start." -ForegroundColor Gray
Write-Host "      Press Ctrl+C to stop.`n" -ForegroundColor Gray

.venv\Scripts\uvicorn.exe app.main:app --host 0.0.0.0 --port 8000 --reload
