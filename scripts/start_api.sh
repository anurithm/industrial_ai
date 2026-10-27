#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# scripts/start_api.sh
# Docker/cloud startup script for the IndustrialAI FastAPI backend.
#
# 1. Builds the Chroma RAG vector store if missing (first deploy)
# 2. Starts uvicorn on $PORT (Render injects PORT; defaults to 8000)
# ─────────────────────────────────────────────────────────────────────────────
set -e

PORT="${PORT:-8000}"

echo "=== IndustrialAI API Startup ==="
echo "Port: $PORT"
echo "Working dir: $(pwd)"

# Build Chroma vector store if the directory is empty or missing
if [ ! -d "chroma_db" ] || [ -z "$(ls -A chroma_db 2>/dev/null)" ]; then
    echo "[startup] Chroma vector store not found — building now..."
    python -c "
from src.maintenance_rag import build_vector_store
try:
    build_vector_store(force_rebuild=False)
    print('[startup] Vector store ready.')
except Exception as e:
    print(f'[startup] WARNING: Vector store build failed (non-fatal): {e}')
"
else
    echo "[startup] Chroma vector store found — skipping rebuild."
fi

echo "[startup] Starting uvicorn on port $PORT ..."
exec uvicorn src.api:app --host 0.0.0.0 --port "$PORT"
