#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# scripts/start_api.sh — Cloud startup script for IndustrialAI FastAPI
#
# Memory-efficient startup for Render free tier (512 MB limit):
#  - Does NOT build the Chroma vector store at startup (was the OOM cause)
#  - Chroma index is pre-built and committed to Git — no rebuild needed
#  - SentenceTransformer loads lazily on the first /maintenance/ask request
#  - Uvicorn starts immediately on $PORT
# ─────────────────────────────────────────────────────────────────────────────
set -e

PORT="${PORT:-8000}"

echo "=== IndustrialAI API Startup ==="
echo "Port        : $PORT"
echo "Working dir : $(pwd)"
echo "Python      : $(python --version)"

# Sanity check: model artifacts must be present (committed to Git)
if [ ! -f "models/best_model.joblib" ]; then
    echo "[startup] ERROR: models/best_model.joblib not found."
    echo "          Commit model artifacts to the repository first."
    exit 1
fi
echo "[startup] Model artifacts present."

# Chroma vector store pre-built and committed — no build step at startup.
# SentenceTransformer loads lazily on first /maintenance/ask call (~300 MB,
# takes a few seconds but does NOT block startup or /health, /predict, /anomaly).
if [ -f "chroma_db/chroma.sqlite3" ]; then
    echo "[startup] Chroma vector store found — RAG ready (lazy load on first use)."
else
    echo "[startup] WARNING: chroma_db not found — /maintenance/ask will return fallback answers."
    echo "          Run: python -c 'from src.maintenance_rag import build_vector_store; build_vector_store()'"
    echo "          Then commit chroma_db/ to Git."
fi

echo "[startup] Starting uvicorn on 0.0.0.0:$PORT ..."
exec uvicorn src.api:app --host 0.0.0.0 --port "$PORT"
