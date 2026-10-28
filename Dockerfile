FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies — dashboard + API combined
COPY requirements.txt requirements_api.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements_api.txt

# ── Pre-download the ONNX embedding model at image build time ─────────────────
# ChromaDB's ONNXMiniLM_L6_V2 downloads the ONNX model weights (~23 MB) on
# first use. We bake it into the image so there's no network call at runtime.
# RAM cost: ~80 MB (vs ~380 MB for sentence-transformers + PyTorch).
RUN python -c "\
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2; \
ef = ONNXMiniLM_L6_V2(); \
result = ef(['model warmup test']); \
print('ONNX EF ready, embedding dim:', len(result[0])) \
"

# Copy application code (includes committed model artifacts and chroma_db)
COPY . .

# Ensure runtime directories exist
RUN mkdir -p data/raw data/processed models reports

# Make startup script executable
RUN chmod +x scripts/start_api.sh

# Expose default port (Render overrides via $PORT env var)
EXPOSE 8000

# Use startup script: validates artifacts, then starts uvicorn on $PORT
CMD ["bash", "scripts/start_api.sh"]
