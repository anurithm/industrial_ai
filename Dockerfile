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

# ── Pre-download the SentenceTransformer model at image build time ─────────────
# This is the most important step for Render free tier performance.
# Without this, the first /maintenance/ask call downloads ~90MB from HuggingFace
# at runtime, which takes 2–5 minutes and causes the 120s dashboard timeout.
# Baking the model into the image means it's available immediately at startup.
ENV SENTENCE_TRANSFORMERS_HOME=/app/.sentence_transformers
RUN python -c "\
import os; \
os.environ.setdefault('SENTENCE_TRANSFORMERS_HOME', '/app/.sentence_transformers'); \
from sentence_transformers import SentenceTransformer; \
m = SentenceTransformer('all-MiniLM-L6-v2'); \
print('Model pre-downloaded OK, embedding dim:', m.get_sentence_embedding_dimension()) \
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
