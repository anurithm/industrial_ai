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

# Copy application code (includes committed model artifacts)
COPY . .

# Ensure runtime directories exist
RUN mkdir -p data/raw data/processed models reports chroma_db

# Make startup script executable
RUN chmod +x scripts/start_api.sh

# Expose default port (Render overrides via $PORT env var)
EXPOSE 8000

# Use startup script: builds Chroma vector store if missing, then starts uvicorn
CMD ["bash", "scripts/start_api.sh"]
