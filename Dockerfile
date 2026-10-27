FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/processed models reports chroma_db

# Expose API port
EXPOSE 8000

# Default: run the API
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
