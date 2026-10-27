"""
Configuration management for IndustrialAI platform.
Reads from environment variables with sensible defaults.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
DOCS_DIR = BASE_DIR / "docs" / "maintenance"
CHROMA_DIR = BASE_DIR / "chroma_db"

# Ensure directories exist
for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR, DOCS_DIR, CHROMA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Model configuration
MODEL_VERSION = os.getenv("MODEL_VERSION", "1.0.0")
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))

# Dataset
DATASET_FILENAME = "ai4i2020.csv"
DATASET_PATH = RAW_DATA_DIR / DATASET_FILENAME

# Expected dataset columns (features + target)
FEATURE_COLUMNS = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
TARGET_COLUMN = "Machine failure"

# Individual failure mode labels (never used as features)
FAILURE_MODE_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]

# All columns that must NOT be used as features
LEAKAGE_COLUMNS = [TARGET_COLUMN] + FAILURE_MODE_COLUMNS + ["UDI", "Product ID"]

# Preprocessing
TEST_SIZE = 0.15
VAL_SIZE = 0.15  # fraction of remaining after test split
MIN_SAMPLES_PER_CLASS = 10

# API configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# Database
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DATA_DIR}/predictions.db")

# LLM Provider
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openrouter")  # "openrouter" or "ollama"

# OpenRouter
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct:free")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Ollama
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

# Anomaly detection
ANOMALY_CONTAMINATION = 0.05  # Expected fraction of outliers

# RAG
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHROMA_COLLECTION = "maintenance_docs"
RAG_N_RESULTS = 5
RAG_CHUNK_SIZE = 500
RAG_CHUNK_OVERLAP = 50
