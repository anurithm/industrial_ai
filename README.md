# IndustrialAI — Explainable Predictive Maintenance & Failure Analysis Platform

> **AI/ML + Generative AI** · Machine Learning failure prediction · Isolation Forest anomaly detection · SHAP explainability · RAG-based GenAI maintenance assistant

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.157-009688?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-1.66-FF4B4B?logo=streamlit)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📌 Overview

IndustrialAI is an end-to-end industrial predictive maintenance platform that uses **machine learning** to predict machine failure risk, **Isolation Forest anomaly detection** to identify unusual sensor patterns, **SHAP explainable AI** to interpret predictions, and a **retrieval-augmented GenAI assistant** to answer maintenance questions using technical documentation.

## 📎 Live Demo

https://industrialai-ai.streamlit.app/

### Key Features

| Capability | Implementation |
|---|---|
| **Failure Prediction** | XGBoost classifier with threshold tuning (Test F1 = 0.86, ROC-AUC = 0.99) |
| **Anomaly Detection** | Isolation Forest trained on normal operating data only |
| **Explainable AI** | SHAP values with plain-language summaries for every prediction |
| **GenAI RAG Assistant** | Chroma vector store + Sentence Transformers + LLM (OpenRouter/Ollama) |
| **REST API** | FastAPI with 6 endpoints, Pydantic validation, SQLite history |
| **Dashboard** | 7-page Streamlit UI with dark industrial theme |
| **Testing** | 31 pytest tests covering data, model, API, and RAG |
| **DevOps** | Dockerfile, docker-compose, GitHub Actions CI |

---

## 🏗 Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     Streamlit Dashboard (7 pages)                │
│  Overview ▸ Predict ▸ Anomaly ▸ Explain ▸ RAG Chat ▸ History ▸ │
│                          Model Report                           │
└────────────────────────────┬─────────────────────────────────────┘
                             │ HTTP
┌────────────────────────────▼─────────────────────────────────────┐
│                   FastAPI REST API (:8000)                        │
│  /health  /predict  /anomaly  /explain  /maintenance/ask         │
│                     /history                                      │
└───┬──────────┬──────────┬──────────┬──────────┬─────────────────┘
    │          │          │          │          │
  Model    Anomaly     SHAP     Chroma+LLM   SQLite
 (XGBoost) (IForest) (Explain)  (RAG)       (History)
```

---

## 📂 Project Structure

```
IndustrialAI/
├── app.py                      # Application entry point
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Container build file
├── docker-compose.yml          # Multi-service orchestration
├── .env.example                # Environment variables template
├── .github/workflows/tests.yml # CI pipeline
│
├── src/
│   ├── config.py               # Centralised configuration
│   ├── data_loader.py          # Dataset loading, validation, auto-download
│   ├── preprocessing.py        # Leakage-safe preprocessing pipeline
│   ├── train_model.py          # Multi-model training & evaluation
│   ├── train_pipeline.py       # End-to-end training orchestration
│   ├── predict.py              # Inference with validation
│   ├── anomaly_detection.py    # Isolation Forest anomaly detector
│   ├── explainability.py       # SHAP-based explanations
│   ├── maintenance_rag.py      # RAG assistant (Chroma + LLM)
│   ├── database.py             # SQLAlchemy prediction history
│   └── api.py                  # FastAPI endpoints
│
├── frontend/
│   └── dashboard.py            # 7-page Streamlit dashboard
│
├── tests/
│   ├── test_data.py            # Data loading & validation tests
│   ├── test_model.py           # Model training & prediction tests
│   ├── test_api.py             # API endpoint tests
│   └── test_rag.py             # RAG assistant tests
│
├── docs/maintenance/           # Sample maintenance documentation for RAG
│   ├── maintenance_guide.md
│   └── sensor_thresholds.md
│
├── data/                       # Auto-created at runtime
├── models/                     # Trained model artifacts
├── reports/                    # Evaluation metrics JSON
└── chroma_db/                  # Vector store persistence
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+ (tested on 3.12)
- pip

### 1. Clone & install

```bash
git clone https://github.com/YOUR_USERNAME/IndustrialAI.git
cd IndustrialAI
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Train models

```bash
python -m src.train_pipeline
```

This will:
- Auto-download the UCI AI4I 2020 dataset (10,000 records)
- Train Logistic Regression, Random Forest, and XGBoost
- Select best model by PR-AUC on validation set
- Evaluate on held-out test set
- Train Isolation Forest anomaly detector
- Build Chroma vector store from maintenance docs

### 3. Start the API

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

### 4. Start the dashboard

```bash
streamlit run frontend/dashboard.py
```

Open **http://localhost:8501** in your browser.

---

## 🧠 GenAI RAG Setup (Optional)

The maintenance assistant works in **offline fallback mode** by default, returning retrieved documentation passages without LLM summarisation. To enable full LLM-powered responses:

### Option A: OpenRouter (free tier available)

```bash
cp .env.example .env
# Edit .env and set:
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=sk-or-v1-your-key-here
OPENROUTER_MODEL=meta-llama/llama-3.1-8b-instruct:free
```

### Option B: Local Ollama

```bash
# Install Ollama: https://ollama.ai
ollama pull llama3.1
# Set in .env:
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1
```

---

## 🐳 Docker

```bash
# Build and run both API + Dashboard
docker-compose up --build

# API: http://localhost:8000
# Dashboard: http://localhost:8501
```

---

## ✅ Test Results

```
31 passed in 12.26s

tests/test_api.py     ✓ 9 tests (health, predict, validate, history, RAG)
tests/test_data.py    ✓ 9 tests (schema, leakage, splits, validation)
tests/test_model.py   ✓ 7 tests (training, metrics, prediction, anomaly)
tests/test_rag.py     ✓ 6 tests (chunking, retrieval, Q&A, context)
```

Run tests:
```bash
pytest tests/ -v
```

---

## 📊 Model Performance

| Model | F1 | PR-AUC | ROC-AUC | Recall |
|---|---|---|---|---|
| Logistic Regression | 0.381 | 0.308 | 0.840 | 0.392 |
| Random Forest | 0.654 | 0.603 | 0.939 | 0.647 |
| **XGBoost (selected)** | **0.711** | **0.745** | **0.941** | **0.628** |

### Test Set (held-out, 1,500 samples)

| Metric | Value |
|---|---|
| Accuracy | 99.07% |
| Precision | 87.76% |
| Recall | 84.31% |
| F1 Score | 86.00% |
| PR-AUC | 0.906 |
| ROC-AUC | 0.990 |
| False Positives | 6 |
| False Negatives | 8 |

---

## 🔻 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check with model readiness |
| `POST` | `/predict` | Failure probability prediction |
| `POST` | `/anomaly` | Isolation Forest anomaly detection |
| `POST` | `/explain` | SHAP feature contributions |
| `POST` | `/maintenance/ask` | RAG-powered maintenance Q&A |
| `GET` | `/history` | Prediction audit trail |

Interactive API docs: **http://localhost:8000/docs**

---

## 📄 License

This project is released under the MIT License.

---

## 🛠 Tech Stack

- **ML:** scikit-learn, XGBoost, SHAP, Isolation Forest
- **GenAI:** Sentence Transformers, ChromaDB, OpenAI-compatible LLMs
- **Backend:** FastAPI, Pydantic, SQLAlchemy, Uvicorn
- **Frontend:** Streamlit with Plotly charts
- **Testing:** pytest (31 tests)
- **DevOps:** Docker, docker-compose, GitHub Actions
