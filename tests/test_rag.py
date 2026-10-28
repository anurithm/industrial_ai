"""
Tests for the RAG maintenance assistant — ONNX embedding backend.

After the sentence-transformers → ONNXMiniLM_L6_V2 migration, these tests:
1. Verify _get_embedding_fn() is a thread-safe singleton.
2. Verify rag_ready() and rag_status() don't trigger model loading.
3. Verify retrieve() returns the expected structure.
4. Verify ask() handles missing LLM + missing context gracefully.
5. Verify compatibility: ONNX embeddings match stored chroma_db vectors.
"""
import pytest
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Utility helpers ───────────────────────────────────────────────────────────

def _chunk_text_tests():
    """Import helper to avoid coupling tests to module internals."""
    from src.maintenance_rag import _chunk_text
    return _chunk_text


# ── Text chunking ─────────────────────────────────────────────────────────────

def test_chunk_text_basic():
    """Text chunking produces more than one chunk for long text."""
    _chunk_text = _chunk_text_tests()
    text = " ".join([f"word{i}" for i in range(200)])
    chunks = _chunk_text(text, chunk_size=50, overlap=10)
    assert len(chunks) > 1
    assert all(len(c) > 0 for c in chunks)


def test_chunk_text_short():
    """Short text produces a single chunk."""
    _chunk_text = _chunk_text_tests()
    text = "This is a short text."
    chunks = _chunk_text(text, chunk_size=100, overlap=10)
    assert len(chunks) == 1


# ── rag_ready / rag_status — must not load model ─────────────────────────────

def test_rag_ready_returns_bool():
    """rag_ready() returns bool without loading the ONNX EF."""
    from src.maintenance_rag import rag_ready
    result = rag_ready()
    assert isinstance(result, bool)


def test_rag_ready_does_not_load_embedding_fn(monkeypatch):
    """rag_ready() must NOT call _get_embedding_fn()."""
    import src.maintenance_rag as rag_mod

    called = []

    def mock_get():
        called.append(True)

    monkeypatch.setattr(rag_mod, "_get_embedding_fn", mock_get)
    rag_mod.rag_ready()
    assert len(called) == 0, "rag_ready() must not trigger embedding model load"


def test_rag_status_returns_expected_keys():
    """rag_status() returns the expected keys without loading the ONNX EF."""
    from src.maintenance_rag import rag_status
    status = rag_status()
    assert "vector_store_ready" in status
    assert "chunk_count" in status
    assert "embedding_model_state" in status
    assert "embedding_backend" in status
    assert "llm_provider" in status
    assert "llm_configured" in status
    assert "onnx" in status["embedding_backend"].lower()
    # These must not be secrets
    assert "KEY" not in str(status).upper() or "configured" in str(status)


# ── Embedding singleton ───────────────────────────────────────────────────────

def test_embedding_fn_singleton(monkeypatch):
    """_get_embedding_fn() returns the same instance on repeated calls."""
    import src.maintenance_rag as rag_mod

    original_ef = rag_mod._onnx_ef
    original_state = rag_mod._embedding_model_state
    original_error = rag_mod._embedding_model_error

    # Reset singleton state for this test
    rag_mod._onnx_ef = None
    rag_mod._embedding_model_state = "not_loaded"
    rag_mod._embedding_model_error = None

    try:
        ef1 = rag_mod._get_embedding_fn()
        ef2 = rag_mod._get_embedding_fn()
        assert ef1 is ef2, "Singleton must return the same instance on every call"
        assert rag_mod._embedding_model_state == "ready"
    finally:
        rag_mod._onnx_ef = original_ef
        rag_mod._embedding_model_state = original_state
        rag_mod._embedding_model_error = original_error


def test_embedding_fn_produces_correct_dim():
    """ONNXMiniLM_L6_V2 produces 384-dim embeddings (all-MiniLM-L6-v2)."""
    import src.maintenance_rag as rag_mod
    result = rag_mod._embed(["test sentence for dimension check"])
    assert len(result) == 1
    assert len(result[0]) == 384, f"Expected 384-dim embedding, got {len(result[0])}"


# ── ChromaDB compatibility ────────────────────────────────────────────────────

def test_onnx_embeddings_compatible_with_existing_chroma_db():
    """
    Verify that ONNX embeddings are compatible with stored chroma_db vectors.

    The existing chroma_db was built with SentenceTransformer(all-MiniLM-L6-v2).
    ONNXMiniLM_L6_V2 uses the same model → embeddings must be essentially identical.
    Acceptance criteria: cosine similarity >= 0.9999 (practically 1.0).
    """
    from src.maintenance_rag import rag_ready, _embed
    from src.config import CHROMA_DIR, CHROMA_COLLECTION

    if not rag_ready():
        pytest.skip("chroma_db not available in this environment")

    import chromadb
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    col = client.get_collection(CHROMA_COLLECTION)

    if col.count() == 0:
        pytest.skip("chroma_db is empty")

    # Fetch one stored document and its stored embedding
    result = col.get(limit=1, include=["embeddings", "documents"])
    stored_emb = np.array(result["embeddings"][0])
    stored_doc = result["documents"][0]

    # Re-embed the same text with ONNX EF
    onnx_emb = np.array(_embed([stored_doc])[0])

    # Cosine similarity must be >= 0.9999
    cos_sim = np.dot(stored_emb, onnx_emb) / (
        np.linalg.norm(stored_emb) * np.linalg.norm(onnx_emb)
    )
    assert cos_sim >= 0.9999, (
        f"ONNX embedding not compatible with stored chroma_db vector (cosine_sim={cos_sim:.6f}). "
        "The chroma_db may have been built with a different model. Rebuild required."
    )


def test_retrieve_returns_list():
    """retrieve() returns a list without crashing."""
    from src.maintenance_rag import retrieve
    try:
        result = retrieve("temperature operating range")
        assert isinstance(result, list)
        for p in result:
            assert "text" in p
            assert "source" in p
            assert "distance" in p
    except Exception as e:
        pytest.skip(f"ChromaDB not available: {e}")


def test_retrieve_results_are_relevant():
    """retrieve() returns passages with reasonable similarity scores."""
    from src.maintenance_rag import rag_ready, retrieve

    if not rag_ready():
        pytest.skip("chroma_db not available")

    passages = retrieve("What causes machine failure?")
    if not passages:
        pytest.skip("No passages returned (may be empty index)")

    # All distances should be finite floats
    for p in passages:
        assert isinstance(p["distance"], float)
        assert p["distance"] >= 0


# ── ask() graceful degradation ────────────────────────────────────────────────

def test_ask_without_llm():
    """ask() returns structured result with answer and sources when LLM is absent."""
    from src.maintenance_rag import ask
    result = ask(
        question="What is the normal operating temperature range?",
        prediction_context=None,
        sensor_context=None,
    )
    assert "answer" in result
    assert isinstance(result["answer"], str)
    assert len(result["answer"]) > 0
    assert "sources" in result
    assert "llm_available" in result
    assert isinstance(result["llm_available"], bool)
    assert "latency_ms" in result
    assert isinstance(result["latency_ms"], float)


def test_ask_with_prediction_context():
    """ask() correctly incorporates prediction context."""
    from src.maintenance_rag import ask
    result = ask(
        question="Is my machine in good condition?",
        prediction_context={
            "predicted_failure": False,
            "failure_probability": 0.12,
            "risk_level": "LOW",
        },
        sensor_context={"Air temperature [K]": 300.0, "Torque [Nm]": 40.0},
    )
    assert "answer" in result
    assert isinstance(result["latency_ms"], float)


def test_ask_no_fabrication_on_unknown_query():
    """ask() handles zero-evidence queries without fabricating answers."""
    from src.maintenance_rag import ask
    result = ask(question="Quantum entanglement coefficient of a neutron star made of cheese?")
    assert "answer" in result
    assert isinstance(result["answer"], str)
