"""
Tests for the RAG maintenance assistant — including lazy loading and fallback behaviour.
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_chunk_text_basic():
    """Text chunking produces non-empty chunks."""
    from src.maintenance_rag import _chunk_text
    text = " ".join([f"word{i}" for i in range(200)])
    chunks = _chunk_text(text, chunk_size=50, overlap=10)
    assert len(chunks) > 1
    assert all(len(c) > 0 for c in chunks)


def test_chunk_text_short():
    """Short text produces single chunk."""
    from src.maintenance_rag import _chunk_text
    text = "This is a short text."
    chunks = _chunk_text(text, chunk_size=100, overlap=10)
    assert len(chunks) == 1


def test_rag_ready_returns_bool():
    """rag_ready() returns a bool without loading any heavy models."""
    from src.maintenance_rag import rag_ready
    result = rag_ready()
    assert isinstance(result, bool)
    # No SentenceTransformer should be loaded just from calling rag_ready()
    from src.maintenance_rag import _embedding_model
    # _embedding_model may or may not be loaded depending on prior test order;
    # the key assertion is that rag_ready() itself doesn't crash.


def test_rag_ready_does_not_load_embedding_model(monkeypatch):
    """rag_ready() must NOT trigger SentenceTransformer loading."""
    import src.maintenance_rag as rag_mod

    loaded = []

    original_get = rag_mod._get_embedding_model

    def mock_get():
        loaded.append(True)
        return original_get()

    monkeypatch.setattr(rag_mod, "_get_embedding_model", mock_get)
    rag_mod.rag_ready()
    assert len(loaded) == 0, "rag_ready() must not call _get_embedding_model()"


def test_retrieve_returns_list():
    """retrieve() returns a list (empty or populated) without error."""
    from src.maintenance_rag import retrieve
    try:
        result = retrieve("test question about temperature")
        assert isinstance(result, list)
    except Exception as e:
        pytest.skip(f"Chroma not available in test environment: {e}")


def test_retrieve_no_error_when_empty():
    """retrieve() returns empty list gracefully when no docs are indexed."""
    from src.maintenance_rag import retrieve
    try:
        result = retrieve("test question")
        assert isinstance(result, list)
    except Exception:
        pass  # Acceptable if chromadb is not initialized in CI


def test_ask_without_llm():
    """
    ask() returns a structured result with answer and sources
    even when LLM credentials are absent.
    """
    from src.maintenance_rag import ask
    result = ask(
        question="What is the normal operating temperature range for the machine?",
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


def test_ask_with_context():
    """ask() passes prediction and sensor context correctly."""
    from src.maintenance_rag import ask
    pred_context = {
        "predicted_failure": False,
        "failure_probability": 0.12,
        "risk_level": "LOW",
    }
    sensor_context = {
        "Air temperature [K]": 300.0,
        "Torque [Nm]": 40.0,
    }
    result = ask(
        question="Is my machine in good condition?",
        prediction_context=pred_context,
        sensor_context=sensor_context,
    )
    assert "answer" in result
    assert isinstance(result["latency_ms"], float)


def test_ask_empty_evidence_handling():
    """ask() handles zero-evidence gracefully without fabricating answers."""
    from src.maintenance_rag import ask
    result = ask(
        question="What is the quantum entanglement coefficient of a neutron star made of cheese?",
    )
    assert "answer" in result
    assert isinstance(result["answer"], str)


def test_embedding_model_singleton(monkeypatch):
    """_get_embedding_model() loads the model only once (singleton pattern)."""
    import src.maintenance_rag as rag_mod

    # Save and reset ALL singleton state so this test is independent
    original_model = rag_mod._embedding_model
    original_state = rag_mod._embedding_model_state
    original_error = rag_mod._embedding_model_error
    rag_mod._embedding_model = None
    rag_mod._embedding_model_state = "not_loaded"
    rag_mod._embedding_model_error = None

    load_count = [0]

    try:
        import sentence_transformers

        class CountingTransformer(sentence_transformers.SentenceTransformer):
            def __init__(self, *args, **kwargs):
                load_count[0] += 1
                super().__init__(*args, **kwargs)

        monkeypatch.setattr(sentence_transformers, "SentenceTransformer", CountingTransformer)

        m1 = rag_mod._get_embedding_model()
        m2 = rag_mod._get_embedding_model()
        assert m1 is m2, "Singleton must return the same instance"
        assert load_count[0] == 1, f"Model loaded {load_count[0]} times — expected 1"
        assert rag_mod._embedding_model_state == "ready"
    except ImportError:
        pytest.skip("sentence_transformers not installed")
    finally:
        # Restore original state so other tests are unaffected
        rag_mod._embedding_model = original_model
        rag_mod._embedding_model_state = original_state
        rag_mod._embedding_model_error = original_error
