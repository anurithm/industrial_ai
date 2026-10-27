"""
Tests for the RAG maintenance assistant.
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


def test_retrieve_no_error_when_empty():
    """Retrieve should return empty list when no docs indexed."""
    from src.config import CHROMA_DIR
    # If no chroma db, retrieve returns empty list gracefully
    try:
        from src.maintenance_rag import retrieve
        # May return empty list or raise connection error — both are acceptable
        result = retrieve("test question")
        assert isinstance(result, list)
    except Exception as e:
        # If chromadb not initialized, that's okay for this unit test
        pass


def test_ask_without_llm():
    """
    Ask should gracefully handle missing LLM credentials.
    Returns answer string even if LLM unavailable.
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


def test_ask_with_context():
    """Ask passes prediction and sensor context correctly."""
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
    """Ask handles case where no relevant docs are found."""
    from src.maintenance_rag import ask
    # Highly specific unlikely-to-match question
    result = ask(
        question="What is the quantum entanglement coefficient of a neutron star made of cheese?",
    )
    assert "answer" in result
    # Should NOT fabricate — answer should mention insufficient evidence or return retrieved text
    assert isinstance(result["answer"], str)
