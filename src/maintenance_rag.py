"""
RAG-based Maintenance Assistant using Chroma + Sentence Transformers + LLM.
Supports OpenRouter and Ollama as LLM providers.

Memory-efficiency notes (Render free tier):
- SentenceTransformer is loaded lazily on first use and cached as a singleton.
  It is NOT imported or loaded at module import time or during API startup.
- Chroma client is also created lazily per call (stateless PersistentClient).
- Use rag_ready() to check availability without loading anything heavy.
"""
import logging
import time
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

from src.config import (
    DOCS_DIR, CHROMA_DIR, EMBEDDING_MODEL, CHROMA_COLLECTION,
    RAG_N_RESULTS, RAG_CHUNK_SIZE, RAG_CHUNK_OVERLAP,
    LLM_PROVIDER, OPENROUTER_API_KEY, OPENROUTER_MODEL, OPENROUTER_BASE_URL,
    OLLAMA_BASE_URL, OLLAMA_MODEL,
)

# ── Lazy singleton for the embedding model ────────────────────────────────────
# Loading SentenceTransformer uses ~300 MB RAM. We load it once on demand and
# cache it for the lifetime of the process — never at module import time.
_embedding_model = None
_embedding_model_lock = Lock()


def _get_embedding_model():
    """Return the cached SentenceTransformer, loading it once on first call."""
    global _embedding_model
    if _embedding_model is None:
        with _embedding_model_lock:
            if _embedding_model is None:
                logger.info(f"[RAG] Loading embedding model '{EMBEDDING_MODEL}' (first use) ...")
                from sentence_transformers import SentenceTransformer
                _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
                logger.info("[RAG] Embedding model loaded and cached.")
    return _embedding_model


def rag_ready() -> bool:
    """
    Check if the Chroma vector store exists and has data — WITHOUT loading
    any heavy models. Safe to call from /health at startup.
    """
    db_path = CHROMA_DIR / "chroma.sqlite3"
    if not db_path.exists():
        return False
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        existing = [c.name for c in client.list_collections()]
        if CHROMA_COLLECTION not in existing:
            return False
        col = client.get_collection(CHROMA_COLLECTION)
        return col.count() > 0
    except Exception as e:
        logger.debug(f"[RAG] rag_ready() check failed: {e}")
        return False


# ── Text utilities ────────────────────────────────────────────────────────────

def _chunk_text(text: str, chunk_size: int = RAG_CHUNK_SIZE, overlap: int = RAG_CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


def _load_document(path: Path) -> str:
    """Load text from TXT, MD, or PDF file."""
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="replace")
    elif suffix == ".pdf":
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(path))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as e:
            logger.warning(f"PDF read error for {path}: {e}")
            return ""
    return ""


# ── Vector store ──────────────────────────────────────────────────────────────

def build_vector_store(force_rebuild: bool = False):
    """
    Ingest maintenance documents into Chroma vector database.
    Idempotent: skips if collection exists and force_rebuild is False.

    Run this locally then commit chroma_db/ to Git so Render never
    needs to build it at runtime (saves ~300 MB startup RAM).
    """
    import chromadb

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    existing = [c.name for c in client.list_collections()]

    if CHROMA_COLLECTION in existing and not force_rebuild:
        logger.info(f"[RAG] Collection '{CHROMA_COLLECTION}' already exists. Skipping rebuild.")
        return client.get_collection(CHROMA_COLLECTION)

    if CHROMA_COLLECTION in existing:
        client.delete_collection(CHROMA_COLLECTION)

    collection = client.create_collection(CHROMA_COLLECTION)
    embedding_model = _get_embedding_model()  # Uses the singleton

    doc_files = (
        list(DOCS_DIR.glob("*.txt"))
        + list(DOCS_DIR.glob("*.md"))
        + list(DOCS_DIR.glob("*.pdf"))
    )
    if not doc_files:
        logger.warning(f"[RAG] No documents found in {DOCS_DIR}")
        return collection

    for doc_path in doc_files:
        text = _load_document(doc_path)
        if not text.strip():
            continue
        chunks = _chunk_text(text)
        logger.info(f"[RAG] Indexing {doc_path.name}: {len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            embedding = embedding_model.encode(chunk).tolist()
            collection.add(
                ids=[f"{doc_path.stem}_chunk_{i}"],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{"source": doc_path.name, "chunk": i}],
            )

    logger.info(f"[RAG] Vector store built: {collection.count()} chunks indexed")
    return collection


def retrieve(query: str, n_results: int = RAG_N_RESULTS) -> List[Dict]:
    """
    Retrieve relevant document chunks for a query.
    Loads the embedding model lazily (singleton — loaded once, cached).

    Returns:
        List of passages with text, source, and relevance score.
        Returns empty list if vector store is unavailable.
    """
    import chromadb

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        collection = client.get_collection(CHROMA_COLLECTION)
    except Exception:
        logger.warning("[RAG] Vector store collection not found — returning empty results.")
        return []

    if collection.count() == 0:
        logger.warning("[RAG] Vector store is empty — returning empty results.")
        return []

    embedding_model = _get_embedding_model()  # Singleton — no reload on every call
    query_embedding = embedding_model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(n_results, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    passages = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        passages.append({
            "text": doc,
            "source": meta.get("source", "unknown"),
            "chunk": meta.get("chunk", 0),
            "distance": round(float(dist), 4),
        })
    return passages


# ── LLM calls ─────────────────────────────────────────────────────────────────

def _call_llm(prompt: str) -> Optional[str]:
    """Call the configured LLM provider."""
    if LLM_PROVIDER == "openrouter":
        return _call_openrouter(prompt)
    elif LLM_PROVIDER == "ollama":
        return _call_ollama(prompt)
    else:
        logger.error(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}")
        return None


def _call_openrouter(prompt: str) -> Optional[str]:
    """Call OpenRouter API."""
    if not OPENROUTER_API_KEY:
        logger.info("[RAG] OPENROUTER_API_KEY not set — LLM unavailable.")
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)
        response = client.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"OpenRouter API error: {e}")
        return None


def _call_ollama(prompt: str) -> Optional[str]:
    """Call Ollama local API."""
    try:
        import httpx
        response = httpx.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=60.0,
        )
        response.raise_for_status()
        return response.json().get("response", "")
    except Exception as e:
        logger.error(f"Ollama API error: {e}")
        return None


# ── Main ask function ─────────────────────────────────────────────────────────

def ask(
    question: str,
    prediction_context: Optional[Dict] = None,
    sensor_context: Optional[Dict] = None,
) -> Dict:
    """
    Answer a maintenance question using RAG.

    Loads embedding model and Chroma lazily on first call.
    Gracefully degrades if vector store or LLM is unavailable.
    """
    start = time.perf_counter()

    # Retrieve evidence (lazy-loads embedding model on first call)
    try:
        passages = retrieve(question)
    except Exception as e:
        logger.error(f"[RAG] retrieve() failed: {e}")
        passages = []

    evidence_text = "\n\n---\n\n".join(
        [f"[Source: {p['source']}, chunk {p['chunk']}]\n{p['text']}" for p in passages]
    )

    # Build context blocks
    context_blocks = []
    if prediction_context:
        pred = prediction_context
        context_blocks.append(
            f"Current ML Prediction:\n"
            f"  - Predicted failure: {pred.get('predicted_failure', 'unknown')}\n"
            f"  - Failure probability: {pred.get('failure_probability', 'N/A')}\n"
            f"  - Risk level: {pred.get('risk_level', 'N/A')}\n"
            f"  - Model: {pred.get('model_name', 'N/A')}\n"
            f"  - Disclaimer: {pred.get('disclaimer', '')}"
        )
    if sensor_context:
        sensor_str = "\n".join([f"  - {k}: {v}" for k, v in sensor_context.items()])
        context_blocks.append(f"Sensor Readings:\n{sensor_str}")

    ml_context = "\n\n".join(context_blocks)

    prompt = f"""You are an industrial maintenance assistant. Answer the user's question using ONLY the provided maintenance documentation and ML prediction context.

MAINTENANCE DOCUMENTATION (retrieved passages):
{evidence_text if evidence_text else "No relevant documentation found."}

{ml_context if ml_context else ""}

IMPORTANT RULES:
1. Base your answer ONLY on the retrieved documentation and ML context above.
2. If the documentation doesn't contain enough information, say so clearly.
3. Cite the source document for each factual claim (e.g., [Source: filename.txt]).
4. Clearly distinguish between documented facts and model-based estimates.
5. Do NOT fabricate maintenance procedures, part numbers, or technical specifications.
6. The ML prediction is probabilistic — do NOT state it as a certainty.

USER QUESTION: {question}

ANSWER:"""

    llm_answer = _call_llm(prompt)
    latency = (time.perf_counter() - start) * 1000

    if llm_answer is None:
        if passages:
            fallback_answer = (
                "LLM service is currently unavailable. "
                "Here are the most relevant documentation passages retrieved:\n\n"
                + "\n\n".join([f"**[{p['source']}]**: {p['text'][:300]}..." for p in passages[:3]])
            )
        else:
            fallback_answer = (
                "LLM service is unavailable and no relevant documentation was found. "
                "Please check your LLM configuration (OPENROUTER_API_KEY or Ollama service) "
                "and ensure maintenance documents are indexed."
            )
        return {
            "answer": fallback_answer,
            "sources": [p["source"] for p in passages],
            "passages": passages,
            "llm_available": False,
            "latency_ms": round(latency, 1),
            "question": question,
        }

    citations = list({p["source"] for p in passages if p["source"] in llm_answer or len(passages) > 0})

    return {
        "answer": llm_answer,
        "sources": citations,
        "passages": passages,
        "llm_available": True,
        "latency_ms": round(latency, 1),
        "question": question,
    }
