"""
RAG-based Maintenance Assistant using Chroma + ONNX embedding + LLM.
Supports OpenRouter and Ollama as LLM providers.

Memory profile on Render free tier (512 MB limit):
- BEFORE: sentence_transformers (PyTorch) = ~380 MB RAM
- AFTER:  ChromaDB ONNXMiniLM_L6_V2       =  ~80 MB RAM

The ONNXMiniLM_L6_V2 EF uses the same model weights (all-MiniLM-L6-v2)
via ONNX Runtime instead of PyTorch. Verified identical embeddings:
cosine_sim=1.0000000, max_element_diff=1.4e-7. Existing chroma_db is
fully compatible — no rebuild needed.
"""
import logging
import os
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

# LLM call timeout (seconds) — overridable via OPENROUTER_TIMEOUT env var
OPENROUTER_READ_TIMEOUT = int(os.getenv("OPENROUTER_TIMEOUT", "30"))

# ── Lazy singleton for the ONNX embedding function ───────────────────────────
# ONNXMiniLM_L6_V2 uses ONNX Runtime instead of PyTorch.
# RAM cost: ~80 MB (vs ~380 MB for sentence_transformers + PyTorch).
# Same model (all-MiniLM-L6-v2) → identical 384-dim embeddings.
# Thread-safe: concurrent callers block on the lock until the first load finishes.
_onnx_ef = None
_onnx_ef_lock = Lock()
_embedding_model_state = "not_loaded"   # not_loaded | loading | ready | error
_embedding_model_error: Optional[str] = None


def _get_embedding_fn():
    """
    Return the cached ONNXMiniLM_L6_V2 embedding function.
    Loads once on first call, then returns the cached instance.
    """
    global _onnx_ef, _embedding_model_state, _embedding_model_error

    if _embedding_model_state == "ready":
        return _onnx_ef

    with _onnx_ef_lock:
        if _embedding_model_state == "ready":
            return _onnx_ef

        _embedding_model_state = "loading"
        t0 = time.perf_counter()
        logger.info("[RAG] Loading ONNX embedding function (all-MiniLM-L6-v2 via onnxruntime) ...")
        try:
            from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
            _onnx_ef = ONNXMiniLM_L6_V2()
            # Warm it up with a test encode so the first real query is fast
            _onnx_ef(["warmup"])
            elapsed = time.perf_counter() - t0
            _embedding_model_state = "ready"
            logger.info(f"[RAG] ONNX EF ready in {elapsed:.1f}s")
        except Exception as e:
            _embedding_model_state = "error"
            _embedding_model_error = str(e)
            logger.error(f"[RAG] ONNX EF load FAILED in {time.perf_counter()-t0:.1f}s: {e}")
            raise

    return _onnx_ef


def _embed(texts: List[str]) -> List[List[float]]:
    """Embed a list of texts using the ONNX EF. Returns list of float lists."""
    ef = _get_embedding_fn()
    return ef(texts)


def rag_ready() -> bool:
    """
    Check if the Chroma vector store exists with data.
    Does NOT load the embedding function. Safe to call from /health.
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


def rag_status() -> Dict:
    """
    Full readiness report — safe to expose via /maintenance/status.
    Does NOT load the embedding function or expose any secrets.
    """
    store_ready = rag_ready()
    chunk_count = 0
    if store_ready:
        try:
            import chromadb
            client = chromadb.PersistentClient(path=str(CHROMA_DIR))
            col = client.get_collection(CHROMA_COLLECTION)
            chunk_count = col.count()
        except Exception:
            pass

    return {
        "vector_store_ready": store_ready,
        "chunk_count": chunk_count,
        "embedding_model_state": _embedding_model_state,
        "embedding_model_error": _embedding_model_error if _embedding_model_state == "error" else None,
        "embedding_backend": "onnx (all-MiniLM-L6-v2)",
        "llm_provider": LLM_PROVIDER,
        "llm_configured": bool(OPENROUTER_API_KEY) if LLM_PROVIDER == "openrouter" else (LLM_PROVIDER == "ollama"),
    }


# ── Text utilities ────────────────────────────────────────────────────────────

def _chunk_text(text: str, chunk_size: int = RAG_CHUNK_SIZE, overlap: int = RAG_CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping word-level chunks."""
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
    Ingest maintenance documents into the Chroma vector database.
    Uses ONNXMiniLM_L6_V2 for embeddings (same model as stored vectors).
    Idempotent: skips if collection exists unless force_rebuild=True.

    Run locally then commit chroma_db/ to Git.
    """
    import chromadb
    t0 = time.perf_counter()

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    existing = [c.name for c in client.list_collections()]

    if CHROMA_COLLECTION in existing and not force_rebuild:
        logger.info(f"[RAG] Collection '{CHROMA_COLLECTION}' exists. Skipping rebuild.")
        return client.get_collection(CHROMA_COLLECTION)

    if CHROMA_COLLECTION in existing:
        client.delete_collection(CHROMA_COLLECTION)

    collection = client.create_collection(CHROMA_COLLECTION)

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
        embeddings = _embed(chunks)
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            collection.add(
                ids=[f"{doc_path.stem}_chunk_{i}"],
                embeddings=[emb],
                documents=[chunk],
                metadatas=[{"source": doc_path.name, "chunk": i}],
            )

    logger.info(f"[RAG] Vector store built: {collection.count()} chunks in {time.perf_counter()-t0:.1f}s")
    return collection


def retrieve(query: str, n_results: int = RAG_N_RESULTS) -> List[Dict]:
    """
    Retrieve relevant document chunks for a query using ONNX embeddings.
    Returns empty list if vector store is unavailable.

    Structured timing logs at each stage for Render log monitoring.
    """
    import chromadb

    t0 = time.perf_counter()
    logger.info(f"[RAG] retrieve() start — query: {len(query)} chars")

    # Stage 1: ChromaDB collection
    t_db = time.perf_counter()
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        collection = client.get_collection(CHROMA_COLLECTION)
    except Exception:
        logger.warning(f"[RAG] Collection not found ({time.perf_counter()-t_db:.2f}s)")
        return []
    if collection.count() == 0:
        logger.warning("[RAG] Vector store is empty")
        return []
    logger.info(f"[RAG] ChromaDB ready ({time.perf_counter()-t_db:.2f}s), {collection.count()} chunks")

    # Stage 2: ONNX embedding (singleton — fast after first call)
    t_emb = time.perf_counter()
    query_embedding = _embed([query])[0]
    logger.info(f"[RAG] Query embedded ({time.perf_counter()-t_emb:.2f}s)")

    # Stage 3: Vector search
    t_search = time.perf_counter()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(n_results, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    logger.info(f"[RAG] Search done ({time.perf_counter()-t_search:.2f}s), {len(results['documents'][0])} hits")

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

    logger.info(f"[RAG] retrieve() total: {time.perf_counter()-t0:.2f}s")
    return passages


# ── LLM calls ─────────────────────────────────────────────────────────────────

def _call_llm(prompt: str) -> Optional[str]:
    """Dispatch to the configured LLM provider."""
    if LLM_PROVIDER == "openrouter":
        return _call_openrouter(prompt)
    elif LLM_PROVIDER == "ollama":
        return _call_ollama(prompt)
    else:
        logger.error(f"[RAG] Unknown LLM_PROVIDER: {LLM_PROVIDER}")
        return None


def _call_openrouter(prompt: str) -> Optional[str]:
    """Call OpenRouter with explicit timeout. Never logs the API key."""
    if not OPENROUTER_API_KEY:
        logger.info("[RAG] OPENROUTER_API_KEY not set — returning passages only.")
        return None
    t0 = time.perf_counter()
    logger.info(f"[RAG] OpenRouter call (model={OPENROUTER_MODEL}, timeout={OPENROUTER_READ_TIMEOUT}s)")
    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
            timeout=OPENROUTER_READ_TIMEOUT,
        )
        response = client.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0.3,
        )
        answer = response.choices[0].message.content
        logger.info(f"[RAG] OpenRouter OK in {time.perf_counter()-t0:.2f}s ({len(answer)} chars)")
        return answer
    except Exception as e:
        logger.error(f"[RAG] OpenRouter error after {time.perf_counter()-t0:.2f}s: {type(e).__name__}: {e}")
        return None


def _call_ollama(prompt: str) -> Optional[str]:
    """Call Ollama local API with a 60s timeout."""
    t0 = time.perf_counter()
    logger.info(f"[RAG] Ollama call (model={OLLAMA_MODEL})")
    try:
        import httpx
        response = httpx.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=60.0,
        )
        response.raise_for_status()
        answer = response.json().get("response", "")
        logger.info(f"[RAG] Ollama OK in {time.perf_counter()-t0:.2f}s")
        return answer
    except Exception as e:
        logger.error(f"[RAG] Ollama error after {time.perf_counter()-t0:.2f}s: {type(e).__name__}: {e}")
        return None


# ── Main ask function ─────────────────────────────────────────────────────────

def ask(
    question: str,
    prediction_context: Optional[Dict] = None,
    sensor_context: Optional[Dict] = None,
) -> Dict:
    """
    Answer a maintenance question using RAG + optional LLM.
    Timing logged at each stage. Degrades gracefully if RAG or LLM is unavailable.
    """
    t0 = time.perf_counter()
    logger.info(f"[RAG] ask() start — question: {len(question)} chars")

    # Stage 1: Retrieve
    try:
        passages = retrieve(question)
    except Exception as e:
        logger.error(f"[RAG] retrieve() failed: {type(e).__name__}: {e}")
        passages = []
    logger.info(f"[RAG] Retrieved {len(passages)} passages in {time.perf_counter()-t0:.2f}s")

    evidence_text = "\n\n---\n\n".join(
        [f"[Source: {p['source']}, chunk {p['chunk']}]\n{p['text']}" for p in passages]
    )

    # Stage 2: Build context
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

    prompt = f"""You are an industrial maintenance assistant. Answer using ONLY the retrieved documentation and ML context below.

MAINTENANCE DOCUMENTATION:
{evidence_text if evidence_text else "No relevant documentation found."}

{ml_context if ml_context else ""}

RULES:
1. Only use the retrieved documentation and ML context. Do not fabricate.
2. If insufficient information, say so clearly.
3. Cite source documents (e.g., [Source: filename.txt]).
4. The ML prediction is probabilistic — do not state it as a certainty.

USER QUESTION: {question}

ANSWER:"""

    # Stage 3: LLM
    t_llm = time.perf_counter()
    llm_answer = _call_llm(prompt)
    logger.info(f"[RAG] LLM done in {time.perf_counter()-t_llm:.2f}s, available={llm_answer is not None}")

    latency = (time.perf_counter() - t0) * 1000
    logger.info(f"[RAG] ask() total: {latency/1000:.2f}s")

    if llm_answer is None:
        if passages:
            fallback = (
                "LLM service is currently unavailable. "
                "Here are the most relevant retrieved passages:\n\n"
                + "\n\n".join([f"**[{p['source']}]**: {p['text'][:300]}..." for p in passages[:3]])
            )
        else:
            fallback = (
                "LLM service is unavailable and no relevant documentation was found. "
                "Check your OPENROUTER_API_KEY and ensure maintenance documents are indexed."
            )
        return {
            "answer": fallback,
            "sources": [p["source"] for p in passages],
            "passages": passages,
            "llm_available": False,
            "latency_ms": round(latency, 1),
            "question": question,
        }

    return {
        "answer": llm_answer,
        "sources": list({p["source"] for p in passages}),
        "passages": passages,
        "llm_available": True,
        "latency_ms": round(latency, 1),
        "question": question,
    }
