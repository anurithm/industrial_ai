"""
RAG-based Maintenance Assistant using Chroma + Sentence Transformers + LLM.
Supports OpenRouter and Ollama as LLM providers.

Memory-efficiency notes (Render free tier):
- SentenceTransformer is loaded lazily on first use and cached as a singleton.
  It is NOT imported or loaded at module import time or during API startup.
- The model is PRE-DOWNLOADED in the Dockerfile so no network call occurs at runtime.
- Chroma client is created lazily per call (stateless PersistentClient).
- Use rag_ready() to check availability without loading anything heavy.
- Use rag_status() for a full readiness report (model state + store state).
"""
import logging
import os
import time
from pathlib import Path
from threading import Lock, Event
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

from src.config import (
    DOCS_DIR, CHROMA_DIR, EMBEDDING_MODEL, CHROMA_COLLECTION,
    RAG_N_RESULTS, RAG_CHUNK_SIZE, RAG_CHUNK_OVERLAP,
    LLM_PROVIDER, OPENROUTER_API_KEY, OPENROUTER_MODEL, OPENROUTER_BASE_URL,
    OLLAMA_BASE_URL, OLLAMA_MODEL,
)

# LLM call timeouts (seconds)
OPENROUTER_CONNECT_TIMEOUT = 10
OPENROUTER_READ_TIMEOUT = int(os.getenv("OPENROUTER_TIMEOUT", "30"))

# ── Lazy singleton for the embedding model ────────────────────────────────────
# The model is pre-downloaded into the Docker image at /app/.sentence_transformers.
# Loading it into RAM is fast (~2-5s). Protected by a lock so concurrent callers
# wait for the first load rather than triggering duplicate loads.
_embedding_model = None
_embedding_model_lock = Lock()
_embedding_model_state = "not_loaded"  # not_loaded | loading | ready | error
_embedding_model_error: Optional[str] = None


def _get_embedding_model():
    """
    Return the cached SentenceTransformer, loading it once on first call.
    Thread-safe singleton — concurrent callers block until load completes.
    State is tracked so rag_status() can report readiness without loading.
    """
    global _embedding_model, _embedding_model_state, _embedding_model_error

    if _embedding_model_state == "ready":
        return _embedding_model

    with _embedding_model_lock:
        # Double-check after acquiring lock
        if _embedding_model_state == "ready":
            return _embedding_model

        _embedding_model_state = "loading"
        t0 = time.perf_counter()
        logger.info(f"[RAG] Loading embedding model '{EMBEDDING_MODEL}' ...")
        try:
            from sentence_transformers import SentenceTransformer
            _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
            elapsed = time.perf_counter() - t0
            _embedding_model_state = "ready"
            logger.info(f"[RAG] Embedding model loaded in {elapsed:.1f}s (dim={_embedding_model.get_embedding_dimension()})")
        except Exception as e:
            _embedding_model_state = "error"
            _embedding_model_error = str(e)
            logger.error(f"[RAG] Embedding model load FAILED in {time.perf_counter()-t0:.1f}s: {e}")
            raise

    return _embedding_model


def rag_ready() -> bool:
    """
    Lightweight check: does the Chroma vector store exist with data?
    Does NOT load any models. Safe to call from /health at startup.
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
    Does NOT load the model or expose any secrets.
    """
    store_ready = rag_ready()
    try:
        if store_ready:
            import chromadb
            client = chromadb.PersistentClient(path=str(CHROMA_DIR))
            col = client.get_collection(CHROMA_COLLECTION)
            chunk_count = col.count()
        else:
            chunk_count = 0
    except Exception:
        chunk_count = 0

    return {
        "vector_store_ready": store_ready,
        "chunk_count": chunk_count,
        "embedding_model_state": _embedding_model_state,  # not_loaded|loading|ready|error
        "embedding_model_error": _embedding_model_error if _embedding_model_state == "error" else None,
        "llm_provider": LLM_PROVIDER,
        "llm_configured": bool(OPENROUTER_API_KEY) if LLM_PROVIDER == "openrouter" else (LLM_PROVIDER == "ollama"),
    }


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
    needs to build it at runtime.
    """
    import chromadb
    t0 = time.perf_counter()

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    existing = [c.name for c in client.list_collections()]

    if CHROMA_COLLECTION in existing and not force_rebuild:
        logger.info(f"[RAG] Collection '{CHROMA_COLLECTION}' already exists. Skipping rebuild.")
        return client.get_collection(CHROMA_COLLECTION)

    if CHROMA_COLLECTION in existing:
        client.delete_collection(CHROMA_COLLECTION)

    collection = client.create_collection(CHROMA_COLLECTION)
    embedding_model = _get_embedding_model()

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

    logger.info(f"[RAG] Vector store built: {collection.count()} chunks in {time.perf_counter()-t0:.1f}s")
    return collection


def retrieve(query: str, n_results: int = RAG_N_RESULTS) -> List[Dict]:
    """
    Retrieve relevant document chunks for a query.
    Embedding model is a singleton — loaded once, then cached.

    Structured timing logs emitted at each stage for debugging.
    Returns empty list if vector store is unavailable.
    """
    import chromadb

    t0 = time.perf_counter()
    logger.info(f"[RAG] retrieve() start — query length: {len(query)} chars")

    # Stage 1: ChromaDB client + collection
    t_db = time.perf_counter()
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        collection = client.get_collection(CHROMA_COLLECTION)
    except Exception:
        logger.warning(f"[RAG] Vector store not found ({time.perf_counter()-t_db:.2f}s) — returning empty.")
        return []

    if collection.count() == 0:
        logger.warning("[RAG] Vector store is empty — returning empty results.")
        return []
    logger.info(f"[RAG] ChromaDB ready ({time.perf_counter()-t_db:.2f}s), {collection.count()} chunks")

    # Stage 2: Embedding model load (singleton — fast after first call)
    t_model = time.perf_counter()
    embedding_model = _get_embedding_model()
    logger.info(f"[RAG] Embedding model ready ({time.perf_counter()-t_model:.2f}s)")

    # Stage 3: Encode query
    t_enc = time.perf_counter()
    query_embedding = embedding_model.encode(query).tolist()
    logger.info(f"[RAG] Query encoded ({time.perf_counter()-t_enc:.2f}s)")

    # Stage 4: Vector search
    t_search = time.perf_counter()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(n_results, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    logger.info(f"[RAG] Vector search done ({time.perf_counter()-t_search:.2f}s), {len(results['documents'][0])} results")

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
    """Call the configured LLM provider."""
    if LLM_PROVIDER == "openrouter":
        return _call_openrouter(prompt)
    elif LLM_PROVIDER == "ollama":
        return _call_ollama(prompt)
    else:
        logger.error(f"[RAG] Unknown LLM_PROVIDER: {LLM_PROVIDER}")
        return None


def _call_openrouter(prompt: str) -> Optional[str]:
    """
    Call OpenRouter API with explicit connect + read timeouts.
    Returns None (graceful degradation) on any failure.
    Never logs the API key.
    """
    if not OPENROUTER_API_KEY:
        logger.info("[RAG] OPENROUTER_API_KEY not set — LLM unavailable, returning passages only.")
        return None

    t0 = time.perf_counter()
    logger.info(f"[RAG] OpenRouter request starting (model={OPENROUTER_MODEL}, read_timeout={OPENROUTER_READ_TIMEOUT}s)")
    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
            timeout=OPENROUTER_READ_TIMEOUT,  # applies to both connect and read
        )
        response = client.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0.3,
        )
        answer = response.choices[0].message.content
        logger.info(f"[RAG] OpenRouter responded in {time.perf_counter()-t0:.2f}s ({len(answer)} chars)")
        return answer
    except Exception as e:
        logger.error(f"[RAG] OpenRouter error after {time.perf_counter()-t0:.2f}s: {type(e).__name__}: {e}")
        return None


def _call_ollama(prompt: str) -> Optional[str]:
    """Call Ollama local API with a 60s timeout."""
    t0 = time.perf_counter()
    logger.info(f"[RAG] Ollama request starting (model={OLLAMA_MODEL})")
    try:
        import httpx
        response = httpx.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=60.0,
        )
        response.raise_for_status()
        answer = response.json().get("response", "")
        logger.info(f"[RAG] Ollama responded in {time.perf_counter()-t0:.2f}s")
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
    Answer a maintenance question using RAG.

    Timing logs emitted at each stage:
      [RAG] ask() start
      [RAG] retrieve() start / done
      [RAG] OpenRouter request / response
      [RAG] ask() total: Xs

    Gracefully degrades if vector store or LLM is unavailable.
    """
    t0 = time.perf_counter()
    logger.info(f"[RAG] ask() start — question length: {len(question)} chars")

    # Stage 1: Retrieve evidence
    try:
        passages = retrieve(question)
    except Exception as e:
        logger.error(f"[RAG] retrieve() failed: {type(e).__name__}: {e}")
        passages = []

    t_retrieved = time.perf_counter()
    logger.info(f"[RAG] Retrieved {len(passages)} passages in {t_retrieved-t0:.2f}s")

    evidence_text = "\n\n---\n\n".join(
        [f"[Source: {p['source']}, chunk {p['chunk']}]\n{p['text']}" for p in passages]
    )

    # Stage 2: Build context blocks
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

    # Stage 3: LLM call
    t_llm = time.perf_counter()
    llm_answer = _call_llm(prompt)
    logger.info(f"[RAG] LLM stage done in {time.perf_counter()-t_llm:.2f}s, available={llm_answer is not None}")

    latency = (time.perf_counter() - t0) * 1000
    logger.info(f"[RAG] ask() total: {latency/1000:.2f}s")

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
