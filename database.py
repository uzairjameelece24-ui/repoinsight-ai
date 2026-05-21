"""
database.py — Thread-safe ChromaDB singleton initialization.

Provides a single global ChromaDB ``PersistentClient`` and a single
``codebase_intelligence`` collection that all services share.

Design
------
* The client is created once per process via module-level initialization
  guarded by a threading.Lock, making it safe under Uvicorn's default
  multi-threaded request handling.
* The collection uses cosine distance because OpenAI embeddings are
  normalized unit vectors; cosine similarity is the canonical metric.
* Helper functions ``upsert_chunks`` and ``delete_repo_vectors`` are exposed
  so that services never import chromadb directly.
"""

import logging
import threading
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from config import get_settings

logger = logging.getLogger(__name__)

# ── Module-level singletons ──────────────────────────────────────────────────

_client_lock: threading.RLock = threading.RLock()
_chroma_client: Optional[chromadb.PersistentClient] = None
_collection: Optional[chromadb.Collection] = None

COLLECTION_NAME = "codebase_intelligence"


def _initialize_client() -> chromadb.PersistentClient:
    """
    Create (or return the cached) ChromaDB ``PersistentClient``.

    The client is initialised with anonymized_telemetry disabled for
    production hygiene. The persist directory is taken from ``Settings``.

    Returns
    -------
    chromadb.PersistentClient
        Ready-to-use ChromaDB client pointing to the configured directory.
    """
    settings = get_settings()
    client = chromadb.PersistentClient(
        path=settings.CHROMA_DB_DIR,
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    logger.info("ChromaDB PersistentClient initialised at '%s'.", settings.CHROMA_DB_DIR)
    return client


def get_client() -> chromadb.PersistentClient:
    """
    Return the process-wide ChromaDB client, initialising it on first call.

    Thread-safe via ``_client_lock``.

    Returns
    -------
    chromadb.PersistentClient
    """
    global _chroma_client
    if _chroma_client is None:
        with _client_lock:
            # Double-checked locking pattern
            if _chroma_client is None:
                _chroma_client = _initialize_client()
    return _chroma_client


def get_collection() -> chromadb.Collection:
    """
    Return the shared ``codebase_intelligence`` collection.

    The collection is created with ``get_or_create_collection`` so repeated
    application restarts are idempotent — existing vectors survive.

    The embedding function is intentionally *not* attached here; embeddings are
    pre-computed by the RAG service via the OpenAI API before being passed to
    ChromaDB. This avoids any dependency on a ChromaDB-bundled embedding model.

    Returns
    -------
    chromadb.Collection
        The global ``codebase_intelligence`` collection.
    """
    global _collection
    if _collection is None:
        with _client_lock:
            if _collection is None:
                client = get_client()
                _collection = client.get_or_create_collection(
                    name=COLLECTION_NAME,
                    metadata={"hnsw:space": "cosine"},
                )
                logger.info(
                    "ChromaDB collection '%s' ready (%d existing vectors).",
                    COLLECTION_NAME,
                    _collection.count(),
                )
    return _collection


# ── Public helpers ───────────────────────────────────────────────────────────


def delete_repo_vectors(repo_id: str) -> int:
    """
    Delete all vectors associated with ``repo_id`` from the collection.

    This is called at the start of every ingestion cycle (delta-indexing) and
    from the ``DELETE /api/repo/{repo_id}`` endpoint.

    Parameters
    ----------
    repo_id : str
        The repository namespace identifier used as a ChromaDB ``where`` filter.

    Returns
    -------
    int
        Number of vectors deleted (approximate — ChromaDB does not always
        return exact counts from ``delete``; we derive it by comparing
        collection size before and after).
    """
    collection = get_collection()
    before = collection.count()
    collection.delete(where={"repo_id": repo_id})
    after = collection.count()
    deleted = max(0, before - after)
    logger.info(
        "Deleted ~%d vectors for repo_id='%s'. Collection size: %d → %d.",
        deleted,
        repo_id,
        before,
        after,
    )
    return deleted


def upsert_chunks(
    ids: List[str],
    embeddings: List[List[float]],
    documents: List[str],
    metadatas: List[Dict[str, Any]],
) -> None:
    """
    Batch-upsert pre-computed embeddings into the ChromaDB collection.

    Splits the input into batches of ``BATCH_SIZE`` (from ``Settings``) to
    avoid hitting payload limits on large repositories.

    Parameters
    ----------
    ids : List[str]
        Unique string identifiers for each chunk (must be globally unique).
    embeddings : List[List[float]]
        Pre-computed embedding vectors, one per chunk.
    documents : List[str]
        Raw text of each chunk (stored verbatim in ChromaDB for retrieval).
    metadatas : List[Dict[str, Any]]
        Metadata dicts conforming to the schema defined in ``services/parser.py``.

    Raises
    ------
    ValueError
        If the lengths of the four input lists differ.
    """
    if not (len(ids) == len(embeddings) == len(documents) == len(metadatas)):
        raise ValueError(
            "ids, embeddings, documents, and metadatas must all have the same length."
        )

    if not ids:
        logger.warning("upsert_chunks called with empty input — nothing to write.")
        return

    batch_size = get_settings().BATCH_SIZE
    collection = get_collection()
    total = len(ids)
    batches = (total + batch_size - 1) // batch_size  # ceiling division

    for batch_idx in range(batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, total)
        collection.upsert(
            ids=ids[start:end],
            embeddings=embeddings[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )
        logger.debug(
            "Upserted batch %d/%d (%d vectors).",
            batch_idx + 1,
            batches,
            end - start,
        )

    logger.info("Upserted %d total vectors across %d batch(es).", total, batches)


def get_collection_stats() -> Dict[str, Any]:
    """
    Return basic statistics about the global collection.

    Returns
    -------
    dict
        Dictionary containing ``total_vectors`` and ``collection_name``.
    """
    collection = get_collection()
    return {
        "collection_name": COLLECTION_NAME,
        "total_vectors": collection.count(),
    }
