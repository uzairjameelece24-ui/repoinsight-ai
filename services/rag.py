"""
services/rag.py — Semantic vector search, prompt engineering, and LLM orchestration.

Responsibilities
----------------
1. Embed an incoming query string using a local SentenceTransformer model.
2. Execute a filtered nearest-neighbour search against ChromaDB.
3. Format a deterministic, hallucination-resistant system prompt that injects
   the retrieved code snippets and their metadata as context.
4. Call the Groq Chat Completions API and stream (or return) the response.
5. Return both the markdown answer and structured source references to the caller.

Design Notes
------------
* The Groq client and SentenceTransformer model are instantiated once and reused.
* The system prompt is defined as a module-level constant for easy auditing and
  modification without touching business logic.
* All ChromaDB result parsing is type-safe and handles missing metadata keys
  gracefully.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from groq import Groq
from sentence_transformers import SentenceTransformer

from config import get_settings
from database import get_collection
from schemas import SourceReference

logger = logging.getLogger(__name__)


# ── Clients (process-wide singletons) ──────────────────────────────────────────

_groq_client: Optional[Groq] = None
_embedding_model: Optional[SentenceTransformer] = None


def _get_groq_client() -> Groq:
    """
    Return the process-wide Groq client, creating it on first call.

    Returns
    -------
    Groq
        Configured Groq SDK client.
    """
    global _groq_client
    if _groq_client is None:
        settings = get_settings()
        _groq_client = Groq(api_key=settings.GROQ_API_KEY)
        logger.info("Groq client initialised.")
    return _groq_client


def _get_embedding_model() -> SentenceTransformer:
    """
    Return the process-wide SentenceTransformer model, creating it on first call.

    Returns
    -------
    SentenceTransformer
        Loaded local embedding model.
    """
    global _embedding_model
    if _embedding_model is None:
        settings = get_settings()
        logger.info("Loading local embedding model '%s'...", settings.EMBEDDING_MODEL)
        _embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
        logger.info("Local embedding model loaded.")
    return _embedding_model


# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT_TEMPLATE = """\
You are an expert codebase intelligence assistant. Your goal is to answer the \
user's question about the repository using ONLY the verified source code \
snippets provided below.

Strict Rules:
1. If the retrieved code snippets do not contain enough context to confidently \
answer the question, state explicitly that you do not have sufficient \
information. Do not invent behaviour, function signatures, or architectural \
claims that are not directly evidenced by the provided snippets.
2. For every claim, architectural observation, or code pattern you describe, \
you MUST cite the source file path and line numbers inline using the format \
[file_path, Lines line_start–line_end]. For example: \
[src/auth.py, Lines 20–45].
3. Maintain technical accuracy. Focus on execution paths, data flow, \
dependencies, and code patterns as they actually appear in the source.
4. Format your response using clean Markdown with appropriate code blocks, \
headers, and bullet points for readability.
5. Never reproduce more than 20 consecutive lines of code verbatim; summarise \
longer blocks and cite them by reference.

Retrieved Context:
{context}
"""


# ── Embedding ─────────────────────────────────────────────────────────────────


def embed_query(query: str) -> List[float]:
    """
    Convert a query string into an embedding vector using the local model.

    Parameters
    ----------
    query : str
        The natural-language or technical question to embed.

    Returns
    -------
    List[float]
        Dense embedding vector.
    """
    model = _get_embedding_model()
    # encode() returns a numpy array by default
    vector = model.encode(query, show_progress_bar=False).tolist()
    logger.debug("Embedded query (%d dims).", len(vector))
    return vector


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Batch-embed a list of text strings using the local model.

    Parameters
    ----------
    texts : List[str]
        List of text strings to embed (e.g., code chunk contents).

    Returns
    -------
    List[List[float]]
        List of embedding vectors in the same order as ``texts``.

    Raises
    ------
    ValueError
        If ``texts`` is empty.
    """
    if not texts:
        raise ValueError("embed_texts received an empty list of texts.")

    model = _get_embedding_model()
    vectors = model.encode(texts, show_progress_bar=False).tolist()
    logger.debug("Batch-embedded %d texts.", len(vectors))
    return vectors


# ── Vector search ──────────────────────────────────────────────────────────────


def search_similar_chunks(
    query_vector: List[float], repo_id: str
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Query ChromaDB for the top-k most semantically similar chunks.

    The search is filtered to ``repo_id`` so only vectors belonging to that
    repository namespace are considered.

    Parameters
    ----------
    query_vector : List[float]
        The embedded query vector.
    repo_id : str
        Repository namespace to filter results by.

    Returns
    -------
    Tuple[List[str], List[Dict[str, Any]]]
        A 2-tuple of (documents, metadatas) where both lists are ordered by
        descending similarity (closest first).

    Raises
    ------
    RuntimeError
        If ChromaDB returns no results or the collection is empty for ``repo_id``.
    """
    settings = get_settings()
    collection = get_collection()

    try:
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=settings.TOP_K_RESULTS,
            where={"repo_id": repo_id},
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:
        raise RuntimeError(
            f"ChromaDB query failed for repo_id='{repo_id}': {exc}"
        ) from exc

    documents: List[str] = results.get("documents", [[]])[0]
    metadatas: List[Dict[str, Any]] = results.get("metadatas", [[]])[0]
    distances: List[float] = results.get("distances", [[]])[0]

    if not documents:
        raise RuntimeError(
            f"No vectors found for repo_id='{repo_id}'. "
            "Ensure the repository has been ingested via POST /api/ingest."
        )

    logger.debug(
        "Retrieved %d chunk(s) for repo_id='%s'. "
        "Closest distance: %.4f.",
        len(documents),
        repo_id,
        distances[0] if distances else float("nan"),
    )
    return documents, metadatas


# ── Prompt construction ────────────────────────────────────────────────────────


def _build_context_block(
    documents: List[str], metadatas: List[Dict[str, Any]]
) -> str:
    """
    Format the retrieved chunks into a structured context block for the prompt.

    Each snippet is prefixed with its file path and line range so the LLM
    can cite them accurately.

    Parameters
    ----------
    documents : List[str]
        Raw text content of each retrieved chunk.
    metadatas : List[Dict[str, Any]]
        Corresponding metadata dicts from ChromaDB.

    Returns
    -------
    str
        Multi-line string ready to be interpolated into the system prompt.
    """
    parts: List[str] = []
    for idx, (doc, meta) in enumerate(zip(documents, metadatas), start=1):
        file_path = meta.get("file_path", "unknown")
        line_start = meta.get("line_start", "?")
        line_end = meta.get("line_end", "?")
        language = meta.get("language", "text")

        header = (
            f"--- Snippet {idx} ---\n"
            f"File: {file_path}\n"
            f"Lines: {line_start}–{line_end}\n"
            f"Language: {language}\n"
        )
        body = f"```{language}\n{doc}\n```"
        parts.append(f"{header}\n{body}")

    return "\n\n".join(parts)


def _build_source_references(
    documents: List[str], metadatas: List[Dict[str, Any]]
) -> List[SourceReference]:
    """
    Convert raw ChromaDB metadata dicts and documents into ``SourceReference`` Pydantic models.

    Parameters
    ----------
    documents : List[str]
        Raw text contents of retrieved code chunks.
    metadatas : List[Dict[str, Any]]
        Metadata dicts from ChromaDB query results.

    Returns
    -------
    List[SourceReference]
        Validated source reference objects.
    """
    refs: List[SourceReference] = []
    for doc, meta in zip(documents, metadatas):
        try:
            ref = SourceReference(
                file_path=str(meta.get("file_path", "unknown")),
                line_start=int(meta.get("line_start", 1)),
                line_end=int(meta.get("line_end", 1)),
                language=str(meta.get("language", "text")),
                content=doc,
            )
            refs.append(ref)
        except Exception as exc:
            logger.warning("Skipping malformed metadata entry: %s — %s", meta, exc)
    return refs


# ── LLM orchestration ─────────────────────────────────────────────────────────


def generate_answer(
    query: str,
    documents: List[str],
    metadatas: List[Dict[str, Any]],
) -> str:
    """
    Call the Groq Chat Completions API with the retrieved context.

    The system prompt embeds the formatted code snippets so the model can cite
    them accurately. The user message is the raw query string.

    Parameters
    ----------
    query : str
        The user's original question.
    documents : List[str]
        Retrieved code chunk texts.
    metadatas : List[Dict[str, Any]]
        Corresponding metadata for each chunk.

    Returns
    -------
    str
        Markdown-formatted answer from the LLM.

    Raises
    ------
    groq.GroqError
        On API errors.
    """
    settings = get_settings()
    client = _get_groq_client()

    context_block = _build_context_block(documents, metadatas)
    system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(context=context_block)

    logger.info(
        "Sending query to '%s'. System prompt: %d chars, context: %d chars.",
        settings.CHAT_MODEL,
        len(system_prompt),
        len(context_block),
    )

    response = client.chat.completions.create(
        model=settings.CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
        temperature=0.1,   # Low temperature → more deterministic, less hallucination
        max_tokens=2048,
    )

    answer = response.choices[0].message.content or ""
    logger.debug(
        "LLM response received: %d chars, finish_reason='%s'.",
        len(answer),
        response.choices[0].finish_reason,
    )
    return answer


# ── Public orchestration entry-point ──────────────────────────────────────────


def run_rag_pipeline(
    repo_id: str, query: str
) -> Tuple[str, List[SourceReference]]:
    """
    Execute the full RAG pipeline for a given query against a repository.

    Steps
    -----
    1. Embed the query string.
    2. Retrieve the top-k most similar code chunks from ChromaDB, filtered by
       ``repo_id``.
    3. Build a structured context block and inject it into the system prompt.
    4. Call the Groq Chat Completions API.
    5. Parse and return the answer and source references.

    Parameters
    ----------
    repo_id : str
        Repository namespace to query.
    query : str
        The user's natural-language or technical question.

    Returns
    -------
    Tuple[str, List[SourceReference]]
        A 2-tuple of (markdown_answer, source_references).

    Raises
    ------
    RuntimeError
        If no vectors are found for ``repo_id`` in ChromaDB.
    groq.GroqError
        On any Groq API error during chat completion.
    """
    logger.info("RAG pipeline started for repo_id='%s'.", repo_id)

    # Step 1: Embed the query
    query_vector = embed_query(query)

    # Step 2: Vector search
    documents, metadatas = search_similar_chunks(query_vector, repo_id)

    # Step 3 & 4: Build prompt and call LLM
    answer = generate_answer(query, documents, metadatas)

    # Step 5: Build source references
    sources = _build_source_references(documents, metadatas)

    logger.info(
        "RAG pipeline complete for repo_id='%s'. "
        "Answer: %d chars, sources: %d.",
        repo_id,
        len(answer),
        len(sources),
    )
    return answer, sources
