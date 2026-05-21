"""
main.py — FastAPI application initialisation, middleware, and routing entrypoint.

Endpoints
---------
POST   /api/ingest          Clone, parse, embed, and index a Git repository.
POST   /api/query           Run the RAG pipeline against an indexed repository.
DELETE /api/repo/{repo_id}  Remove all vectors and the cloned workspace.
GET    /                    Health check and API metadata.
GET    /api/stats           Collection statistics.

Design
------
* Blocking I/O (git clone, file reads, ChromaDB writes) is offloaded to a
  thread-pool executor so the asyncio event loop remains unblocked.
* All errors are caught and returned as structured JSON with appropriate HTTP
  status codes; unhandled exceptions surface as 500 responses.
* CORS is enabled for all origins in development. Tighten ``allow_origins`` for
  production deployments.
* Structured JSON logging is configured at startup.
"""

import asyncio
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, List

import uvicorn
from fastapi import FastAPI, HTTPException, Path, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from config import get_settings
from database import delete_repo_vectors, get_collection_stats, upsert_chunks
from schemas import (
    DeleteResponse,
    ErrorDetail,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)
from services.ingestion import (
    clone_repository,
    delete_repository_workspace,
    resolve_repo_id,
    walk_repository,
)
from services.parser import parse_repository
from services.rag import embed_texts, run_rag_pipeline

# ── Logging configuration ─────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    FastAPI lifespan context manager.

    Validates the configuration and warms up the ChromaDB client and OpenAI
    connection pool at startup so the first request doesn't bear cold-start
    latency.
    """
    logger.info("=== RepoInsight AI starting up ===")
    try:
        settings = get_settings()
        logger.info(
            "Configuration validated. ChromaDB dir: '%s', Repos dir: '%s'.",
            settings.CHROMA_DB_DIR,
            settings.REPOS_STORAGE_DIR,
        )
        # Warm up ChromaDB collection
        stats = get_collection_stats()
        logger.info(
            "ChromaDB collection '%s' ready — %d existing vectors.",
            stats["collection_name"],
            stats["total_vectors"],
        )
    except Exception as exc:
        logger.critical("Startup failed: %s", exc, exc_info=True)
        raise

    yield  # Application runs here

    logger.info("=== RepoInsight AI shutting down ===")


# ── FastAPI application ────────────────────────────────────────────────────────

app = FastAPI(
    title="RepoInsight AI",
    description=(
        "Production-grade Retrieval-Augmented Generation (RAG) service for "
        "codebase intelligence. Clone any Git repository, index it semantically, "
        "and ask natural-language questions grounded in verified source code."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
    responses={
        422: {"description": "Validation Error"},
        500: {"model": ErrorDetail, "description": "Internal Server Error"},
    },
)

# ── Middleware ────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_timing_middleware(request: Request, call_next):
    """Log each request's method, path, status code, and wall-clock duration."""
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "%s %s → %d (%.1f ms)",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


# ── Exception handlers ────────────────────────────────────────────────────────


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all handler that converts unexpected exceptions into structured 500 responses."""
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorDetail(
            error="InternalServerError",
            detail=str(exc),
        ).model_dump(),
    )


# ── Health check ──────────────────────────────────────────────────────────────


@app.get(
    "/",
    summary="Dashboard UI",
    tags=["UI"],
    response_class=HTMLResponse,
)
async def root() -> HTMLResponse:
    """
    Serve the highly aesthetic RepoInsight AI frontend dashboard.
    """
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            content = f.read()
        return HTMLResponse(content=content, status_code=200)
    except Exception as exc:
        return HTMLResponse(
            content=f"<h3>Failed to load dashboard: {exc}</h3>",
            status_code=500,
        )


@app.get(
    "/api/repos",
    summary="List ingested repositories",
    tags=["Repository"],
    response_model=Dict[str, List[str]],
)
async def list_repositories() -> Dict[str, List[str]]:
    """
    Return a list of all currently ingested repository IDs.
    """
    settings = get_settings()
    storage_dir = settings.REPOS_STORAGE_DIR
    repos = []
    if os.path.exists(storage_dir):
        for entry in os.listdir(storage_dir):
            if os.path.isdir(os.path.join(storage_dir, entry)):
                repos.append(entry)
    return {"repos": repos}


@app.get(
    "/api/stats",
    summary="Collection statistics",
    tags=["Health"],
    response_model=Dict[str, Any],
)
async def get_stats() -> Dict[str, Any]:
    """
    Return current ChromaDB collection statistics.

    Useful for monitoring the total number of indexed vectors across all
    repositories without querying the collection directly.
    """
    return get_collection_stats()


# ── POST /api/ingest ──────────────────────────────────────────────────────────


@app.post(
    "/api/ingest",
    summary="Ingest a Git repository",
    tags=["Repository"],
    response_model=IngestResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": ErrorDetail, "description": "Invalid request payload"},
        422: {"description": "Validation Error"},
        500: {"model": ErrorDetail, "description": "Ingestion pipeline failure"},
    },
)
async def ingest_repository(payload: IngestRequest) -> IngestResponse:
    """
    Clone, parse, embed, and index a Git repository.

    **Pipeline steps (all CPU/IO-bound work runs in a thread-pool):**
    1. Resolve or derive the `repo_id` slug.
    2. Delete any existing vectors for this `repo_id` (delta-indexing).
    3. Clone the repository (shallow, depth=1).
    4. Recursively walk and filter the file tree.
    5. Split each file into syntax-aware chunks with precise line metadata.
    6. Batch-embed all chunks via the OpenAI Embeddings API.
    7. Write the vectors + metadata to ChromaDB in batches of 100.
    8. Return ingestion statistics.

    **Supported repository types:** Any publicly accessible HTTPS or SSH Git URL.
    Private repositories require the server's Git credentials to be configured.
    """
    loop = asyncio.get_event_loop()

    def _run_ingestion() -> Dict[str, Any]:
        """Blocking ingestion pipeline executed inside a thread-pool worker."""
        repo_id = resolve_repo_id(payload.repo_url, payload.repo_id)
        logger.info("Starting ingestion for repo_id='%s'.", repo_id)

        # ── 1. Delta-index: purge existing vectors ───────────────────────────
        deleted = delete_repo_vectors(repo_id)
        if deleted > 0:
            logger.info("Removed %d stale vectors for repo_id='%s'.", deleted, repo_id)

        # ── 2. Clone the repository ──────────────────────────────────────────
        try:
            clone_path = clone_repository(payload.repo_url, repo_id)
        except RuntimeError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc

        # ── 3. Walk & collect files ───────────────────────────────────────────
        file_records = list(walk_repository(clone_path))
        total_files = len(file_records)
        logger.info("Collected %d source file(s) for indexing.", total_files)

        if total_files == 0:
            # Nothing to index; clean up and return early
            delete_repository_workspace(repo_id)
            return {
                "repo_id": repo_id,
                "total_chunks": 0,
                "total_files": 0,
                "message": (
                    "No indexable source files found in the repository. "
                    "All files may have been excluded by the content filter."
                ),
            }

        # ── 4. Parse into chunks ─────────────────────────────────────────────
        chunk_records = parse_repository(file_records, repo_id)
        total_chunks = len(chunk_records)
        logger.info("Generated %d chunk(s) from %d file(s).", total_chunks, total_files)

        if total_chunks == 0:
            delete_repository_workspace(repo_id)
            return {
                "repo_id": repo_id,
                "total_chunks": 0,
                "total_files": total_files,
                "message": "Parsing produced no chunks. Check file contents.",
            }

        # ── 5. Embed all chunks ──────────────────────────────────────────────
        logger.info("Embedding %d chunk(s) via OpenAI API…", total_chunks)
        texts = [cr.content for cr in chunk_records]

        # Embed in batches of 1000 (OpenAI limit per request is 2048;
        # we use 1000 to stay comfortably within token limits per batch)
        EMBED_BATCH = 1000
        all_vectors = []
        for batch_start in range(0, total_chunks, EMBED_BATCH):
            batch_texts = texts[batch_start: batch_start + EMBED_BATCH]
            batch_vectors = embed_texts(batch_texts)
            all_vectors.extend(batch_vectors)
            logger.info(
                "Embedded batch %d/%d (%d texts).",
                batch_start // EMBED_BATCH + 1,
                (total_chunks + EMBED_BATCH - 1) // EMBED_BATCH,
                len(batch_texts),
            )

        # ── 6. Write to ChromaDB ─────────────────────────────────────────────
        ids = [cr.chunk_id for cr in chunk_records]
        metadatas = [cr.to_metadata() for cr in chunk_records]

        upsert_chunks(
            ids=ids,
            embeddings=all_vectors,
            documents=texts,
            metadatas=metadatas,
        )
        logger.info(
            "Ingestion complete for repo_id='%s': %d file(s), %d chunk(s).",
            repo_id,
            total_files,
            total_chunks,
        )

        return {
            "repo_id": repo_id,
            "total_chunks": total_chunks,
            "total_files": total_files,
            "message": "Repository indexed successfully.",
        }

    try:
        result = await loop.run_in_executor(None, _run_ingestion)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Ingestion pipeline failed for '%s'.", payload.repo_url)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {exc}",
        ) from exc

    return IngestResponse(
        status="success",
        repo_id=result["repo_id"],
        total_chunks=result["total_chunks"],
        total_files=result["total_files"],
        message=result["message"],
    )


# ── POST /api/query ────────────────────────────────────────────────────────────


@app.post(
    "/api/query",
    summary="Query an indexed repository",
    tags=["Query"],
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    responses={
        404: {"model": ErrorDetail, "description": "Repository not indexed"},
        422: {"description": "Validation Error"},
        500: {"model": ErrorDetail, "description": "RAG pipeline failure"},
    },
)
async def query_repository(payload: QueryRequest) -> QueryResponse:
    """
    Run the full RAG pipeline against an indexed repository.

    **Pipeline steps:**
    1. Embed the incoming `query` string using `text-embedding-3-small`.
    2. Retrieve the top-4 most semantically similar code chunks from ChromaDB,
       filtered strictly to `repo_id`.
    3. Inject the retrieved snippets (with file paths and line numbers) into a
       deterministic system prompt designed to prevent hallucinations.
    4. Call the OpenAI Chat Completions API (`gpt-4o-mini` by default).
    5. Return the Markdown-formatted answer and an array of `SourceReference`
       objects pointing to the exact files and line ranges used as context.

    **Prerequisites:** The target repository must first be indexed via
    `POST /api/ingest`.
    """
    loop = asyncio.get_event_loop()

    def _run_query() -> Dict[str, Any]:
        try:
            answer, sources = run_rag_pipeline(
                repo_id=payload.repo_id,
                query=payload.query,
            )
        except RuntimeError as exc:
            # RuntimeError is raised when no vectors exist for repo_id
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(exc),
            ) from exc
        return {"answer": answer, "sources": sources}

    try:
        result = await loop.run_in_executor(None, _run_query)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(
            "RAG pipeline failed for repo_id='%s', query='%s'.",
            payload.repo_id,
            payload.query[:80],
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query pipeline failed: {exc}",
        ) from exc

    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"],
        repo_id=payload.repo_id,
    )


# ── DELETE /api/repo/{repo_id} ────────────────────────────────────────────────


@app.delete(
    "/api/repo/{repo_id}",
    summary="Delete a repository and its vectors",
    tags=["Repository"],
    response_model=DeleteResponse,
    status_code=status.HTTP_200_OK,
    responses={
        404: {"model": ErrorDetail, "description": "Repository not found"},
        500: {"model": ErrorDetail, "description": "Deletion failure"},
    },
)
async def delete_repository(
    repo_id: str = Path(
        ...,
        description="The repository namespace identifier to delete.",
        min_length=1,
        max_length=128,
        examples=["fastapi-main"],
    ),
) -> DeleteResponse:
    """
    Delete all vectors for a repository from ChromaDB and remove its cloned workspace.

    This operation is **irreversible**. To re-index the repository after deletion,
    call `POST /api/ingest` again with the same `repo_url`.

    **What is deleted:**
    * All ChromaDB vectors filtered by `repo_id`.
    * The local clone directory under `REPOS_STORAGE_DIR`.

    **What is NOT deleted:**
    * The ChromaDB collection itself (shared across all repositories).
    * Any other repositories' vectors.
    """
    loop = asyncio.get_event_loop()

    def _run_deletion() -> Dict[str, Any]:
        # Delete vectors from ChromaDB
        deleted_vectors = delete_repo_vectors(repo_id)

        # Delete local clone workspace
        workspace_deleted = delete_repository_workspace(repo_id)

        return {
            "deleted_vectors": deleted_vectors,
            "workspace_deleted": workspace_deleted,
        }

    try:
        result = await loop.run_in_executor(None, _run_deletion)
    except Exception as exc:
        logger.exception("Deletion failed for repo_id='%s'.", repo_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Deletion failed: {exc}",
        ) from exc

    deleted_vectors = result["deleted_vectors"]
    workspace_deleted = result["workspace_deleted"]

    parts = []
    if deleted_vectors > 0:
        parts.append(f"{deleted_vectors} vector(s) removed from ChromaDB")
    else:
        parts.append("No vectors found in ChromaDB for this repo_id")

    if workspace_deleted:
        parts.append("cloned workspace deleted from disk")
    else:
        parts.append("no cloned workspace found on disk")

    message = "; ".join(parts) + "."

    return DeleteResponse(
        status="success",
        repo_id=repo_id,
        message=message,
    )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=False,   # Our custom middleware handles request logging
    )
