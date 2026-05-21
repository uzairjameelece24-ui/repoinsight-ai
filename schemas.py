"""
schemas.py — Pydantic request/response models for the RepoInsight AI API.

All models are strict-typed with rich docstrings and example values so
FastAPI can auto-generate accurate OpenAPI documentation.
"""

from typing import List, Optional

from pydantic import BaseModel, Field, HttpUrl, field_validator


# ── Ingest Endpoint ──────────────────────────────────────────────────────────


class IngestRequest(BaseModel):
    """
    Payload for ``POST /api/ingest``.

    Attributes
    ----------
    repo_url : str
        HTTPS or SSH URL of the Git repository to clone and index.
    repo_id : Optional[str]
        Human-readable identifier used to namespace the vectors in ChromaDB.
        If omitted, a stable slug is derived from the repository URL.
    """

    repo_url: str = Field(
        ...,
        description="HTTPS or SSH URL of the Git repository to clone and index.",
        examples=["https://github.com/tiangolo/fastapi"],
    )
    repo_id: Optional[str] = Field(
        default=None,
        description=(
            "Optional human-readable identifier for this repository. "
            "Used to namespace vectors in ChromaDB. "
            "If omitted, a stable slug is derived from the URL."
        ),
        examples=["fastapi-main"],
        min_length=1,
        max_length=128,
    )

    @field_validator("repo_url")
    @classmethod
    def validate_repo_url(cls, v: str) -> str:
        """Ensure the URL is a non-empty string pointing to a plausible Git host."""
        stripped = v.strip()
        if not stripped:
            raise ValueError("repo_url must not be empty.")
        # Accept HTTPS and SSH git URLs
        if not (
            stripped.startswith("https://")
            or stripped.startswith("http://")
            or stripped.startswith("git@")
            or stripped.startswith("git://")
        ):
            raise ValueError(
                f"repo_url must be a valid HTTPS or SSH Git URL. Got: {stripped!r}"
            )
        return stripped

    @field_validator("repo_id")
    @classmethod
    def sanitize_repo_id(cls, v: Optional[str]) -> Optional[str]:
        """Strip whitespace and enforce safe characters in the repo_id slug."""
        if v is None:
            return v
        import re
        sanitized = re.sub(r"[^a-zA-Z0-9_\-]", "-", v.strip())
        if not sanitized:
            raise ValueError("repo_id must contain at least one alphanumeric character.")
        return sanitized

    model_config = {
        "json_schema_extra": {
            "example": {
                "repo_url": "https://github.com/tiangolo/fastapi",
                "repo_id": "fastapi-main",
            }
        }
    }


class IngestResponse(BaseModel):
    """
    Response for ``POST /api/ingest``.

    Attributes
    ----------
    status : str
        Human-readable status string, e.g. ``"success"``.
    repo_id : str
        The effective repo_id used to namespace the indexed vectors.
    total_chunks : int
        Total number of text/code chunks embedded and stored in ChromaDB.
    total_files : int
        Number of source files processed during the ingestion run.
    message : str
        Descriptive human-readable summary of the operation.
    """

    status: str = Field(..., examples=["success"])
    repo_id: str = Field(..., examples=["fastapi-main"])
    total_chunks: int = Field(..., ge=0, examples=[342])
    total_files: int = Field(..., ge=0, examples=[47])
    message: str = Field(..., examples=["Repository indexed successfully."])

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "success",
                "repo_id": "fastapi-main",
                "total_chunks": 342,
                "total_files": 47,
                "message": "Repository indexed successfully.",
            }
        }
    }


# ── Query Endpoint ───────────────────────────────────────────────────────────


class QueryRequest(BaseModel):
    """
    Payload for ``POST /api/query``.

    Attributes
    ----------
    repo_id : str
        Identifies which repository namespace to search in ChromaDB.
    query : str
        The natural-language or technical question to answer about the codebase.
    """

    repo_id: str = Field(
        ...,
        description="Identifies which repository namespace to query in ChromaDB.",
        examples=["fastapi-main"],
        min_length=1,
        max_length=128,
    )
    query: str = Field(
        ...,
        description="Natural-language or technical question about the codebase.",
        examples=["How does FastAPI handle dependency injection?"],
        min_length=3,
        max_length=2000,
    )

    @field_validator("query")
    @classmethod
    def strip_query(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("query must not be empty or whitespace only.")
        return stripped

    model_config = {
        "json_schema_extra": {
            "example": {
                "repo_id": "fastapi-main",
                "query": "How does FastAPI handle dependency injection?",
            }
        }
    }


class SourceReference(BaseModel):
    """
    A single source code location cited in the LLM response.

    Attributes
    ----------
    file_path : str
        Repository-relative path of the source file.
    line_start : int
        First line number of the cited chunk (1-indexed).
    line_end : int
        Last line number of the cited chunk (1-indexed).
    language : str
        Detected programming language of the file.
    content : str
        The raw code content of the retrieved chunk.
    """

    file_path: str = Field(..., examples=["fastapi/dependencies/utils.py"])
    line_start: int = Field(..., ge=1, examples=[20])
    line_end: int = Field(..., ge=1, examples=[45])
    language: str = Field(..., examples=["python"])
    content: str = Field(
        default="",
        description="The raw code content of the retrieved chunk.",
        examples=["def example():\n    return 'Hello World'"],
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "file_path": "fastapi/dependencies/utils.py",
                "line_start": 20,
                "line_end": 45,
                "language": "python",
                "content": "def example():\n    return 'Hello World'",
            }
        }
    }


class QueryResponse(BaseModel):
    """
    Response for ``POST /api/query``.

    Attributes
    ----------
    answer : str
        Markdown-formatted LLM response grounded in the retrieved code snippets.
    sources : List[SourceReference]
        Array of source file locations that were provided as context to the LLM.
    repo_id : str
        The repository namespace that was queried.
    """

    answer: str = Field(
        ...,
        description="Markdown-formatted LLM response grounded in retrieved code.",
    )
    sources: List[SourceReference] = Field(
        default_factory=list,
        description="Source code locations used as context for the answer.",
    )
    repo_id: str = Field(..., examples=["fastapi-main"])

    model_config = {
        "json_schema_extra": {
            "example": {
                "answer": "FastAPI handles dependency injection via `Depends()`...",
                "sources": [
                    {
                        "file_path": "fastapi/dependencies/utils.py",
                        "line_start": 20,
                        "line_end": 45,
                        "language": "python",
                    }
                ],
                "repo_id": "fastapi-main",
            }
        }
    }


# ── Delete Endpoint ──────────────────────────────────────────────────────────


class DeleteResponse(BaseModel):
    """
    Response for ``DELETE /api/repo/{repo_id}``.

    Attributes
    ----------
    status : str
        Human-readable status string.
    repo_id : str
        The repository namespace that was deleted.
    message : str
        Descriptive summary of what was removed.
    """

    status: str = Field(..., examples=["success"])
    repo_id: str = Field(..., examples=["fastapi-main"])
    message: str = Field(
        ...,
        examples=["Repository vectors and cloned workspace deleted successfully."],
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "success",
                "repo_id": "fastapi-main",
                "message": "Repository vectors and cloned workspace deleted successfully.",
            }
        }
    }


# ── Generic Error ────────────────────────────────────────────────────────────


class ErrorDetail(BaseModel):
    """
    Standard error response body returned on 4xx/5xx responses.

    Attributes
    ----------
    error : str
        Short error category label (e.g. ``"ValidationError"``).
    detail : str
        Human-readable description of what went wrong.
    """

    error: str = Field(..., examples=["RepositoryNotFound"])
    detail: str = Field(
        ...,
        examples=["No vectors found for repo_id 'my-repo'. Did you run /api/ingest?"],
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "error": "RepositoryNotFound",
                "detail": "No vectors found for repo_id 'my-repo'.",
            }
        }
    }
