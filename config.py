"""
config.py — Global application configuration using pydantic-settings.

All environment variables are loaded and validated at application startup.
A missing GROQ_API_KEY raises an explicit, descriptive error immediately.
"""

from functools import lru_cache
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables or a .env file.

    Attributes
    ----------
    GROQ_API_KEY : str
        Required. Groq secret key used for chat completions.
    CHROMA_DB_DIR : str
        Directory path where ChromaDB persists vector data to disk.
        Defaults to ``./chroma_data``.
    REPOS_STORAGE_DIR : str
        Directory path where cloned Git repositories are stored temporarily.
        Defaults to ``./cloned_repos``.
    EMBEDDING_MODEL : str
        SentenceTransformers embedding model identifier. Defaults to ``all-MiniLM-L6-v2``.
    CHAT_MODEL : str
        Groq chat completion model identifier. Defaults to ``llama3-8b-8192``.
    MAX_CHUNK_SIZE : int
        Target maximum character length per code chunk. Defaults to 800.
    CHUNK_OVERLAP : int
        Character overlap between consecutive chunks. Defaults to 100.
    TOP_K_RESULTS : int
        Number of nearest-neighbour results retrieved from ChromaDB per query.
        Defaults to 4.
    BATCH_SIZE : int
        Number of vectors uploaded per ChromaDB batch write. Defaults to 100.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # ── Required ────────────────────────────────────────────────────────────
    GROQ_API_KEY: str = Field(
        ...,
        description="Groq secret API key (required). Set via environment or .env file.",
    )

    # ── Optional with sensible defaults ─────────────────────────────────────
    CHROMA_DB_DIR: str = Field(
        default="./chroma_data",
        description="Directory where ChromaDB persists vector data.",
    )
    REPOS_STORAGE_DIR: str = Field(
        default="./cloned_repos",
        description="Root directory for cloned Git repositories.",
    )
    EMBEDDING_MODEL: str = Field(
        default="all-MiniLM-L6-v2",
        description="SentenceTransformers embedding model name.",
    )
    CHAT_MODEL: str = Field(
        default="llama-3.1-8b-instant",
        description="Groq chat completion model name.",
    )
    MAX_CHUNK_SIZE: int = Field(
        default=800,
        ge=100,
        description="Maximum characters per chunk.",
    )
    CHUNK_OVERLAP: int = Field(
        default=100,
        ge=0,
        description="Character overlap between consecutive chunks.",
    )
    TOP_K_RESULTS: int = Field(
        default=4,
        ge=1,
        le=20,
        description="Number of top-k vector search results.",
    )
    BATCH_SIZE: int = Field(
        default=100,
        ge=1,
        description="ChromaDB batch upload size.",
    )

    @field_validator("GROQ_API_KEY")
    @classmethod
    def groq_key_must_not_be_empty(cls, v: str) -> str:
        """Ensure the API key is a non-empty, non-whitespace string."""
        stripped = v.strip()
        if not stripped:
            raise ValueError(
                "GROQ_API_KEY must not be empty. "
                "Set it in your environment or a .env file."
            )
        if not stripped.startswith("gsk_"):
            import warnings
            warnings.warn(
                "GROQ_API_KEY does not start with 'gsk_'. "
                "Ensure this is a valid Groq API key.",
                stacklevel=2,
            )
        return stripped

    @field_validator("CHUNK_OVERLAP")
    @classmethod
    def overlap_less_than_chunk_size(cls, v: int, info) -> int:
        """Ensure chunk overlap is strictly less than chunk size."""
        chunk_size = info.data.get("MAX_CHUNK_SIZE")
        if chunk_size is not None and v >= chunk_size:
            raise ValueError(
                f"CHUNK_OVERLAP ({v}) must be less than MAX_CHUNK_SIZE ({chunk_size})."
            )
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return a cached singleton instance of Settings.

    Using ``@lru_cache`` ensures the .env file is parsed exactly once per
    process lifetime, avoiding repeated I/O on every request.

    Raises
    ------
    pydantic.ValidationError
        If required environment variables are missing or invalid.
    """
    return Settings()
