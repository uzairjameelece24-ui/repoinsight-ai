"""
services/ingestion.py — Repository cloning and file-system traversal.

Responsibilities
----------------
1. Clone a remote Git repository into a local workspace directory.
2. Walk the file tree recursively, applying an immutable exclusion filter.
3. Read each accepted file as UTF-8 text (skipping undecodable binary files).
4. Hand off the collected ``FileRecord`` objects to the parser service.
5. Clean up the cloned workspace when explicitly asked (via the delete endpoint).

Design Notes
------------
* All I/O is synchronous and run inside FastAPI's ``run_in_executor`` at the
  endpoint layer, so the event loop is never blocked.
* The exclusion filter is defined as a frozenset (immutable, O(1) lookup).
* GitPython's ``Repo.clone_from`` is used for cloning; shallow clones
  (depth=1) are performed to minimise disk usage and clone latency.
"""

import hashlib
import logging
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, List, Optional

import git
from git.exc import GitCommandError, InvalidGitRepositoryError

from config import get_settings

logger = logging.getLogger(__name__)


# ── Exclusion filter ─────────────────────────────────────────────────────────

#: Immutable set of directory names and file extensions that must be skipped.
#: Checked against both directory entries and individual file suffixes.
EXCLUDED_DIRS: frozenset = frozenset(
    {
        ".git",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        ".env",
        "env",
        ".tox",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "dist",
        "build",
        ".next",
        ".nuxt",
        "coverage",
        ".coverage",
        "htmlcoverage",
        ".idea",
        ".vscode",
        ".DS_Store",
    }
)

#: File names that should always be excluded regardless of extension.
EXCLUDED_FILENAMES: frozenset = frozenset(
    {
        "package-lock.json",
        "yarn.lock",
        "poetry.lock",
        "pnpm-lock.yaml",
        "Pipfile.lock",
        "composer.lock",
        "Cargo.lock",
        "Gemfile.lock",
        ".DS_Store",
        "Thumbs.db",
    }
)

#: File extensions that are binary or otherwise non-parseable as source code.
EXCLUDED_EXTENSIONS: frozenset = frozenset(
    {
        # Data / config blobs
        ".json",
        ".lock",
        # Images
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".svg",
        ".ico",
        ".webp",
        ".bmp",
        ".tiff",
        ".tif",
        # Documents / archives
        ".pdf",
        ".zip",
        ".tar",
        ".gz",
        ".bz2",
        ".xz",
        ".7z",
        ".rar",
        # Compiled / binary
        ".pyc",
        ".pyo",
        ".so",
        ".dll",
        ".dylib",
        ".exe",
        ".bin",
        ".obj",
        ".o",
        ".a",
        ".lib",
        ".wasm",
        # Media
        ".mp3",
        ".mp4",
        ".wav",
        ".ogg",
        ".flac",
        ".avi",
        ".mov",
        # Fonts
        ".ttf",
        ".otf",
        ".woff",
        ".woff2",
        ".eot",
        # Database files
        ".db",
        ".sqlite",
        ".sqlite3",
    }
)

#: Maximum file size (bytes) to read. Files larger than this are skipped to
#: avoid loading huge auto-generated or minified files into memory.
MAX_FILE_SIZE_BYTES: int = 500_000  # 500 KB


# ── Data structures ──────────────────────────────────────────────────────────


@dataclass
class FileRecord:
    """
    Represents a single source file read from the repository.

    Attributes
    ----------
    relative_path : str
        Path relative to the repository root (forward-slash separated).
    absolute_path : str
        Absolute path on the local filesystem.
    content : str
        Full UTF-8 decoded text content of the file.
    extension : str
        Lower-cased file extension including the leading dot (e.g. ``".py"``).
    size_bytes : int
        File size in bytes.
    """

    relative_path: str
    absolute_path: str
    content: str
    extension: str
    size_bytes: int


# ── Utility helpers ───────────────────────────────────────────────────────────


def _derive_repo_id(repo_url: str) -> str:
    """
    Derive a stable, URL-safe repo_id slug from a repository URL.

    The slug is the last path component of the URL with ``.git`` stripped,
    combined with the first 8 characters of the URL's SHA-256 digest to
    guarantee uniqueness across forks.

    Parameters
    ----------
    repo_url : str
        The repository remote URL.

    Returns
    -------
    str
        A stable, filesystem-safe slug such as ``"fastapi-a1b2c3d4"``.
    """
    # Extract the repo name from the last URL path segment
    name_part = re.split(r"[/:]", repo_url.rstrip("/"))[-1]
    name_part = re.sub(r"\.git$", "", name_part, flags=re.IGNORECASE)
    name_part = re.sub(r"[^a-zA-Z0-9_\-]", "-", name_part)

    # Append a short hash for uniqueness
    url_hash = hashlib.sha256(repo_url.encode()).hexdigest()[:8]
    return f"{name_part}-{url_hash}"


def _get_clone_dir(repo_id: str) -> Path:
    """
    Return the absolute path of the local clone directory for ``repo_id``.

    Parameters
    ----------
    repo_id : str
        Repository namespace identifier.

    Returns
    -------
    Path
        Absolute path; the parent directory is guaranteed to exist after this call.
    """
    settings = get_settings()
    base = Path(settings.REPOS_STORAGE_DIR).resolve()
    base.mkdir(parents=True, exist_ok=True)
    return base / repo_id


def _is_excluded_path(path: Path) -> bool:
    """
    Return ``True`` if ``path`` (file or directory) should be skipped.

    Checks are performed in order of cost (cheapest first):
    1. Any path component matches ``EXCLUDED_DIRS``.
    2. File name is in ``EXCLUDED_FILENAMES``.
    3. File extension is in ``EXCLUDED_EXTENSIONS``.

    Parameters
    ----------
    path : Path
        Path to evaluate.

    Returns
    -------
    bool
    """
    # 1. Check all path components for excluded directory names
    for part in path.parts:
        if part in EXCLUDED_DIRS:
            return True

    # 2. Check the filename itself
    if path.name in EXCLUDED_FILENAMES:
        return True

    # 3. Check the extension
    if path.suffix.lower() in EXCLUDED_EXTENSIONS:
        return True

    return False


def _is_likely_binary(file_path: Path) -> bool:
    """
    Heuristically detect binary files by scanning the first 8 KB for null bytes.

    Parameters
    ----------
    file_path : Path
        Path to the file.

    Returns
    -------
    bool
        ``True`` if the file appears to be binary.
    """
    try:
        with open(file_path, "rb") as fh:
            chunk = fh.read(8192)
        return b"\x00" in chunk
    except OSError:
        return True


# ── Core service functions ────────────────────────────────────────────────────


def clone_repository(repo_url: str, repo_id: str) -> Path:
    """
    Clone a remote Git repository into the local workspace.

    A shallow clone (``depth=1``) is performed to minimise disk usage and
    clone latency. If a clone directory already exists for ``repo_id``, it
    is deleted first to ensure a clean state.

    Parameters
    ----------
    repo_url : str
        The remote repository URL.
    repo_id : str
        Unique identifier used to name the local clone directory.

    Returns
    -------
    Path
        Absolute path to the cloned repository root.

    Raises
    ------
    RuntimeError
        If the ``git clone`` operation fails for any reason.
    """
    clone_dir = _get_clone_dir(repo_id)

    # Wipe any existing clone to ensure a clean re-index
    if clone_dir.exists():
        logger.info("Removing existing clone at '%s'.", clone_dir)
        shutil.rmtree(clone_dir, ignore_errors=True)

    logger.info("Cloning '%s' → '%s' (depth=1).", repo_url, clone_dir)
    try:
        git.Repo.clone_from(
            repo_url,
            str(clone_dir),
            depth=1,
            single_branch=True,
        )
    except GitCommandError as exc:
        raise RuntimeError(
            f"Git clone failed for URL '{repo_url}': {exc}"
        ) from exc

    logger.info("Clone complete: '%s'.", clone_dir)
    return clone_dir


def walk_repository(repo_root: Path) -> Generator[FileRecord, None, None]:
    """
    Recursively walk ``repo_root`` and yield ``FileRecord`` objects.

    Files are skipped if:
    * Any ancestor directory name is in ``EXCLUDED_DIRS``.
    * The file name is in ``EXCLUDED_FILENAMES``.
    * The file extension is in ``EXCLUDED_EXTENSIONS``.
    * The file is larger than ``MAX_FILE_SIZE_BYTES``.
    * The file appears to be binary (null-byte heuristic).
    * The file cannot be decoded as UTF-8.

    Parameters
    ----------
    repo_root : Path
        Absolute path to the repository root.

    Yields
    ------
    FileRecord
        One record per accepted source file.
    """
    repo_root = repo_root.resolve()

    for dirpath, dirnames, filenames in os.walk(repo_root):
        current_dir = Path(dirpath)

        # Prune excluded directories in-place (modifies os.walk traversal)
        dirnames[:] = [
            d for d in dirnames
            if d not in EXCLUDED_DIRS and not d.startswith(".")
        ]

        for filename in filenames:
            file_path = current_dir / filename

            # Apply exclusion filters
            if _is_excluded_path(file_path):
                logger.debug("Skipping excluded path: %s", file_path)
                continue

            # Size guard
            try:
                size = file_path.stat().st_size
            except OSError as exc:
                logger.warning("Cannot stat '%s': %s — skipping.", file_path, exc)
                continue

            if size == 0:
                logger.debug("Skipping empty file: %s", file_path)
                continue

            if size > MAX_FILE_SIZE_BYTES:
                logger.debug(
                    "Skipping oversized file (%d bytes): %s", size, file_path
                )
                continue

            # Binary heuristic
            if _is_likely_binary(file_path):
                logger.debug("Skipping binary file: %s", file_path)
                continue

            # UTF-8 decode
            try:
                content = file_path.read_text(encoding="utf-8", errors="strict")
            except UnicodeDecodeError:
                logger.debug("Skipping non-UTF-8 file: %s", file_path)
                continue
            except OSError as exc:
                logger.warning("Cannot read '%s': %s — skipping.", file_path, exc)
                continue

            # Compute relative path from repo root (forward slashes for portability)
            try:
                relative = file_path.relative_to(repo_root).as_posix()
            except ValueError:
                relative = str(file_path)

            yield FileRecord(
                relative_path=relative,
                absolute_path=str(file_path),
                content=content,
                extension=file_path.suffix.lower(),
                size_bytes=size,
            )


def delete_repository_workspace(repo_id: str) -> bool:
    """
    Delete the cloned repository workspace from disk.

    Parameters
    ----------
    repo_id : str
        Repository namespace identifier whose local clone should be removed.

    Returns
    -------
    bool
        ``True`` if the directory existed and was deleted; ``False`` otherwise.
    """
    clone_dir = _get_clone_dir(repo_id)
    if clone_dir.exists():
        shutil.rmtree(clone_dir, ignore_errors=True)
        logger.info("Deleted workspace for repo_id='%s' at '%s'.", repo_id, clone_dir)
        return True
    logger.info(
        "No workspace found for repo_id='%s' at '%s' — nothing deleted.",
        repo_id,
        clone_dir,
    )
    return False


def resolve_repo_id(repo_url: str, user_provided_id: Optional[str]) -> str:
    """
    Return the effective ``repo_id`` to use for a given ingestion request.

    If the user provided a ``repo_id``, it is returned as-is (already
    sanitised by Pydantic). Otherwise a stable slug is derived from the URL.

    Parameters
    ----------
    repo_url : str
        The repository URL (used for slug derivation if needed).
    user_provided_id : Optional[str]
        User-supplied repo_id from the request payload.

    Returns
    -------
    str
        The effective repo_id.
    """
    if user_provided_id:
        return user_provided_id
    return _derive_repo_id(repo_url)
