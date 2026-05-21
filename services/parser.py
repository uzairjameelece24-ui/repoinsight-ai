"""
services/parser.py — Language-aware syntax chunking and metadata enrichment.

Responsibilities
----------------
1. Map file extensions to LangChain ``Language`` enum values.
2. Use ``RecursiveCharacterTextSplitter.from_language`` for language-aware
   splitting; fall back to a standard ``RecursiveCharacterTextSplitter`` for
   unknown extensions.
3. Compute precise ``line_start`` / ``line_end`` metadata for every chunk by
   tracking character offsets against the original file's newline positions.
4. Return a flat list of ``ChunkRecord`` dataclasses ready for embedding.

Design Notes
------------
* Chunk sizes and overlaps come from ``Settings`` so they are configurable
  without code changes.
* The line-number calculation uses bisection (``bisect_right``) over a
  pre-computed list of cumulative character offsets for O(log n) performance
  on large files.
* Every ``ChunkRecord`` carries all metadata fields required by the ChromaDB
  schema — nothing is left as ``None``.
"""

import bisect
import hashlib
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from config import get_settings
from services.ingestion import FileRecord

logger = logging.getLogger(__name__)


# ── Extension → Language mapping ─────────────────────────────────────────────

#: Maps lower-cased file extensions to LangChain ``Language`` enum values.
EXTENSION_LANGUAGE_MAP: Dict[str, Language] = {
    # Python
    ".py": Language.PYTHON,
    ".pyw": Language.PYTHON,
    ".pyi": Language.PYTHON,
    # JavaScript / TypeScript
    ".js": Language.JS,
    ".jsx": Language.JS,
    ".mjs": Language.JS,
    ".cjs": Language.JS,
    ".ts": Language.TS,
    ".tsx": Language.TS,
    # C / C++
    ".c": Language.C,
    ".h": Language.C,
    ".cpp": Language.CPP,
    ".cc": Language.CPP,
    ".cxx": Language.CPP,
    ".hpp": Language.CPP,
    ".hxx": Language.CPP,
    # Java
    ".java": Language.JAVA,
    # Go
    ".go": Language.GO,
    # Rust
    ".rs": Language.RUST,
    # Ruby
    ".rb": Language.RUBY,
    ".rake": Language.RUBY,
    # PHP
    ".php": Language.PHP,
    # Swift
    ".swift": Language.SWIFT,
    # Kotlin
    ".kt": Language.KOTLIN,
    ".kts": Language.KOTLIN,
    # Scala
    ".scala": Language.SCALA,
    # Markdown / RST (treated as text)
    ".md": Language.MARKDOWN,
    ".mdx": Language.MARKDOWN,
    ".rst": Language.RST,
    # HTML / CSS / SCSS
    ".html": Language.HTML,
    ".htm": Language.HTML,
}

#: Human-readable language labels stored in metadata (keyed by ``Language`` enum).
LANGUAGE_LABEL_MAP: Dict[Language, str] = {
    Language.PYTHON: "python",
    Language.JS: "javascript",
    Language.TS: "typescript",
    Language.C: "c",
    Language.CPP: "cpp",
    Language.JAVA: "java",
    Language.GO: "go",
    Language.RUST: "rust",
    Language.RUBY: "ruby",
    Language.PHP: "php",
    Language.SWIFT: "swift",
    Language.KOTLIN: "kotlin",
    Language.SCALA: "scala",
    Language.MARKDOWN: "markdown",
    Language.RST: "rst",
    Language.HTML: "html",
}


# ── Data structures ───────────────────────────────────────────────────────────


@dataclass
class ChunkRecord:
    """
    A single syntax-aware code/text chunk ready for embedding.

    Attributes
    ----------
    chunk_id : str
        Globally unique identifier derived from repo_id + file_path + chunk index.
    repo_id : str
        Repository namespace identifier.
    file_path : str
        Repository-relative path of the source file.
    content : str
        Raw text of the chunk.
    line_start : int
        First line number within the original file (1-indexed).
    line_end : int
        Last line number within the original file (1-indexed).
    language : str
        Human-readable language label (e.g. ``"python"``).
    """

    chunk_id: str
    repo_id: str
    file_path: str
    content: str
    line_start: int
    line_end: int
    language: str

    def to_metadata(self) -> Dict[str, object]:
        """
        Return the ChromaDB-compatible metadata dictionary for this chunk.

        Returns
        -------
        dict
            Keys: ``repo_id``, ``file_path``, ``line_start``, ``line_end``,
            ``language``.
        """
        return {
            "repo_id": self.repo_id,
            "file_path": self.file_path,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "language": self.language,
        }


# ── Line-number utilities ─────────────────────────────────────────────────────


def _build_line_offsets(text: str) -> List[int]:
    """
    Build a list of cumulative character offsets for each line boundary.

    The returned list has one entry per ``\n`` in ``text``. Each entry is the
    index of the character *immediately after* the newline (i.e., the start of
    the next line). This enables O(log n) line-number lookup via bisection.

    For example, for the text ``"ab\ncd\nef"``:
    * Line 1 starts at char 0 (``a``).
    * Line 2 starts at char 3 (``c``), after the ``\n`` at index 2.
    * Line 3 starts at char 6 (``e``).
    * Returns ``[3, 6]``.

    Parameters
    ----------
    text : str
        The full file content.

    Returns
    -------
    List[int]
        Sorted list of character offsets where each new line starts.
    """
    offsets: List[int] = []
    idx = 0
    while True:
        newline_pos = text.find("\n", idx)
        if newline_pos == -1:
            break
        offsets.append(newline_pos + 1)  # +1: first char of the *next* line
        idx = newline_pos + 1
    return offsets


def _char_offset_to_line(char_offset: int, line_offsets: List[int]) -> int:
    """
    Convert a character offset into a 1-indexed line number.

    Uses ``bisect_right`` against the pre-computed ``line_offsets`` list for
    O(log n) performance.

    Parameters
    ----------
    char_offset : int
        Zero-based character offset into the full file content.
    line_offsets : List[int]
        Output of ``_build_line_offsets``.

    Returns
    -------
    int
        1-indexed line number.
    """
    return bisect.bisect_right(line_offsets, char_offset) + 1


def _find_chunk_offset(full_text: str, chunk_text: str, search_start: int = 0) -> int:
    """
    Find the character offset of ``chunk_text`` within ``full_text``.

    Searches from ``search_start`` to support sequential chunk matching
    without re-scanning the entire file for every chunk.

    Parameters
    ----------
    full_text : str
        The original file content.
    chunk_text : str
        The chunk text to locate.
    search_start : int
        Offset in ``full_text`` from which to begin searching.

    Returns
    -------
    int
        Zero-based character offset of the first occurrence of ``chunk_text``
        at or after ``search_start``. Returns ``search_start`` as a fallback
        if the chunk cannot be found (prevents crashes on edge cases).
    """
    idx = full_text.find(chunk_text, search_start)
    if idx == -1:
        # Fallback: try stripping leading/trailing whitespace from both sides
        stripped_chunk = chunk_text.strip()
        if stripped_chunk:
            idx = full_text.find(stripped_chunk, search_start)
    if idx == -1:
        logger.debug(
            "Could not locate chunk at offset %d; using fallback position.",
            search_start,
        )
        return search_start
    return idx


# ── Splitter factory ──────────────────────────────────────────────────────────


def _get_splitter(language: Optional[Language]) -> tuple["RecursiveCharacterTextSplitter", int]:
    """
    Return the appropriate ``RecursiveCharacterTextSplitter`` and its overlap value.

    Returns a 2-tuple so callers can advance the search cursor correctly
    without relying on private splitter attributes.

    Parameters
    ----------
    language : Optional[Language]
        LangChain ``Language`` enum value, or ``None`` for plain-text fallback.

    Returns
    -------
    tuple[RecursiveCharacterTextSplitter, int]
        (splitter instance, chunk_overlap value)
    """
    settings = get_settings()
    chunk_size = settings.MAX_CHUNK_SIZE
    overlap = settings.CHUNK_OVERLAP

    if language is not None:
        try:
            splitter = RecursiveCharacterTextSplitter.from_language(
                language=language,
                chunk_size=chunk_size,
                chunk_overlap=overlap,
            )
            return splitter, overlap
        except Exception as exc:  # pragma: no cover
            logger.warning(
                "Could not create language-specific splitter for %s: %s. "
                "Falling back to generic splitter.",
                language,
                exc,
            )

    # Generic text fallback
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        add_start_index=False,
    )
    return splitter, overlap


# ── Public API ────────────────────────────────────────────────────────────────


def parse_file(file_record: FileRecord, repo_id: str) -> List[ChunkRecord]:
    """
    Split a single ``FileRecord`` into annotated ``ChunkRecord`` objects.

    Steps
    -----
    1. Look up the ``Language`` enum from the file extension.
    2. Select the appropriate ``RecursiveCharacterTextSplitter``.
    3. Split the file content into chunks.
    4. For each chunk, locate its character offset within the original text and
       derive 1-indexed ``line_start`` / ``line_end`` values.
    5. Construct ``ChunkRecord`` objects with fully populated metadata.

    Parameters
    ----------
    file_record : FileRecord
        Source file data from the ingestion service.
    repo_id : str
        Repository namespace identifier.

    Returns
    -------
    List[ChunkRecord]
        Ordered list of chunk records; may be empty if splitting yields nothing.
    """
    language_enum = EXTENSION_LANGUAGE_MAP.get(file_record.extension)
    language_label = (
        LANGUAGE_LABEL_MAP.get(language_enum, "text")
        if language_enum is not None
        else "text"
    )

    splitter, chunk_overlap = _get_splitter(language_enum)

    try:
        raw_chunks: List[str] = splitter.split_text(file_record.content)
    except Exception as exc:
        logger.error(
            "Failed to split '%s': %s — skipping file.",
            file_record.relative_path,
            exc,
        )
        return []

    if not raw_chunks:
        return []

    full_text = file_record.content
    line_offsets = _build_line_offsets(full_text)

    chunk_records: List[ChunkRecord] = []
    search_cursor = 0  # advance sequentially to speed up repeated find()

    for chunk_idx, chunk_text in enumerate(raw_chunks):
        if not chunk_text.strip():
            continue

        # Locate this chunk's position within the original file
        char_start = _find_chunk_offset(full_text, chunk_text, search_cursor)
        char_end = char_start + len(chunk_text) - 1

        line_start = _char_offset_to_line(char_start, line_offsets)
        line_end = _char_offset_to_line(max(char_end, char_start), line_offsets)

        # Advance the search cursor past the start of this chunk so the next
        # chunk is found sequentially (chunks are ordered, non-overlapping start).
        search_cursor = char_start + max(1, len(chunk_text) - chunk_overlap)

        # Derive a stable, unique ID for this chunk
        id_source = f"{repo_id}::{file_record.relative_path}::{chunk_idx}"
        chunk_id = hashlib.sha256(id_source.encode()).hexdigest()

        chunk_records.append(
            ChunkRecord(
                chunk_id=chunk_id,
                repo_id=repo_id,
                file_path=file_record.relative_path,
                content=chunk_text,
                line_start=line_start,
                line_end=line_end,
                language=language_label,
            )
        )

    logger.debug(
        "Parsed '%s' → %d chunk(s) [%s].",
        file_record.relative_path,
        len(chunk_records),
        language_label,
    )
    return chunk_records


def parse_repository(
    file_records: List[FileRecord], repo_id: str
) -> List[ChunkRecord]:
    """
    Parse an entire repository's file records into chunk records.

    Iterates over every ``FileRecord`` and collects all ``ChunkRecord``
    objects into a single flat list.

    Parameters
    ----------
    file_records : List[FileRecord]
        All accepted files from ``services.ingestion.walk_repository``.
    repo_id : str
        Repository namespace identifier.

    Returns
    -------
    List[ChunkRecord]
        All chunks across all files, in traversal order.
    """
    all_chunks: List[ChunkRecord] = []
    for file_record in file_records:
        chunks = parse_file(file_record, repo_id)
        all_chunks.extend(chunks)
    logger.info(
        "Parsed %d file(s) → %d total chunk(s) for repo_id='%s'.",
        len(file_records),
        len(all_chunks),
        repo_id,
    )
    return all_chunks
