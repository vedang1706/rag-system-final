"""
verification_1: RAG Context & Evidence Snapshot Utility

Preserves the exact retrieved chunks, context chunks, and final prompt context
delivered to the LLM during generation for downstream verification experiments.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import logging
import re

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Constants matching src/generate.py
DEFAULT_MAX_CONTEXT_CHARS = 20000
DEFAULT_SNAPSHOT_DIR = Path(__file__).resolve().parent / "output" / "rag_snapshots"

# Required chunk keys produced by src/retrieve.py
REQUIRED_CHUNK_KEYS = {
    "chunk_id",
    "text",
    "section",
    "section_path",
    "chapter",
    "page_start",
    "page_end",
    "pages",
    "rrf_score",
    "source",
    "is_supplementary",
}


def extract_context_chunks(
    retrieved_chunks: List[Dict[str, Any]],
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Deterministically extracts the subset of retrieved chunks that fit within
    the LLM context character limit and builds the exact final context string.

    Mirrors the prompt context construction in src/generate.py build_prompt().

    Args:
        retrieved_chunks: Full list of chunks returned by the retrieval stage.
        max_context_chars: Maximum character limit for context passages.

    Returns:
        A tuple of (context_chunks, final_context_str).
    """
    context_chunks: List[Dict[str, Any]] = []
    context_parts: List[str] = []
    total_chars = 0

    for i, chunk in enumerate(retrieved_chunks):
        section = chunk.get("section", "")
        page_start = chunk.get("page_start", "")
        page_end = chunk.get("page_end", "")
        text = chunk.get("text", "")

        part = (
            f"[Source {i+1}] "
            f"Section: {section} | "
            f"Pages: {page_start}-{page_end}\n"
            f"{text}"
        )

        # Stop adding chunks once the character threshold would be exceeded
        if total_chars + len(part) > max_context_chars:
            break

        context_chunks.append(chunk)
        context_parts.append(part)
        total_chars += len(part)

    final_context = "\n\n---\n\n".join(context_parts)
    return context_chunks, final_context


def create_snapshot(
    query_id: str,
    question: str,
    answer: str,
    retrieved_chunks: List[Dict[str, Any]],
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    metadata_extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Creates a standardized verification snapshot preserving all retrieval
    and context evidence for a single query.

    Args:
        query_id: Unique query identifier.
        question: User query text.
        answer: Generated answer text.
        retrieved_chunks: Full list of retrieved chunk dictionaries.
        max_context_chars: Context character limit used during generation.
        metadata_extra: Optional extra metadata dictionary.

    Returns:
        Structured snapshot dictionary conforming to the verification schema.
    """
    context_chunks, final_context = extract_context_chunks(
        retrieved_chunks=retrieved_chunks, max_context_chars=max_context_chars
    )

    metadata = {
        "max_context_chars": max_context_chars,
        "retrieved_chunk_count": len(retrieved_chunks),
        "context_chunk_count": len(context_chunks),
    }
    if metadata_extra and isinstance(metadata_extra, dict):
        metadata.update(metadata_extra)

    snapshot: Dict[str, Any] = {
        "query_id": str(query_id),
        "question": str(question),
        "answer": str(answer),
        "retrieved_chunks": retrieved_chunks,
        "context_chunks": context_chunks,
        "final_context": final_context,
        "metadata": metadata,
    }

    validate_snapshot(snapshot)
    return snapshot


def validate_snapshot(snapshot: Dict[str, Any]) -> bool:
    """
    Validates that a snapshot dictionary satisfies all structural and
    type requirements.

    Raises:
        ValueError or TypeError if validation fails.

    Returns:
        True if valid.
    """
    if not isinstance(snapshot, dict):
        raise TypeError(f"Snapshot must be a dict, got {type(snapshot).__name__}")

    required_top_keys = [
        "query_id",
        "question",
        "answer",
        "retrieved_chunks",
        "context_chunks",
        "final_context",
        "metadata",
    ]
    for key in required_top_keys:
        if key not in snapshot:
            raise ValueError(f"Snapshot missing required top-level key: '{key}'")

    if not isinstance(snapshot["query_id"], str) or not snapshot["query_id"].strip():
        raise ValueError("Snapshot 'query_id' must be a non-empty string.")
    if not isinstance(snapshot["question"], str):
        raise TypeError("Snapshot 'question' must be a string.")
    if not isinstance(snapshot["answer"], str):
        raise TypeError("Snapshot 'answer' must be a string.")
    if not isinstance(snapshot["retrieved_chunks"], list):
        raise TypeError("Snapshot 'retrieved_chunks' must be a list.")
    if not isinstance(snapshot["context_chunks"], list):
        raise TypeError("Snapshot 'context_chunks' must be a list.")
    if not isinstance(snapshot["final_context"], str):
        raise TypeError("Snapshot 'final_context' must be a string.")
    if not isinstance(snapshot["metadata"], dict):
        raise TypeError("Snapshot 'metadata' must be a dict.")

    # Validate metadata counts
    meta = snapshot["metadata"]
    for m_key in ["max_context_chars", "retrieved_chunk_count", "context_chunk_count"]:
        if m_key not in meta:
            raise ValueError(f"Snapshot metadata missing '{m_key}'")

    if meta["retrieved_chunk_count"] != len(snapshot["retrieved_chunks"]):
        raise ValueError(
            f"Metadata retrieved_chunk_count ({meta['retrieved_chunk_count']}) "
            f"does not match actual retrieved_chunks count ({len(snapshot['retrieved_chunks'])})"
        )
    if meta["context_chunk_count"] != len(snapshot["context_chunks"]):
        raise ValueError(
            f"Metadata context_chunk_count ({meta['context_chunk_count']}) "
            f"does not match actual context_chunks count ({len(snapshot['context_chunks'])})"
        )

    # Validate invariant: context_chunks must be a prefix of retrieved_chunks
    retrieved = snapshot["retrieved_chunks"]
    context = snapshot["context_chunks"]
    if len(context) > len(retrieved):
        raise ValueError("context_chunks cannot have more elements than retrieved_chunks.")

    for i, c_chunk in enumerate(context):
        if c_chunk != retrieved[i]:
            raise ValueError(
                f"context_chunks[{i}] does not match retrieved_chunks[{i}]. "
                "Invariant violated: context_chunks must be an exact prefix of retrieved_chunks."
            )

    # Validate chunk structure for all retrieved chunks
    for idx, chunk in enumerate(retrieved):
        if not isinstance(chunk, dict):
            raise TypeError(f"Chunk at index {idx} in retrieved_chunks is not a dict.")
        missing = REQUIRED_CHUNK_KEYS - set(chunk.keys())
        if missing:
            raise ValueError(
                f"Chunk at index {idx} ({chunk.get('chunk_id', 'unknown')}) "
                f"is missing required fields: {sorted(list(missing))}"
            )

    return True


def sanitize_filename(name: str) -> str:
    """
    Sanitizes a string for safe filesystem usage.
    """
    safe = re.sub(r"[^\w\-.]", "_", str(name))
    return safe if safe else "unnamed_snapshot"


def save_snapshot(
    snapshot: Dict[str, Any],
    output_dir: Union[str, Path] = DEFAULT_SNAPSHOT_DIR,
) -> Path:
    """
    Saves a validated snapshot dictionary to disk as a JSON file.

    Args:
        snapshot: The validated snapshot dictionary.
        output_dir: Directory where the snapshot file will be stored.

    Returns:
        The Path to the saved snapshot file.
    """
    validate_snapshot(snapshot)

    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{sanitize_filename(snapshot['query_id'])}.json"
    target_path = out_dir / filename

    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, ensure_ascii=False)

    logger.debug(f"Saved snapshot to {target_path}")
    return target_path


def load_snapshot(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Loads and validates a snapshot dictionary from a JSON file.

    Args:
        file_path: Path to the JSON snapshot file.

    Returns:
        Parsed and validated snapshot dictionary.
    """
    path = Path(file_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Snapshot file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    validate_snapshot(data)
    return data
