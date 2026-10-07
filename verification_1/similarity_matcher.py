"""
verification_1: Claim-to-Evidence Semantic Similarity Matcher

Computes raw semantic similarity scores between extracted sentence-level claims
and original RAG evidence chunks (both context_chunks and retrieved_chunks).
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import logging
import numpy as np

# Ensure project root is on sys.path for direct script execution
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from verification_1.claim_extractor import extract_claims
from verification_1.context_snapshot import load_snapshot

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_SNAPSHOT_DIR = Path(__file__).resolve().parent / "output" / "rag_snapshots"
DEFAULT_SIMILARITY_OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "similarity_scores"
DEFAULT_CHUNKS_PATH = Path(__file__).resolve().parent.parent / "cache" / "chunks.json"
DEFAULT_EMBEDDINGS_PATH = Path(__file__).resolve().parent.parent / "cache" / "embeddings.npy"

# Global model and embedding cache
_MODEL = None
_CHUNK_EMBEDDING_MAP: Optional[Dict[str, np.ndarray]] = None


def get_embedding_model():
    """
    Loads and caches the SentenceTransformer model once per session.
    """
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is not installed. Please install it using: pip install sentence-transformers"
        ) from exc

    logger.info(f"Loading embedding model: {EMBED_MODEL_NAME}")
    _MODEL = SentenceTransformer(EMBED_MODEL_NAME)
    return _MODEL


def compute_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Computes cosine similarity between two 1D vectors.
    If vectors are already L2-normalized, returns their dot product.
    Otherwise computes dot(a, b) / (norm(a) * norm(b)).
    """
    a = np.asarray(vec_a, dtype=np.float32).flatten()
    b = np.asarray(vec_b, dtype=np.float32).flatten()

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    # If both are unit vectors (within precision), use direct dot product
    if abs(norm_a - 1.0) < 1e-4 and abs(norm_b - 1.0) < 1e-4:
        sim = float(np.dot(a, b))
    else:
        sim = float(np.dot(a, b) / (norm_a * norm_b))

    # Clamp float precision boundary to [-1.0, 1.0]
    return max(-1.0, min(1.0, round(sim, 6)))


def load_cached_chunk_embeddings(
    chunks_path: Union[str, Path] = DEFAULT_CHUNKS_PATH,
    embeddings_path: Union[str, Path] = DEFAULT_EMBEDDINGS_PATH,
) -> Dict[str, np.ndarray]:
    """
    Loads precomputed chunk embeddings from cache/embeddings.npy and maps them
    to chunk_id using cache/chunks.json with strict alignment checks.
    """
    global _CHUNK_EMBEDDING_MAP
    if _CHUNK_EMBEDDING_MAP is not None:
        return _CHUNK_EMBEDDING_MAP

    c_path = Path(chunks_path).resolve()
    e_path = Path(embeddings_path).resolve()

    if not c_path.exists() or not e_path.exists():
        logger.warning(
            f"Embedding cache not found at {e_path} or {c_path}. "
            "Will fall back to dynamic on-the-fly chunk encoding."
        )
        _CHUNK_EMBEDDING_MAP = {}
        return _CHUNK_EMBEDDING_MAP

    try:
        with open(c_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        embeddings = np.load(e_path)

        if len(chunks) != embeddings.shape[0]:
            logger.warning(
                f"Cache mismatch: {len(chunks)} chunks in chunks.json vs {embeddings.shape[0]} in embeddings.npy. "
                "Will encode affected chunks on the fly."
            )
            _CHUNK_EMBEDDING_MAP = {}
            return _CHUNK_EMBEDDING_MAP

        mapping: Dict[str, np.ndarray] = {}
        for idx, chunk in enumerate(chunks):
            cid = chunk.get("chunk_id")
            if cid:
                vec = embeddings[idx]
                # Ensure unit normalization
                norm = np.linalg.norm(vec)
                if norm > 0 and abs(norm - 1.0) > 1e-4:
                    vec = vec / norm
                mapping[cid] = vec

        logger.info(f"Loaded {len(mapping)} cached chunk embeddings from {e_path}")
        _CHUNK_EMBEDDING_MAP = mapping
        return _CHUNK_EMBEDDING_MAP

    except Exception as e:
        logger.warning(f"Error loading embedding cache: {e}. Falling back to on-the-fly encoding.")
        _CHUNK_EMBEDDING_MAP = {}
        return _CHUNK_EMBEDDING_MAP


def get_chunk_vector(
    chunk: Dict[str, Any],
    cached_map: Dict[str, np.ndarray],
    model: Any = None,
) -> np.ndarray:
    """
    Gets the normalized embedding vector for a chunk.
    Uses cached embeddings if available; otherwise falls back to encoding chunk['text'].
    """
    cid = chunk.get("chunk_id")
    if cid and cid in cached_map:
        return cached_map[cid]

    # Fallback to encoding chunk text
    if model is None:
        model = get_embedding_model()

    text = chunk.get("text", "")
    logger.debug(f"Encoding chunk '{cid}' on the fly")
    vec = model.encode(text, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vec, dtype=np.float32).flatten()


def compute_similarity_for_snapshot(
    snapshot: Dict[str, Any],
    model: Any = None,
    cached_map: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, Any]:
    """
    Computes raw semantic similarity scores between all extracted sentence claims
    and all retrieved chunks in a RAG snapshot.

    Args:
        snapshot: Validated RAG evidence snapshot dictionary.
        model: SentenceTransformer model instance (lazy loaded if None).
        cached_map: Pre-loaded chunk_id -> embedding dict (lazy loaded if None).

    Returns:
        Structured dictionary conforming to the similarity score schema.
    """
    query_id = snapshot["query_id"]
    question = snapshot["question"]
    answer = snapshot["answer"]
    retrieved_chunks = snapshot.get("retrieved_chunks", [])
    context_chunks = snapshot.get("context_chunks", [])

    if model is None:
        model = get_embedding_model()

    if cached_map is None:
        cached_map = load_cached_chunk_embeddings()

    # Determine context chunk ID set for LLM visibility tracking
    context_chunk_ids = {c["chunk_id"] for c in context_chunks if "chunk_id" in c}

    # Extract sentence-level claims
    claim_extraction = extract_claims(answer=answer, query_id=query_id)
    claims_list = claim_extraction.get("claims", [])

    # If no claims extracted (e.g. empty answer), return valid empty structure
    if not claims_list:
        return {
            "query_id": query_id,
            "question": question,
            "answer": answer,
            "claims": [],
            "metadata": {
                "embedding_model": EMBED_MODEL_NAME,
                "similarity_metric": "cosine_similarity",
                "claim_count": 0,
                "evaluated_chunks_count": len(retrieved_chunks),
            },
        }

    # Batch encode claim texts
    claim_texts = [c["text"] for c in claims_list]
    claim_vectors = model.encode(
        claim_texts, normalize_embeddings=True, show_progress_bar=False
    )
    claim_vectors = np.asarray(claim_vectors, dtype=np.float32)

    # Obtain chunk vectors for all retrieved chunks
    chunk_vectors: List[np.ndarray] = []
    for chunk in retrieved_chunks:
        c_vec = get_chunk_vector(chunk, cached_map=cached_map, model=model)
        chunk_vectors.append(c_vec)

    # Build similarity score matches for each claim
    processed_claims: List[Dict[str, Any]] = []

    for i, claim in enumerate(claims_list):
        c_vec = claim_vectors[i]
        evidence_matches: List[Dict[str, Any]] = []

        for j, chunk in enumerate(retrieved_chunks):
            chunk_vec = chunk_vectors[j]
            score = compute_cosine_similarity(c_vec, chunk_vec)
            in_context = chunk.get("chunk_id") in context_chunk_ids

            match_record = {
                "chunk_id": chunk.get("chunk_id", f"unknown_{j}"),
                "similarity_score": round(score, 4),
                "in_llm_context": in_context,
                "section": chunk.get("section", ""),
                "section_path": chunk.get("section_path", ""),
                "chapter": chunk.get("chapter", ""),
                "page_start": chunk.get("page_start", 0),
                "page_end": chunk.get("page_end", 0),
                "pages": chunk.get("pages", []),
                "rrf_score": chunk.get("rrf_score", 0.0),
            }
            evidence_matches.append(match_record)

        processed_claims.append(
            {
                "claim_id": claim["claim_id"],
                "sentence_index": claim["sentence_index"],
                "text": claim["text"],
                "evidence_matches": evidence_matches,
            }
        )

    output_data: Dict[str, Any] = {
        "query_id": query_id,
        "question": question,
        "answer": answer,
        "claims": processed_claims,
        "metadata": {
            "embedding_model": EMBED_MODEL_NAME,
            "similarity_metric": "cosine_similarity",
            "claim_count": len(processed_claims),
            "evaluated_chunks_count": len(retrieved_chunks),
        },
    }

    return output_data


def process_snapshot_file(
    snapshot_path: Union[str, Path],
    output_dir: Union[str, Path] = DEFAULT_SIMILARITY_OUTPUT_DIR,
    model: Any = None,
    cached_map: Optional[Dict[str, np.ndarray]] = None,
) -> Path:
    """
    Processes a single snapshot JSON file and writes the similarity scores to disk.
    """
    s_path = Path(snapshot_path).resolve()
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    snapshot = load_snapshot(s_path)
    result = compute_similarity_for_snapshot(
        snapshot=snapshot, model=model, cached_map=cached_map
    )

    out_file = out_dir / f"{s_path.stem}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logger.info(
        f"Processed {s_path.name} -> {result['metadata']['claim_count']} claims x "
        f"{result['metadata']['evaluated_chunks_count']} chunks -> Saved to {out_file}"
    )
    return out_file


def process_all_snapshots(
    snapshot_dir: Union[str, Path] = DEFAULT_SNAPSHOT_DIR,
    output_dir: Union[str, Path] = DEFAULT_SIMILARITY_OUTPUT_DIR,
) -> List[Path]:
    """
    Processes all snapshot JSON files in the specified snapshot directory.
    """
    snap_dir = Path(snapshot_dir).resolve()
    out_dir = Path(output_dir).resolve()

    if not snap_dir.exists():
        raise FileNotFoundError(f"Snapshot directory not found: {snap_dir}")

    snapshot_files = sorted(list(snap_dir.glob("*.json")))
    if not snapshot_files:
        logger.warning(f"No snapshot JSON files found in {snap_dir}")
        return []

    model = get_embedding_model()
    cached_map = load_cached_chunk_embeddings()

    generated_files: List[Path] = []
    for s_file in snapshot_files:
        out_path = process_snapshot_file(
            snapshot_path=s_file,
            output_dir=out_dir,
            model=model,
            cached_map=cached_map,
        )
        generated_files.append(out_path)

    return generated_files


def main():
    logger.info("Running Claim-to-Evidence Similarity Matcher...")
    generated = process_all_snapshots()
    logger.info(f"Successfully processed {len(generated)} snapshot(s).")


if __name__ == "__main__":
    main()
