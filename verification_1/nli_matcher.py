"""
verification_1: Claim-to-Evidence Natural Language Inference (NLI) Matcher

Evaluates directional natural language inference between retrieved evidence chunks (Premise)
and extracted sentence-level claims (Hypothesis) using a cross-encoder NLI model (cross-encoder/nli-deberta-v3-base).
Produces raw probabilities for entailment, neutral, and contradiction without applying thresholds or verdicts.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import logging
import torch

# Ensure project root is on sys.path for direct script execution
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from verification_1.claim_extractor import extract_claims
from verification_1.context_snapshot import load_snapshot

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-base"
DEFAULT_SNAPSHOT_DIR = Path(__file__).resolve().parent / "output" / "rag_snapshots"
DEFAULT_SIMILARITY_DIR = Path(__file__).resolve().parent / "output" / "similarity_scores"
DEFAULT_NLI_OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "nli_scores"

# Global model and tokenizer cache
_NLI_MODEL_BUNDLE: Optional[Tuple[Any, Any, Dict[str, int]]] = None


def get_nli_model(model_name: str = NLI_MODEL_NAME) -> Tuple[Any, Any, Dict[str, int]]:
    """
    Loads and caches the Hugging Face DeBERTa NLI tokenizer, model, and label mapping.
    
    Returns:
        A tuple of (tokenizer, model, label_indices_map) where label_indices_map
        maps 'entailment', 'neutral', and 'contradiction' to their corresponding logit indices.
    """
    global _NLI_MODEL_BUNDLE
    if _NLI_MODEL_BUNDLE is not None:
        return _NLI_MODEL_BUNDLE

    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
    except ImportError as exc:
        raise ImportError(
            "transformers is not installed. Please install it using: pip install transformers"
        ) from exc

    logger.info(f"Loading NLI model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()

    # Determine dynamic label mapping from model config
    id2label = model.config.id2label
    label_map: Dict[str, int] = {}
    for idx, label_name in id2label.items():
        name_lower = str(label_name).lower()
        if "entail" in name_lower:
            label_map["entailment"] = int(idx)
        elif "contra" in name_lower:
            label_map["contradiction"] = int(idx)
        elif "neut" in name_lower:
            label_map["neutral"] = int(idx)

    required_labels = {"entailment", "neutral", "contradiction"}
    if not required_labels.issubset(label_map.keys()):
        raise ValueError(
            f"Could not resolve all 3 NLI labels from model config. Found: {label_map} in id2label: {id2label}"
        )

    _NLI_MODEL_BUNDLE = (tokenizer, model, label_map)
    return _NLI_MODEL_BUNDLE


def compute_nli_scores_batch(
    pairs: List[Tuple[str, str]],
    model_bundle: Optional[Tuple[Any, Any, Dict[str, int]]] = None,
    batch_size: int = 16,
    max_length: int = 512,
) -> List[Dict[str, float]]:
    """
    Computes raw NLI probabilities for a list of (premise, hypothesis) text pairs.

    Direction:
        Premise    = pair[0] (textbook chunk)
        Hypothesis = pair[1] (generated claim)

    Args:
        pairs: List of (premise, hypothesis) tuples.
        model_bundle: Optional pre-loaded (tokenizer, model, label_map) tuple.
        batch_size: Inference batch size.
        max_length: Token truncation limit.

    Returns:
        List of dictionaries with keys: 'entailment_score', 'neutral_score', 'contradiction_score'.
    """
    if not pairs:
        return []

    if model_bundle is None:
        model_bundle = get_nli_model()

    tokenizer, model, label_map = model_bundle
    ent_idx = label_map["entailment"]
    neu_idx = label_map["neutral"]
    con_idx = label_map["contradiction"]

    results: List[Dict[str, float]] = []

    for i in range(0, len(pairs), batch_size):
        batch_pairs = pairs[i : i + batch_size]
        
        # Tokenize (Premise = text_a, Hypothesis = text_b)
        encoded = tokenizer(
            batch_pairs,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = model(**encoded)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

        for row in probs:
            ent_p = float(row[ent_idx])
            neu_p = float(row[neu_idx])
            con_p = float(row[con_idx])
            results.append(
                {
                    "entailment_score": round(ent_p, 4),
                    "neutral_score": round(neu_p, 4),
                    "contradiction_score": round(con_p, 4),
                }
            )

    return results


def load_similarity_map_for_snapshot(
    query_id: str, similarity_dir: Union[str, Path] = DEFAULT_SIMILARITY_DIR
) -> Dict[Tuple[str, str], float]:
    """
    Loads precomputed similarity scores from Step 3 if available.
    Returns mapping: (claim_id, chunk_id) -> similarity_score.
    """
    sim_path = Path(similarity_dir).resolve() / f"{query_id}.json"
    if not sim_path.exists():
        return {}

    try:
        with open(sim_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        mapping: Dict[Tuple[str, str], float] = {}
        for claim in data.get("claims", []):
            cid = claim.get("claim_id")
            for match in claim.get("evidence_matches", []):
                chunk_id = match.get("chunk_id")
                sim_score = match.get("similarity_score")
                if cid and chunk_id and sim_score is not None:
                    mapping[(cid, chunk_id)] = float(sim_score)
        return mapping
    except Exception as e:
        logger.warning(f"Could not load similarity scores for {query_id}: {e}")
        return {}


def compute_nli_for_snapshot(
    snapshot: Dict[str, Any],
    model_bundle: Optional[Tuple[Any, Any, Dict[str, int]]] = None,
    similarity_dir: Optional[Union[str, Path]] = DEFAULT_SIMILARITY_DIR,
) -> Dict[str, Any]:
    """
    Computes raw NLI probabilities between all extracted claims and all retrieved evidence chunks.

    Direction:
        Premise    = retrieved chunk text
        Hypothesis = extracted claim text

    Args:
        snapshot: Validated RAG evidence snapshot dictionary.
        model_bundle: Optional pre-loaded NLI model bundle.
        similarity_dir: Optional path to Step 3 similarity scores for cross-referencing.

    Returns:
        Structured dictionary conforming to the NLI score schema.
    """
    query_id = snapshot["query_id"]
    question = snapshot["question"]
    answer = snapshot["answer"]
    retrieved_chunks = snapshot.get("retrieved_chunks", [])
    context_chunks = snapshot.get("context_chunks", [])

    if model_bundle is None:
        model_bundle = get_nli_model()

    # Determine context chunk ID set for LLM visibility tracking
    context_chunk_ids = {c["chunk_id"] for c in context_chunks if "chunk_id" in c}

    # Load Step 3 similarity scores if available
    sim_map = (
        load_similarity_map_for_snapshot(query_id, similarity_dir)
        if similarity_dir
        else {}
    )

    # Extract sentence-level claims
    claim_extraction = extract_claims(answer=answer, query_id=query_id)
    claims_list = claim_extraction.get("claims", [])

    if not claims_list:
        return {
            "query_id": query_id,
            "question": question,
            "answer": answer,
            "claims": [],
            "metadata": {
                "nli_model": NLI_MODEL_NAME,
                "premise_source": "textbook_chunk",
                "hypothesis_source": "extracted_claim",
                "claim_count": 0,
                "evaluated_chunks_count": len(retrieved_chunks),
            },
        }

    # Build all (Premise = chunk_text, Hypothesis = claim_text) pairs
    pair_records: List[Dict[str, Any]] = []
    text_pairs: List[Tuple[str, str]] = []

    for claim in claims_list:
        claim_id = claim["claim_id"]
        claim_text = claim["text"]

        for j, chunk in enumerate(retrieved_chunks):
            chunk_id = chunk.get("chunk_id", f"unknown_{j}")
            chunk_text = chunk.get("text", "")
            in_context = chunk_id in context_chunk_ids
            sim_score = sim_map.get((claim_id, chunk_id))

            pair_records.append(
                {
                    "claim_id": claim_id,
                    "sentence_index": claim["sentence_index"],
                    "claim_text": claim_text,
                    "chunk_id": chunk_id,
                    "chunk_text": chunk_text,
                    "in_llm_context": in_context,
                    "similarity_score": sim_score,
                    "section": chunk.get("section", ""),
                    "section_path": chunk.get("section_path", ""),
                    "chapter": chunk.get("chapter", ""),
                    "page_start": chunk.get("page_start", 0),
                    "page_end": chunk.get("page_end", 0),
                    "pages": chunk.get("pages", []),
                    "rrf_score": chunk.get("rrf_score", 0.0),
                }
            )
            # Premise = chunk_text, Hypothesis = claim_text
            text_pairs.append((chunk_text, claim_text))

    # Batch compute NLI inference
    nli_scores = compute_nli_scores_batch(
        pairs=text_pairs, model_bundle=model_bundle, batch_size=16
    )

    # Group matches by claim
    claims_dict: Dict[str, Dict[str, Any]] = {}
    for claim in claims_list:
        claims_dict[claim["claim_id"]] = {
            "claim_id": claim["claim_id"],
            "sentence_index": claim["sentence_index"],
            "text": claim["text"],
            "evidence_matches": [],
        }

    for record, score in zip(pair_records, nli_scores):
        match_entry = {
            "claim_id": record["claim_id"],
            "chunk_id": record["chunk_id"],
            "claim_text": record["claim_text"],
            "chunk_text": record["chunk_text"],
            "in_llm_context": record["in_llm_context"],
            "similarity_score": record["similarity_score"],
            "entailment_score": score["entailment_score"],
            "neutral_score": score["neutral_score"],
            "contradiction_score": score["contradiction_score"],
            "section": record["section"],
            "section_path": record["section_path"],
            "chapter": record["chapter"],
            "page_start": record["page_start"],
            "page_end": record["page_end"],
            "pages": record["pages"],
            "rrf_score": record["rrf_score"],
        }
        claims_dict[record["claim_id"]]["evidence_matches"].append(match_entry)

    processed_claims = list(claims_dict.values())

    output_data: Dict[str, Any] = {
        "query_id": query_id,
        "question": question,
        "answer": answer,
        "claims": processed_claims,
        "metadata": {
            "nli_model": NLI_MODEL_NAME,
            "premise_source": "textbook_chunk",
            "hypothesis_source": "extracted_claim",
            "claim_count": len(processed_claims),
            "evaluated_chunks_count": len(retrieved_chunks),
        },
    }

    return output_data


def process_snapshot_file(
    snapshot_path: Union[str, Path],
    output_dir: Union[str, Path] = DEFAULT_NLI_OUTPUT_DIR,
    similarity_dir: Optional[Union[str, Path]] = DEFAULT_SIMILARITY_DIR,
    model_bundle: Optional[Tuple[Any, Any, Dict[str, int]]] = None,
) -> Path:
    """
    Processes a single snapshot JSON file and writes the NLI scores to disk.
    """
    s_path = Path(snapshot_path).resolve()
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    snapshot = load_snapshot(s_path)
    result = compute_nli_for_snapshot(
        snapshot=snapshot, model_bundle=model_bundle, similarity_dir=similarity_dir
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
    output_dir: Union[str, Path] = DEFAULT_NLI_OUTPUT_DIR,
    similarity_dir: Optional[Union[str, Path]] = DEFAULT_SIMILARITY_DIR,
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

    model_bundle = get_nli_model()

    generated_files: List[Path] = []
    for s_file in snapshot_files:
        out_path = process_snapshot_file(
            snapshot_path=s_file,
            output_dir=out_dir,
            similarity_dir=similarity_dir,
            model_bundle=model_bundle,
        )
        generated_files.append(out_path)

    return generated_files


def main():
    logger.info("Running Claim-to-Evidence NLI Matcher...")
    generated = process_all_snapshots()
    logger.info(f"Successfully processed {len(generated)} snapshot(s).")


if __name__ == "__main__":
    main()
