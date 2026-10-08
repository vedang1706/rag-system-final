"""
verification_1: ModernBERT-based Cross-Encoder NLI Matcher (Long-Context)

Evaluates directional natural language inference between retrieved evidence chunks (Premise)
and extracted atomic claims (Hypothesis) using dleemiller/ModernCE-base-nli.
Supports extended context lengths up to 2048 tokens (ModernBERT architecture).

IMPORTANT TECHNICAL NOTE ON LABEL MAPPING:
The model configuration on Hugging Face (config.json) contains a legacy tasksource id2label:
  {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
However, as documented in the author's official model card and verified empirically on
ground-truth NLI pairs, fine-tuning on sentence-transformers AllNLI produced the following mapping:
  Index 0 -> 'contradiction'
  Index 1 -> 'entailment'
  Index 2 -> 'neutral'
This module explicitly applies this verified label mapping.
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

MODERN_NLI_MODEL_NAME = "dleemiller/ModernCE-base-nli"
DEFAULT_MAX_LENGTH = 2048  # ModernBERT base supports extended context up to 2048/8192 tokens

DEFAULT_SNAPSHOT_DIR = Path(__file__).resolve().parent / "output" / "rag_snapshots"
DEFAULT_SIMILARITY_DIR = Path(__file__).resolve().parent / "output" / "similarity_scores"
DEFAULT_MODERN_NLI_OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "modern_nli_scores"

# Verified label mapping for dleemiller/ModernCE-base-nli
MODERN_NLI_LABEL_MAP: Dict[str, int] = {
    "contradiction": 0,
    "entailment": 1,
    "neutral": 2,
}
MODERN_NLI_ID2LABEL: Dict[int, str] = {
    0: "contradiction",
    1: "entailment",
    2: "neutral",
}

# Global model and tokenizer cache
_MODERN_NLI_BUNDLE: Optional[Tuple[Any, Any, Dict[str, int]]] = None


def get_modern_nli_model(
    model_name: str = MODERN_NLI_MODEL_NAME,
) -> Tuple[Any, Any, Dict[str, int]]:
    """
    Loads and caches the Hugging Face ModernBERT NLI tokenizer, model, and verified label mapping.

    Returns:
        A tuple of (tokenizer, model, label_map) where label_map maps
        'contradiction' -> 0, 'entailment' -> 1, 'neutral' -> 2.
    """
    global _MODERN_NLI_BUNDLE
    if _MODERN_NLI_BUNDLE is not None:
        return _MODERN_NLI_BUNDLE

    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
    except ImportError as exc:
        raise ImportError(
            "transformers is not installed. Please install it using: pip install transformers"
        ) from exc

    logger.info(f"Loading ModernBERT NLI model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()

    label_map = dict(MODERN_NLI_LABEL_MAP)
    _MODERN_NLI_BUNDLE = (tokenizer, model, label_map)
    return _MODERN_NLI_BUNDLE


def predict_pair(
    premise: str,
    hypothesis: str,
    model_bundle: Optional[Tuple[Any, Any, Dict[str, int]]] = None,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> Dict[str, Any]:
    """
    Evaluates NLI for a single (premise, hypothesis) pair.

    Args:
        premise: Evidence / context text (Premise).
        hypothesis: Claim text (Hypothesis).
        model_bundle: Optional pre-loaded model bundle.
        max_length: Maximum sequence length (default: 2048).

    Returns:
        Dictionary with logits, probabilities, predicted label, and token count.
    """
    if model_bundle is None:
        model_bundle = get_modern_nli_model()

    tokenizer, model, label_map = model_bundle

    encoded = tokenizer(
        premise,
        hypothesis,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    token_count = int(encoded["input_ids"].shape[1])

    with torch.no_grad():
        outputs = model(**encoded)
        logits = outputs.logits[0].cpu().numpy().tolist()
        probs = torch.softmax(outputs.logits[0], dim=-1).cpu().numpy().tolist()

    contra_idx = label_map["contradiction"]
    entail_idx = label_map["entailment"]
    neutral_idx = label_map["neutral"]

    contra_p = float(probs[contra_idx])
    entail_p = float(probs[entail_idx])
    neutral_p = float(probs[neutral_idx])

    # Determine predicted label from maximum probability
    pred_idx = int(torch.tensor(probs).argmax().item())
    pred_label = MODERN_NLI_ID2LABEL.get(pred_idx, "unknown")

    return {
        "premise": premise,
        "hypothesis": hypothesis,
        "token_count": token_count,
        "max_length": max_length,
        "raw_logits": [round(x, 4) for x in logits],
        "entailment_score": round(entail_p, 4),
        "neutral_score": round(neutral_p, 4),
        "contradiction_score": round(contra_p, 4),
        "predicted_label": pred_label,
        "probabilities": {
            "entailment": round(entail_p, 4),
            "neutral": round(neutral_p, 4),
            "contradiction": round(contra_p, 4),
        },
    }


def compute_modern_nli_scores_batch(
    pairs: List[Tuple[str, str]],
    model_bundle: Optional[Tuple[Any, Any, Dict[str, int]]] = None,
    batch_size: int = 16,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> List[Dict[str, float]]:
    """
    Computes raw NLI probabilities for a list of (premise, hypothesis) text pairs.

    Direction:
        Premise    = pair[0] (textbook chunk / evidence)
        Hypothesis = pair[1] (generated atomic claim)

    Args:
        pairs: List of (premise, hypothesis) tuples.
        model_bundle: Optional pre-loaded (tokenizer, model, label_map) tuple.
        batch_size: Inference batch size.
        max_length: Token truncation limit (default: 2048).

    Returns:
        List of dictionaries with keys: 'entailment_score', 'neutral_score', 'contradiction_score'.
    """
    if not pairs:
        return []

    if model_bundle is None:
        model_bundle = get_modern_nli_model()

    tokenizer, model, label_map = model_bundle
    ent_idx = label_map["entailment"]
    neu_idx = label_map["neutral"]
    con_idx = label_map["contradiction"]

    results: List[Dict[str, float]] = []

    for i in range(0, len(pairs), batch_size):
        batch_pairs = pairs[i : i + batch_size]

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


def compute_modern_nli_for_snapshot(
    snapshot: Dict[str, Any],
    model_bundle: Optional[Tuple[Any, Any, Dict[str, int]]] = None,
    similarity_dir: Optional[Union[str, Path]] = DEFAULT_SIMILARITY_DIR,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> Dict[str, Any]:
    """
    Computes raw ModernBERT NLI probabilities between all extracted claims and all retrieved evidence chunks.

    Args:
        snapshot: Validated RAG evidence snapshot dictionary.
        model_bundle: Optional pre-loaded NLI model bundle.
        similarity_dir: Optional path to Step 3 similarity scores for cross-referencing.
        max_length: Maximum sequence length (default: 2048).

    Returns:
        Structured dictionary conforming to the NLI score schema.
    """
    query_id = snapshot["query_id"]
    question = snapshot["question"]
    answer = snapshot["answer"]
    retrieved_chunks = snapshot.get("retrieved_chunks", [])
    context_chunks = snapshot.get("context_chunks", [])

    if model_bundle is None:
        model_bundle = get_modern_nli_model()

    context_chunk_ids = {c["chunk_id"] for c in context_chunks if "chunk_id" in c}

    # Extract claims using canonical claim_extractor
    claim_extraction = extract_claims(answer=answer, query_id=query_id)
    claims_list = claim_extraction.get("claims", [])

    if not claims_list:
        return {
            "query_id": query_id,
            "question": question,
            "answer": answer,
            "claims": [],
            "metadata": {
                "nli_model": MODERN_NLI_MODEL_NAME,
                "premise_source": "textbook_chunk",
                "hypothesis_source": "extracted_claim",
                "claim_count": 0,
                "evaluated_chunks_count": len(retrieved_chunks),
                "max_sequence_length": max_length,
            },
        }

    pair_records: List[Dict[str, Any]] = []
    text_pairs: List[Tuple[str, str]] = []

    for claim in claims_list:
        claim_id = claim["claim_id"]
        claim_text = claim["text"]

        for j, chunk in enumerate(retrieved_chunks):
            chunk_id = chunk.get("chunk_id", f"unknown_{j}")
            chunk_text = chunk.get("text", "")
            in_context = chunk_id in context_chunk_ids

            pair_records.append(
                {
                    "claim_id": claim_id,
                    "sentence_index": claim["sentence_index"],
                    "claim_text": claim_text,
                    "chunk_id": chunk_id,
                    "chunk_text": chunk_text,
                    "in_llm_context": in_context,
                    "similarity_score": None,
                    "section": chunk.get("section", ""),
                    "section_path": chunk.get("section_path", ""),
                    "chapter": chunk.get("chapter", ""),
                    "page_start": chunk.get("page_start", 0),
                    "page_end": chunk.get("page_end", 0),
                    "pages": chunk.get("pages", []),
                    "rrf_score": chunk.get("rrf_score", 0.0),
                }
            )
            text_pairs.append((chunk_text, claim_text))

    nli_scores = compute_modern_nli_scores_batch(
        pairs=text_pairs,
        model_bundle=model_bundle,
        batch_size=16,
        max_length=max_length,
    )

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
            "nli_model": MODERN_NLI_MODEL_NAME,
            "premise_source": "textbook_chunk",
            "hypothesis_source": "extracted_claim",
            "claim_count": len(processed_claims),
            "evaluated_chunks_count": len(retrieved_chunks),
            "max_sequence_length": max_length,
        },
    }

    return output_data


def process_snapshot_file(
    snapshot_path: Union[str, Path],
    output_dir: Union[str, Path] = DEFAULT_MODERN_NLI_OUTPUT_DIR,
    similarity_dir: Optional[Union[str, Path]] = DEFAULT_SIMILARITY_DIR,
    model_bundle: Optional[Tuple[Any, Any, Dict[str, int]]] = None,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> Path:
    """
    Processes a single snapshot JSON file and writes the ModernBERT NLI scores to disk.
    """
    s_path = Path(snapshot_path).resolve()
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    snapshot = load_snapshot(s_path)
    result = compute_modern_nli_for_snapshot(
        snapshot=snapshot,
        model_bundle=model_bundle,
        similarity_dir=similarity_dir,
        max_length=max_length,
    )

    out_file = out_dir / f"{s_path.stem}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logger.info(
        f"Processed {s_path.name} -> {result['metadata']['claim_count']} claims x "
        f"{result['metadata']['evaluated_chunks_count']} chunks -> Saved to {out_file}"
    )
    return out_file
