"""
verification_1: Production-Ready ModernCE Complete-Answer Multi-Chunk Verifier

Evaluates whether a complete generated RAG answer is supported by retrieved textbook
evidence using long-context ModernBERT cross-encoder NLI (dleemiller/ModernCE-base-nli).

Architecture:
  Generated Answer + Retrieved Chunks
  1. Check for explicit abstention ("Not found in textbook") -> return NOT_FOUND status.
  2. Score individual chunks against complete answer via ModernCE.
  3. Rank chunks by entailment probability descending.
  4. Evaluate complete answer against Top-2 combined chunks.
  5. If Top-2 predicts ENTAILMENT -> return SUPPORTED (ENTAILED_BY_TOP2).
  6. Otherwise evaluate against Top-3 combined chunks as fallback.
  7. If Top-3 predicts ENTAILMENT -> return SUPPORTED (ENTAILED_BY_TOP3).
  8. Otherwise return INSUFFICIENT_EVIDENCE / CONTRADICTED (NOT_ENTAILED).
"""

import sys
import time
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure project root is in sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from verification_1.modern_nli_matcher import (
    MODERN_NLI_MODEL_NAME,
    MODERN_NLI_ID2LABEL,
    MODERN_NLI_LABEL_MAP,
    DEFAULT_MAX_LENGTH,
    get_modern_nli_model,
)

# ─────────────────────────────────────────────
# CONSTANTS & CONCEPTUAL STATUSES
# ─────────────────────────────────────────────

# User-facing conceptual verification statuses
STATUS_SUPPORTED = "SUPPORTED"
STATUS_INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
STATUS_CONTRADICTED = "CONTRADICTED"
STATUS_NOT_FOUND = "NOT_FOUND"

# Internal/detailed NLI verdicts
VERDICT_ENTAILED_BY_TOP2 = "ENTAILED_BY_TOP2"
VERDICT_ENTAILED_BY_TOP3 = "ENTAILED_BY_TOP3"
VERDICT_NOT_ENTAILED = "NOT_ENTAILED"
VERDICT_ABSTENTION = "ABSTENTION"


# ─────────────────────────────────────────────
# STRUCTURED DATA MODELS
# ─────────────────────────────────────────────

@dataclass
class ChunkScore:
    """Evaluation result for an individual retrieved evidence chunk."""
    chunk_number: int
    chunk_id: str
    section: str = "Unknown"
    pages: str = "?"
    hybrid_score: float = 0.0
    token_count: int = 0
    is_truncated: bool = False
    entailment_prob: float = 0.0
    neutral_prob: float = 0.0
    contradiction_prob: float = 0.0
    predicted_label: str = "neutral"
    eval_time_sec: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MultiChunkScore:
    """Evaluation result for a combined multi-chunk premise."""
    combination_type: str  # "TOP_2" or "TOP_3"
    selected_chunk_numbers: List[int]
    selected_chunk_ids: List[str]
    token_count: int = 0
    is_truncated: bool = False
    entailment_prob: float = 0.0
    neutral_prob: float = 0.0
    contradiction_prob: float = 0.0
    predicted_label: str = "neutral"
    eval_time_sec: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VerificationResult:
    """
    Complete structured verification output returned to callers.
    """
    status: str  # SUPPORTED, INSUFFICIENT_EVIDENCE, CONTRADICTED, NOT_FOUND
    verdict: str  # ENTAILED_BY_TOP2, ENTAILED_BY_TOP3, NOT_ENTAILED, ABSTENTION
    is_abstention: bool = False
    combination_used: Optional[str] = None  # "TOP_2", "TOP_3", or None
    entailment_prob: float = 0.0
    neutral_prob: float = 0.0
    contradiction_prob: float = 0.0
    predicted_label: str = "neutral"
    selected_chunk_numbers: List[int] = field(default_factory=list)
    selected_chunk_ids: List[str] = field(default_factory=list)
    individual_chunk_evaluations: List[Dict[str, Any]] = field(default_factory=list)
    ranked_chunk_numbers: List[int] = field(default_factory=list)
    top_2_evaluation: Optional[Dict[str, Any]] = None
    top_3_evaluation: Optional[Dict[str, Any]] = None
    total_nli_time_sec: float = 0.0
    explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────

def is_abstention_answer(answer_text: str) -> bool:
    """
    Detects whether an answer is an explicit abstention.
    Matches standard textbook out-of-scope response phrasings.
    """
    if not answer_text or not isinstance(answer_text, str):
        return False
    norm = answer_text.strip().lower()
    if "not found in the provided textbook" in norm:
        return True
    if len(answer_text.strip()) < 50 and "not found" in norm:
        return True
    return False


def _normalize_chunk(chunk: Any, index: int) -> Dict[str, Any]:
    """
    Normalizes a chunk object from various dictionary formats (ChromaDB, parsed txt, etc.)
    into a consistent dictionary.
    """
    if isinstance(chunk, str):
        return {
            "chunk_number": index + 1,
            "chunk_id": f"chunk_{index + 1}",
            "section": "Unknown",
            "pages": "?",
            "hybrid_score": 0.0,
            "text": chunk.strip()
        }

    if not isinstance(chunk, dict):
        return {
            "chunk_number": index + 1,
            "chunk_id": f"chunk_{index + 1}",
            "section": "Unknown",
            "pages": "?",
            "hybrid_score": 0.0,
            "text": str(chunk)
        }

    chunk_num = chunk.get("chunk_number")
    if chunk_num is None:
        chunk_num = index + 1

    chunk_id = chunk.get("chunk_id") or chunk.get("id") or f"chunk_{chunk_num}"
    section = chunk.get("section") or chunk.get("section_path") or "Unknown"
    
    pages = chunk.get("pages")
    if pages is None:
        p_start = chunk.get("page_start")
        p_end = chunk.get("page_end")
        if p_start is not None and p_end is not None:
            pages = f"{p_start}-{p_end}" if p_start != p_end else str(p_start)
        else:
            pages = "?"

    hybrid_score = float(chunk.get("hybrid_score") or chunk.get("rrf_score") or chunk.get("score") or 0.0)
    raw_text = chunk.get("text") or chunk.get("raw_text") or chunk.get("content") or ""

    return {
        "chunk_number": int(chunk_num),
        "chunk_id": str(chunk_id),
        "section": str(section),
        "pages": str(pages),
        "hybrid_score": round(hybrid_score, 4),
        "text": raw_text.strip()
    }


def _evaluate_pair(
    premise: str,
    hypothesis: str,
    tokenizer: Any,
    model: Any,
    label_map: Dict[str, int],
    max_length: int = DEFAULT_MAX_LENGTH,
) -> Dict[str, Any]:
    """Evaluates NLI for a single premise-hypothesis pair using ModernBERT."""
    import torch
    import numpy as np

    t0 = time.perf_counter()
    encoded = tokenizer(
        premise,
        hypothesis,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    token_count = int(encoded["input_ids"].shape[1])
    is_truncated = token_count >= max_length

    device = next(model.parameters()).device
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)
        logits = outputs.logits[0].cpu().numpy()
        probs = torch.softmax(outputs.logits[0], dim=-1).cpu().numpy()

    eval_time = time.perf_counter() - t0

    contra_idx = label_map["contradiction"]
    entail_idx = label_map["entailment"]
    neutral_idx = label_map["neutral"]

    contra_p = float(probs[contra_idx])
    entail_p = float(probs[entail_idx])
    neutral_p = float(probs[neutral_idx])

    pred_idx = int(np.argmax(probs))
    pred_label = MODERN_NLI_ID2LABEL.get(pred_idx, "unknown")

    return {
        "token_count": token_count,
        "is_truncated": is_truncated,
        "raw_logits": [round(float(x), 4) for x in logits],
        "entailment_prob": round(entail_p, 4),
        "neutral_prob": round(neutral_p, 4),
        "contradiction_prob": round(contra_p, 4),
        "predicted_label": pred_label,
        "eval_time_sec": round(eval_time, 4),
    }


# ─────────────────────────────────────────────
# MAIN VERIFIER CLASS
# ─────────────────────────────────────────────

class ModernCEVerifier:
    """
    Production verifier that uses ModernBERT Cross-Encoder NLI on complete generated answers
    and adaptive Top-2 / Top-3 evidence aggregation.
    """

    def __init__(
        self,
        model_bundle: Optional[Tuple[Any, Any, Dict[str, int]]] = None,
        max_length: int = DEFAULT_MAX_LENGTH,
    ):
        self.max_length = max_length
        if model_bundle is None:
            self.tokenizer, self.model, self.label_map = get_modern_nli_model()
        else:
            self.tokenizer, self.model, self.label_map = model_bundle

    def is_abstention(self, answer: str) -> bool:
        """Exposes abstention detection logic."""
        return is_abstention_answer(answer)

    def verify(
        self,
        answer: str,
        retrieved_chunks: List[Any],
        query: Optional[str] = None,
    ) -> VerificationResult:
        """
        Verifies a generated answer against retrieved evidence chunks.

        Args:
            answer: The complete original generated answer string.
            retrieved_chunks: List of retrieved chunk dictionaries or strings (typically 5 chunks).
            query: Optional user question for logging/context.

        Returns:
            Structured VerificationResult object.
        """
        t0_total = time.perf_counter()

        # -------------------------------------------------------------
        # 1. Check for Explicit Abstention
        # -------------------------------------------------------------
        if self.is_abstention(answer):
            total_time = time.perf_counter() - t0_total
            return VerificationResult(
                status=STATUS_NOT_FOUND,
                verdict=VERDICT_ABSTENTION,
                is_abstention=True,
                combination_used=None,
                entailment_prob=0.0,
                neutral_prob=0.0,
                contradiction_prob=0.0,
                predicted_label="abstention",
                selected_chunk_numbers=[],
                selected_chunk_ids=[],
                individual_chunk_evaluations=[],
                ranked_chunk_numbers=[],
                top_2_evaluation=None,
                top_3_evaluation=None,
                total_nli_time_sec=round(total_time, 4),
                explanation="Answer explicitly states that information was not found in the textbook. NLI verification against retrieved chunks was safely bypassed."
            )

        # -------------------------------------------------------------
        # 2. Validate Retrieved Chunks
        # -------------------------------------------------------------
        if not retrieved_chunks:
            total_time = time.perf_counter() - t0_total
            return VerificationResult(
                status=STATUS_INSUFFICIENT_EVIDENCE,
                verdict=VERDICT_NOT_ENTAILED,
                is_abstention=False,
                combination_used=None,
                entailment_prob=0.0,
                neutral_prob=1.0,
                contradiction_prob=0.0,
                predicted_label="neutral",
                selected_chunk_numbers=[],
                selected_chunk_ids=[],
                individual_chunk_evaluations=[],
                ranked_chunk_numbers=[],
                top_2_evaluation=None,
                top_3_evaluation=None,
                total_nli_time_sec=round(total_time, 4),
                explanation="No retrieved evidence chunks were provided for verification."
            )

        normalized_chunks = [_normalize_chunk(c, i) for i, c in enumerate(retrieved_chunks)]

        # -------------------------------------------------------------
        # 3. Individual Chunk Evaluations & Ranking
        # -------------------------------------------------------------
        individual_scores: List[ChunkScore] = []
        total_nli_time = 0.0

        for chunk in normalized_chunks:
            res = _evaluate_pair(
                premise=chunk["text"],
                hypothesis=answer,
                tokenizer=self.tokenizer,
                model=self.model,
                label_map=self.label_map,
                max_length=self.max_length,
            )
            total_nli_time += res["eval_time_sec"]

            c_score = ChunkScore(
                chunk_number=chunk["chunk_number"],
                chunk_id=chunk["chunk_id"],
                section=chunk["section"],
                pages=chunk["pages"],
                hybrid_score=chunk["hybrid_score"],
                token_count=res["token_count"],
                is_truncated=res["is_truncated"],
                entailment_prob=res["entailment_prob"],
                neutral_prob=res["neutral_prob"],
                contradiction_prob=res["contradiction_prob"],
                predicted_label=res["predicted_label"],
                eval_time_sec=res["eval_time_sec"],
            )
            individual_scores.append(c_score)

        # Rank chunks by entailment_prob descending (stable tie-break by chunk_number)
        ranked_chunks = sorted(
            individual_scores,
            key=lambda x: (x.entailment_prob, -x.chunk_number),
            reverse=True,
        )
        ranked_chunk_nums = [c.chunk_number for c in ranked_chunks]

        # -------------------------------------------------------------
        # 4. Top-2 Evidence Evaluation
        # -------------------------------------------------------------
        k_top2 = min(2, len(ranked_chunks))
        top2_candidates = ranked_chunks[:k_top2]
        # Preserve original retrieval order
        top2_ordered = sorted(top2_candidates, key=lambda x: x.chunk_number)
        top2_nums = [c.chunk_number for c in top2_ordered]
        top2_ids = [c.chunk_id for c in top2_ordered]

        top2_texts = []
        for c in top2_ordered:
            ch_raw = next(ch["text"] for ch in normalized_chunks if ch["chunk_number"] == c.chunk_number)
            top2_texts.append(ch_raw)
        top2_premise = "\n\n".join(top2_texts)

        res_top2 = _evaluate_pair(
            premise=top2_premise,
            hypothesis=answer,
            tokenizer=self.tokenizer,
            model=self.model,
            label_map=self.label_map,
            max_length=self.max_length,
        )
        total_nli_time += res_top2["eval_time_sec"]

        top2_obj = MultiChunkScore(
            combination_type="TOP_2",
            selected_chunk_numbers=top2_nums,
            selected_chunk_ids=top2_ids,
            token_count=res_top2["token_count"],
            is_truncated=res_top2["is_truncated"],
            entailment_prob=res_top2["entailment_prob"],
            neutral_prob=res_top2["neutral_prob"],
            contradiction_prob=res_top2["contradiction_prob"],
            predicted_label=res_top2["predicted_label"],
            eval_time_sec=res_top2["eval_time_sec"],
        )

        # STOP CONDITION: If Top-2 predicts ENTAILMENT -> Return SUPPORTED
        if top2_obj.predicted_label == "entailment":
            return VerificationResult(
                status=STATUS_SUPPORTED,
                verdict=VERDICT_ENTAILED_BY_TOP2,
                is_abstention=False,
                combination_used="TOP_2",
                entailment_prob=top2_obj.entailment_prob,
                neutral_prob=top2_obj.neutral_prob,
                contradiction_prob=top2_obj.contradiction_prob,
                predicted_label=top2_obj.predicted_label,
                selected_chunk_numbers=top2_obj.selected_chunk_numbers,
                selected_chunk_ids=top2_obj.selected_chunk_ids,
                individual_chunk_evaluations=[c.to_dict() for c in individual_scores],
                ranked_chunk_numbers=ranked_chunk_nums,
                top_2_evaluation=top2_obj.to_dict(),
                top_3_evaluation=None,
                total_nli_time_sec=round(total_nli_time, 4),
                explanation=f"Generated answer is strongly supported by Top-2 retrieved chunks {top2_nums} (Entailment: {top2_obj.entailment_prob:.4f})."
            )

        # -------------------------------------------------------------
        # 5. Top-3 Evidence Evaluation (Fallback)
        # -------------------------------------------------------------
        if len(ranked_chunks) < 3:
            # Not enough chunks for Top-3
            final_status = STATUS_CONTRADICTED if top2_obj.predicted_label == "contradiction" else STATUS_INSUFFICIENT_EVIDENCE
            return VerificationResult(
                status=final_status,
                verdict=VERDICT_NOT_ENTAILED,
                is_abstention=False,
                combination_used="TOP_2",
                entailment_prob=top2_obj.entailment_prob,
                neutral_prob=top2_obj.neutral_prob,
                contradiction_prob=top2_obj.contradiction_prob,
                predicted_label=top2_obj.predicted_label,
                selected_chunk_numbers=top2_obj.selected_chunk_numbers,
                selected_chunk_ids=top2_obj.selected_chunk_ids,
                individual_chunk_evaluations=[c.to_dict() for c in individual_scores],
                ranked_chunk_numbers=ranked_chunk_nums,
                top_2_evaluation=top2_obj.to_dict(),
                top_3_evaluation=None,
                total_nli_time_sec=round(total_nli_time, 4),
                explanation=f"Top-2 chunks were {top2_obj.predicted_label.upper()} (Entailment: {top2_obj.entailment_prob:.4f}). Insufficient additional chunks for Top-3."
            )

        k_top3 = 3
        top3_candidates = ranked_chunks[:k_top3]
        top3_ordered = sorted(top3_candidates, key=lambda x: x.chunk_number)
        top3_nums = [c.chunk_number for c in top3_ordered]
        top3_ids = [c.chunk_id for c in top3_ordered]

        top3_texts = []
        for c in top3_ordered:
            ch_raw = next(ch["text"] for ch in normalized_chunks if ch["chunk_number"] == c.chunk_number)
            top3_texts.append(ch_raw)
        top3_premise = "\n\n".join(top3_texts)

        res_top3 = _evaluate_pair(
            premise=top3_premise,
            hypothesis=answer,
            tokenizer=self.tokenizer,
            model=self.model,
            label_map=self.label_map,
            max_length=self.max_length,
        )
        total_nli_time += res_top3["eval_time_sec"]

        top3_obj = MultiChunkScore(
            combination_type="TOP_3",
            selected_chunk_numbers=top3_nums,
            selected_chunk_ids=top3_ids,
            token_count=res_top3["token_count"],
            is_truncated=res_top3["is_truncated"],
            entailment_prob=res_top3["entailment_prob"],
            neutral_prob=res_top3["neutral_prob"],
            contradiction_prob=res_top3["contradiction_prob"],
            predicted_label=res_top3["predicted_label"],
            eval_time_sec=res_top3["eval_time_sec"],
        )

        if top3_obj.predicted_label == "entailment":
            final_status = STATUS_SUPPORTED
            final_verdict = VERDICT_ENTAILED_BY_TOP3
            explanation = f"Generated answer required Top-3 chunks {top3_nums} to achieve full evidence support (Entailment: {top3_obj.entailment_prob:.4f})."
        elif top3_obj.predicted_label == "contradiction":
            final_status = STATUS_CONTRADICTED
            final_verdict = VERDICT_NOT_ENTAILED
            explanation = f"Generated answer conflicts with retrieved evidence in Top-3 chunks {top3_nums} (Contradiction: {top3_obj.contradiction_prob:.4f})."
        else:
            final_status = STATUS_INSUFFICIENT_EVIDENCE
            final_verdict = VERDICT_NOT_ENTAILED
            explanation = f"Retrieved evidence in Top-3 chunks {top3_nums} is insufficient to fully entail the answer (Neutral: {top3_obj.neutral_prob:.4f}, Entailment: {top3_obj.entailment_prob:.4f})."

        return VerificationResult(
            status=final_status,
            verdict=final_verdict,
            is_abstention=False,
            combination_used="TOP_3",
            entailment_prob=top3_obj.entailment_prob,
            neutral_prob=top3_obj.neutral_prob,
            contradiction_prob=top3_obj.contradiction_prob,
            predicted_label=top3_obj.predicted_label,
            selected_chunk_numbers=top3_obj.selected_chunk_numbers,
            selected_chunk_ids=top3_obj.selected_chunk_ids,
            individual_chunk_evaluations=[c.to_dict() for c in individual_scores],
            ranked_chunk_numbers=ranked_chunk_nums,
            top_2_evaluation=top2_obj.to_dict(),
            top_3_evaluation=top3_obj.to_dict(),
            total_nli_time_sec=round(total_nli_time, 4),
            explanation=explanation
        )


# Global singleton verifier instance
_GLOBAL_VERIFIER: Optional[ModernCEVerifier] = None


def get_verifier() -> ModernCEVerifier:
    """Returns a shared, lazily-initialized ModernCEVerifier instance."""
    global _GLOBAL_VERIFIER
    if _GLOBAL_VERIFIER is None:
        _GLOBAL_VERIFIER = ModernCEVerifier()
    return _GLOBAL_VERIFIER


def verify_answer(
    answer: str,
    retrieved_chunks: List[Any],
    query: Optional[str] = None,
    verifier: Optional[ModernCEVerifier] = None,
) -> VerificationResult:
    """
    Convenience function to verify a generated answer against retrieved chunks.
    """
    if verifier is None:
        verifier = get_verifier()
    return verifier.verify(answer=answer, retrieved_chunks=retrieved_chunks, query=query)


# Alias for verify_answer matching standard functional signatures
verify = verify_answer
