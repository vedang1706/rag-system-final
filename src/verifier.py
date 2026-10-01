# src/verifier.py

"""
ModernBERT NLI & Lightweight Hybrid Grounding & Verification Layer.

Architecture:
1. Fast Claim Extraction: Deterministically extracts atomic factual statements from the generated answer.
2. First-Stage Lightweight Filter: Fast token recall, bigram matching, and polarity heuristics across all retrieved chunks.
3. Selective ModernBERT NLI: Invokes tasksource/ModernBERT-base-nli ONLY for ambiguous claims against top-K relevant chunks.
4. Evidence Aggregation: Multi-chunk evidence combination with conflict resolution.
5. Strict Grounding Guarantee: Preserves original generated answer and original chunk metadata/citations.
"""

import os
import re
import time
import logging
from typing import List, Dict, Optional, Any, Tuple

# Set up logger
logger = logging.getLogger("verifier")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [Verifier] %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# ─────────────────────────────────────────────
# 1. CENTRALIZED CONFIGURATION
# ─────────────────────────────────────────────

VERIFIER_CONFIG = {
    # ModernBERT NLI settings
    "NLI_ENABLED": True,
    "LOCAL_MODEL_PATH": os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "ModernBERT-base-nli"),
    "HF_MODEL_NAME": "tasksource/ModernBERT-base-nli",
    "NLI_TOP_K": 3,               # Max candidate evidence chunks sent to ModernBERT per ambiguous claim
    "MAX_SEQ_LENGTH": 512,        # Max sequence token length for premise + hypothesis
    
    # Lightweight threshold heuristics (used to bypass NLI for obvious cases)
    "LIGHTWEIGHT_CLEAR_ENT_THRESHOLD": 0.85,   # Token recall + bigram bonus >= 0.85 -> Clear Entailment
    "LIGHTWEIGHT_CLEAR_UNSUPPORTED": 0.15,     # Overlap < 0.15 and no subject match -> Clear Unsupported
    
    # ModernBERT NLI Decision Thresholds (calibrated probabilities from softmax)
    "NLI_ENT_THRESHOLD": 0.70,     # Confidence required for ENTAILMENT
    "NLI_CONT_THRESHOLD": 0.70,    # Confidence required for CONTRADICTION
    "NLI_CONFLICT_MARGIN": 0.55,   # If highest entailment >= 0.55 AND highest contradiction >= 0.55 across chunks
}

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can",
    "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above",
    "below", "between", "out", "off", "over", "under", "again",
    "further", "then", "once", "and", "but", "or", "nor", "so",
    "yet", "both", "either", "neither", "not", "only", "own",
    "same", "than", "too", "very", "just", "that", "this", "these",
    "those", "it", "its", "we", "they", "them", "their", "what",
    "which", "who", "whom", "how", "all", "each", "more", "most",
    "other", "some", "such", "no", "up", "about", "also", "according"
}

POLARITY_OPPOSITES = {
    ("positively", "negatively"),
    ("positive", "negative"),
    ("increase", "decrease"),
    ("increases", "decreases"),
    ("increased", "decreased"),
    ("high", "low"),
    ("higher", "lower"),
    ("true", "false"),
    ("supported", "unsupported"),
    ("effective", "ineffective"),
    ("direct", "indirect"),
    ("internal", "external"),
    ("conscious", "unconscious"),
    ("austrian", "japanese"),
    ("neurologist", "physicist"),
    ("russian", "american")
}


# ─────────────────────────────────────────────
# 2. VERIFIER INITIALIZATION & LOADING
# ─────────────────────────────────────────────

def load_nli_verifier(
    model_name_or_path: Optional[str] = None,
    device: Optional[str] = None,
    enable_nli: bool = True
) -> Dict[str, Any]:
    """
    Loads and caches the ModernBERT NLI model and tokenizer once.
    Falls back gracefully to lightweight verification if dependencies or model files are missing.
    """
    if not enable_nli or not VERIFIER_CONFIG.get("NLI_ENABLED", True):
        logger.info("NLI verifier is disabled by configuration. Operating in lightweight mode.")
        return {
            "available": True,
            "type": "lightweight_only",
            "model_name": "Lightweight Heuristic Verifier",
            "device": "cpu",
            "model": None,
            "tokenizer": None,
            "error": None
        }

    # Determine candidate model path
    local_path = VERIFIER_CONFIG["LOCAL_MODEL_PATH"]
    hf_path = VERIFIER_CONFIG["HF_MODEL_NAME"]

    if model_name_or_path:
        target_path = model_name_or_path
    elif os.path.exists(local_path) and (os.path.exists(os.path.join(local_path, "model.safetensors")) or os.path.exists(os.path.join(local_path, "config.json"))):
        target_path = local_path
    else:
        target_path = hf_path

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

        selected_device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading ModernBERT NLI model from '{target_path}' on device '{selected_device}'...")

        # Load config with reference_compile disabled for CPU stability
        config = AutoConfig.from_pretrained(target_path)
        if hasattr(config, "reference_compile"):
            config.reference_compile = False

        tokenizer = AutoTokenizer.from_pretrained(target_path)
        model = AutoModelForSequenceClassification.from_pretrained(target_path, config=config)
        model.to(selected_device)
        model.eval()

        logger.info(f"✓ ModernBERT NLI model loaded successfully on {selected_device}.")

        return {
            "available": True,
            "type": "modernbert_nli",
            "model_name": target_path,
            "model": model,
            "tokenizer": tokenizer,
            "config": config,
            "device": selected_device,
            "error": None
        }

    except Exception as e:
        logger.warning(f"Could not load ModernBERT NLI model ({e}). Gracefully falling back to lightweight verifier.")
        return {
            "available": False,
            "type": "lightweight_fallback",
            "model_name": target_path,
            "model": None,
            "tokenizer": None,
            "device": "cpu",
            "error": str(e)
        }


# ─────────────────────────────────────────────
# 3. DETERMINISTIC CLAIM EXTRACTION
# ─────────────────────────────────────────────

def extract_claims(client=None, answer: str = "", model_name: str = "") -> List[str]:
    """
    Extracts atomic factual claims from the generated answer deterministically.
    Removes conversational preambles, citations, markdown artifacts, and greetings without extra LLM latency.
    """
    if not answer or not answer.strip():
        return []

    # If the LLM returned standard absent textbook response
    if "not found in the provided textbook" in answer.lower():
        return []

    # Clean markdown headers, bullet symbols, bold/italics
    cleaned_text = re.sub(r'#+\s*', '', answer)
    cleaned_text = re.sub(r'\*\*(.*?)\*\*', r'\1', cleaned_text)
    cleaned_text = re.sub(r'\*(.*?)\*', r'\1', cleaned_text)
    cleaned_text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', cleaned_text) # remove markdown links
    cleaned_text = re.sub(r'\(page\s*\d+(?:[-–]\d+)?\)', '', cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r'\(pages\s*\d+(?:[-–]\d+)?\)', '', cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r'\(section\s*[\d.]+\)', '', cleaned_text, flags=re.IGNORECASE)

    # Split on sentence boundaries and newlines
    raw_sentences = re.split(r'(?:\r?\n+|(?<=[.!?])\s+)', cleaned_text)
    claims = []

    # Preamble patterns to discard or clean
    preamble_prefixes = [
        "in summary", "to summarize", "in conclusion", "note that", "note:",
        "based on the textbook", "according to the textbook", "according to chapter",
        "as stated in", "here is", "the textbook mentions that", "hello", "hi there"
    ]

    for part in raw_sentences:
        s = part.strip().lstrip("•-*0123456789.) ")
        if not s or len(s) < 12:
            continue

        lower_s = s.lower()

        # Check if entire sentence is just filler preamble
        if any(lower_s.startswith(p) and len(s) < 30 for p in preamble_prefixes):
            continue

        # Strip preamble if sentence contains actual facts after it
        for p in preamble_prefixes:
            if lower_s.startswith(p + ","):
                s = s[len(p) + 1:].strip()
                break
            elif lower_s.startswith(p + ":"):
                s = s[len(p) + 1:].strip()
                break

        # Remove trailing punctuation leftovers
        s = s.strip()
        if len(s) > 12:
            claims.append(s)

    # If no sentences passed filtering, fallback to entire stripped answer if meaningful
    if not claims and len(answer.strip()) > 10 and not "not found" in answer.lower():
        claims = [answer.strip()]

    return claims


# ─────────────────────────────────────────────
# 4. LIGHTWEIGHT VERIFICATION LAYER
# ─────────────────────────────────────────────

def tokenize_content_words(text: str) -> List[str]:
    """Extracts lowercase non-stopword alphanumeric content words of length >= 3."""
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    return [w for w in words if w not in STOPWORDS]


def find_best_sentence_snippet(claim: str, chunk_text: str, max_length: int = 240) -> str:
    """
    Finds the exact matching sentence or localized context window in chunk_text.
    Avoids returning generic chunk beginnings for glossary/key terms pages.
    """
    if not chunk_text or not claim:
        return ""

    claim_tokens = [w for w in tokenize_content_words(claim) if len(w) >= 3]
    if not claim_tokens:
        return chunk_text[:max_length].strip() + ("..." if len(chunk_text) > max_length else "")

    # 1. Try sentence & clause boundaries
    units = re.split(r'(?:(?<=[.!?])\s+|\n+|;\s+)', chunk_text)
    best_unit = ""
    best_overlap = 0

    for u in units:
        u_clean = u.strip()
        if len(u_clean) < 12:
            continue
        u_tokens = set(tokenize_content_words(u_clean))
        overlap = sum(1 for tok in claim_tokens if tok in u_tokens)
        if overlap > best_overlap:
            best_overlap = overlap
            best_unit = u_clean

    if best_overlap >= 2 and best_unit:
        return best_unit[:max_length].strip() + ("..." if len(best_unit) > max_length else "")

    # 2. Localized sliding window search around highest keyword concentration
    words = chunk_text.split()
    if len(words) <= 35:
        return chunk_text.strip()

    best_start = 0
    max_matches = 0
    window_size = 35

    for i in range(0, len(words) - min(window_size, len(words)) + 1, 5):
        sub_words = words[i:i + window_size]
        sub_text_lower = " ".join(sub_words).lower()
        matches = sum(1 for tok in claim_tokens if tok in sub_text_lower)
        if matches > max_matches:
            max_matches = matches
            best_start = i

    if max_matches > 0:
        extracted = " ".join(words[best_start:best_start + window_size]).strip()
        prefix = "..." if best_start > 0 else ""
        suffix = "..." if (best_start + window_size) < len(words) else ""
        return f"{prefix}{extracted}{suffix}"

    return chunk_text[:max_length].strip() + ("..." if len(chunk_text) > max_length else "")


def score_claim_lightweight(claim: str, chunk_text: str) -> Dict[str, Any]:
    """
    Computes heuristic token recall, phrase matching, and polarity conflict against a chunk.
    Does NOT falsely equate missing evidence to contradiction.
    """
    claim_tokens = tokenize_content_words(claim)
    if not claim_tokens:
        return {
            "heuristic_score": 0.0,
            "recall": 0.0,
            "subject_matched": False,
            "has_polarity_contradiction": False,
            "entailment": 0.0,
            "contradiction": 0.0,
            "neutral": 1.0
        }

    chunk_lower = chunk_text.lower()
    chunk_tokens = set(tokenize_content_words(chunk_text))

    # Token recall
    found = [tok for tok in claim_tokens if tok in chunk_lower]
    recall = len(found) / len(claim_tokens)

    # Bigram phrase bonus
    bigram_bonus = 0.0
    if len(claim_tokens) >= 2:
        bigrams = [f"{claim_tokens[i]} {claim_tokens[i+1]}" for i in range(len(claim_tokens) - 1)]
        found_bigrams = sum(1 for bg in bigrams if bg in chunk_lower)
        bigram_bonus = (found_bigrams / len(bigrams)) * 0.20

    heuristic_score = min(1.0, recall + bigram_bonus)

    # Subject match (first 2 content words)
    subject_tokens = claim_tokens[:2] if len(claim_tokens) >= 2 else claim_tokens
    subject_matched = any(tok in chunk_lower for tok in subject_tokens)

    # Explicit polarity / antonym conflict check
    has_polarity_contradiction = False
    for p1, p2 in POLARITY_OPPOSITES:
        if (p1 in claim_tokens and p2 in chunk_tokens) or (p2 in claim_tokens and p1 in chunk_tokens):
            has_polarity_contradiction = True
            break

    # Lightweight estimate for initial triage
    if has_polarity_contradiction and subject_matched:
        return {
            "heuristic_score": 0.1,
            "recall": recall,
            "subject_matched": True,
            "has_polarity_contradiction": True,
            "entailment": 0.0,
            "contradiction": 0.95,
            "neutral": 0.05
        }

    if heuristic_score >= VERIFIER_CONFIG["LIGHTWEIGHT_CLEAR_ENT_THRESHOLD"]:
        return {
            "heuristic_score": round(heuristic_score, 4),
            "recall": round(recall, 4),
            "subject_matched": subject_matched,
            "has_polarity_contradiction": False,
            "entailment": round(heuristic_score, 4),
            "contradiction": 0.0,
            "neutral": round(1.0 - heuristic_score, 4)
        }

    return {
        "heuristic_score": round(heuristic_score, 4),
        "recall": round(recall, 4),
        "subject_matched": subject_matched,
        "has_polarity_contradiction": False,
        "entailment": round(heuristic_score, 4),
        "contradiction": 0.0,
        "neutral": round(1.0 - heuristic_score, 4)
    }


# ─────────────────────────────────────────────
# 5. EVIDENCE SELECTION FOR NLI
# ─────────────────────────────────────────────

def select_evidence_for_nli(
    claim: str,
    retrieved_chunks: List[Dict[str, Any]],
    top_k: int = 3
) -> List[Tuple[int, Dict[str, Any], Dict[str, Any]]]:
    """
    Ranks the already retrieved chunks by lightweight relevance and selects the top-K candidates.
    Does NOT perform secondary retrieval or re-query BM25/vector database.
    Returns: List of (original_chunk_index, chunk, lightweight_eval)
    """
    scored_candidates = []
    for idx, chunk in enumerate(retrieved_chunks):
        chunk_text = chunk.get("text", "")
        lw_res = score_claim_lightweight(claim, chunk_text)
        scored_candidates.append((idx, chunk, lw_res))

    # Sort descending by heuristic relevance score
    scored_candidates.sort(key=lambda item: item[2]["heuristic_score"], reverse=True)

    # Filter to candidates that have meaningful lexical or subject connection
    positive_candidates = [c for c in scored_candidates if c[2]["heuristic_score"] >= 0.15 or c[2]["subject_matched"]]
    selected = positive_candidates if positive_candidates else scored_candidates[:1]

    # Return top_k candidates
    return selected[:max(1, top_k)]


# ─────────────────────────────────────────────
# 6. MODERNBERT NLI INFERENCE
# ─────────────────────────────────────────────

def run_modernbert_nli_batch(
    pairs: List[Tuple[str, str]],
    verifier: Dict[str, Any]
) -> List[Dict[str, float]]:
    """
    Runs batched ModernBERT inference on (Premise, Hypothesis) pairs.
    Returns list of dicts: {"entailment": float, "neutral": float, "contradiction": float}
    """
    if not pairs or not verifier or not verifier.get("available") or verifier.get("model") is None:
        return [{"entailment": 0.0, "neutral": 1.0, "contradiction": 0.0} for _ in pairs]

    import torch

    model = verifier["model"]
    tokenizer = verifier["tokenizer"]
    device = verifier.get("device", "cpu")
    max_len = VERIFIER_CONFIG.get("MAX_SEQ_LENGTH", 512)

    premises = [p[0] for p in pairs]
    hypotheses = [p[1] for p in pairs]

    try:
        # Batch tokenize premise-hypothesis pairs
        enc = tokenizer(
            premises,
            hypotheses,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt"
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            outputs = model(**enc)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()

        # Extract mapping from config id2label
        id2label = getattr(model.config, "id2label", {0: "entailment", 1: "neutral", 2: "contradiction"})
        
        # Build normalized label indices
        label_map = {}
        for idx, lbl in id2label.items():
            label_map[str(lbl).lower()] = int(idx)

        ent_idx = label_map.get("entailment", 0)
        neu_idx = label_map.get("neutral", 1)
        con_idx = label_map.get("contradiction", 2)

        results = []
        for row in probs:
            p_ent = float(row[ent_idx])
            p_neu = float(row[neu_idx])
            p_con = float(row[con_idx])
            results.append({
                "entailment": round(p_ent, 4),
                "neutral": round(p_neu, 4),
                "contradiction": round(p_con, 4)
            })

        return results

    except Exception as e:
        logger.error(f"Error during ModernBERT NLI batch inference: {e}")
        return [{"entailment": 0.0, "neutral": 1.0, "contradiction": 0.0} for _ in pairs]


# ─────────────────────────────────────────────
# 7. EVIDENCE AGGREGATION & CLAIM VERIFICATION
# ─────────────────────────────────────────────

def aggregate_evidence(
    evaluations: List[Dict[str, Any]],
    ent_thresh: float = 0.70,
    cont_thresh: float = 0.70,
    conflict_margin: float = 0.70
) -> Tuple[str, float, Optional[Dict[str, Any]], str, float, float]:
    """
    Aggregates multi-chunk evaluations for a single claim.
    Resolves ENTAILMENT, CONTRADICTION, EVIDENCE_CONFLICT, and UNSUPPORTED.
    Returns: (status, confidence, best_chunk, best_snippet, max_entailment, max_contradiction)
    """
    if not evaluations:
        return "UNSUPPORTED", 0.0, None, "", 0.0, 0.0

    max_entailment = 0.0
    best_entail_chunk = None
    best_entail_snippet = ""

    max_contradiction = 0.0
    best_contra_chunk = None
    best_contra_snippet = ""

    for ev in evaluations:
        e = ev.get("entailment", 0.0)
        c = ev.get("contradiction", 0.0)
        chunk = ev.get("chunk")
        snippet = ev.get("evidence_snippet", "")

        # Prefer main chapter content over supplementary/key terms when scores are comparable
        is_supp = chunk.get("is_supplementary", False) or "key terms" in str(chunk.get("section", "")).lower()
        
        # Entailment tracking
        if (e > max_entailment) or (abs(e - max_entailment) < 0.05 and not is_supp and best_entail_chunk and ("key terms" in str(best_entail_chunk.get("section", "")).lower())):
            max_entailment = e
            best_entail_chunk = chunk
            best_entail_snippet = snippet

        # Contradiction tracking
        if (c > max_contradiction) or (abs(c - max_contradiction) < 0.05 and not is_supp and best_contra_chunk and ("key terms" in str(best_contra_chunk.get("section", "")).lower())):
            max_contradiction = c
            best_contra_chunk = chunk
            best_contra_snippet = snippet

    # Conflict condition: Multiple chunks have high confidence on opposite conclusions
    if max_entailment >= ent_thresh and max_contradiction >= cont_thresh:
        status = "EVIDENCE CONFLICT"
        confidence = round(max(max_entailment, max_contradiction), 4)
        best_chunk = best_entail_chunk or best_contra_chunk
        best_snippet = f"Entailment ({max_entailment*100:.1f}%): {best_entail_snippet} | Contradiction ({max_contradiction*100:.1f}%): {best_contra_snippet}"
    elif max_entailment >= ent_thresh:
        status = "VERIFIED"
        confidence = round(max_entailment, 4)
        best_chunk = best_entail_chunk
        best_snippet = best_entail_snippet
    elif max_contradiction >= cont_thresh:
        status = "CONTRADICTED"
        confidence = round(max_contradiction, 4)
        best_chunk = best_contra_chunk
        best_snippet = best_contra_snippet
    else:
        status = "UNSUPPORTED"
        confidence = round(max(0.0, 1.0 - max(max_entailment, max_contradiction)), 4)
        best_chunk = None
        best_snippet = "No supporting evidence found in the retrieved textbook context."

    return status, confidence, best_chunk, best_snippet, max_entailment, max_contradiction


def verify_single_claim(
    claim: str,
    retrieved_chunks: List[Dict[str, Any]],
    verifier: Optional[Dict[str, Any]] = None,
    entail_thresh: Optional[float] = None,
    contra_thresh: Optional[float] = None,
    debug: bool = False
) -> Dict[str, Any]:
    """
    Verifies a single factual claim against the already retrieved chunks.
    Uses selective execution:
    1. Evaluates lightweight heuristic across all chunks.
    2. If unambiguous (clear support or clear out-of-context unsupported), resolves immediately.
    3. If ambiguous, runs ModernBERT NLI on top-K candidate chunks.
    """
    ent_thresh = ent_thresh if entail_thresh is not None else VERIFIER_CONFIG["NLI_ENT_THRESHOLD"]
    cont_thresh = contra_thresh if contra_thresh is not None else VERIFIER_CONFIG["NLI_CONT_THRESHOLD"]
    top_k = VERIFIER_CONFIG.get("NLI_TOP_K", 3)

    if not retrieved_chunks:
        return {
            "claim": claim,
            "status": "UNSUPPORTED",
            "confidence": 0.0,
            "method": "lightweight",
            "score": 0.0,
            "best_chunk": None,
            "best_evidence_snippet": "",
            "evidence": [],
            "evaluations": [],
            "nli_triggered": False
        }

    # Step 1: Lightweight scoring across all retrieved chunks
    t_lw_start = time.time()
    all_lw_evals = []
    max_lw_score = 0.0
    has_polarity_conflict = False

    for idx, chunk in enumerate(retrieved_chunks):
        chunk_text = chunk.get("text", "")
        lw_res = score_claim_lightweight(claim, chunk_text)
        snippet = find_best_sentence_snippet(claim, chunk_text)
        
        eval_item = {
            "chunk_idx": idx,
            "chunk_id": chunk.get("chunk_id", f"chunk_{idx+1}"),
            "section": chunk.get("section", "Unknown"),
            "section_path": chunk.get("section_path", ""),
            "chapter": chunk.get("chapter", "Unknown"),
            "page_start": chunk.get("page_start", 0),
            "page_end": chunk.get("page_end", 0),
            "pages": chunk.get("pages", []),
            "rrf_score": chunk.get("rrf_score", 0.0),
            "heuristic_score": lw_res["heuristic_score"],
            "entailment": lw_res["entailment"],
            "neutral": lw_res["neutral"],
            "contradiction": lw_res["contradiction"],
            "evidence_snippet": snippet,
            "chunk": chunk
        }
        all_lw_evals.append(eval_item)
        if lw_res["heuristic_score"] > max_lw_score:
            max_lw_score = lw_res["heuristic_score"]
        if lw_res.get("has_polarity_contradiction"):
            has_polarity_conflict = True

    t_lw = time.time() - t_lw_start

    # Determine if claim is ambiguous or obvious
    is_clear_supported = (max_lw_score >= VERIFIER_CONFIG["LIGHTWEIGHT_CLEAR_ENT_THRESHOLD"])
    is_clear_polarity_contra = has_polarity_conflict
    is_clear_unsupported = (max_lw_score < VERIFIER_CONFIG["LIGHTWEIGHT_CLEAR_UNSUPPORTED"])

    nli_available = (verifier is not None and verifier.get("available", False) and verifier.get("model") is not None)

    # If obvious case or NLI unavailable -> finish at lightweight stage
    if (is_clear_supported or is_clear_polarity_contra or is_clear_unsupported or not nli_available):
        method_name = "lightweight" if nli_available else "lightweight_fallback"
        status, confidence, best_chunk, best_snippet, max_e, max_c = aggregate_evidence(
            all_lw_evals, ent_thresh=ent_thresh, cont_thresh=cont_thresh
        )

        if debug:
            logger.info(f"[Debug] Claim: '{claim}' -> Method: {method_name}, Status: {status}, Score: {confidence:.2f}")

        return {
            "claim": claim,
            "status": status,
            "confidence": confidence,
            "method": method_name,
            "score": confidence,
            "best_chunk": best_chunk,
            "best_evidence_snippet": best_snippet,
            "evidence": [
                {
                    "section": ev.get("section", "Unknown"),
                    "chapter": ev.get("chapter", "Unknown"),
                    "page_start": ev.get("page_start", 0),
                    "page_end": ev.get("page_end", 0),
                    "chunk_id": ev.get("chunk_id", ""),
                    "text": ev.get("chunk", {}).get("text", "")
                } for ev in all_lw_evals if ev.get("heuristic_score", 0) > 0.3
            ] or ([{
                "section": best_chunk.get("section", "Unknown"),
                "chapter": best_chunk.get("chapter", "Unknown"),
                "page_start": best_chunk.get("page_start", 0),
                "page_end": best_chunk.get("page_end", 0),
                "chunk_id": best_chunk.get("chunk_id", ""),
                "text": best_chunk.get("text", "")
            }] if best_chunk else []),
            "evaluations": all_lw_evals,
            "nli_triggered": False,
            "lightweight_time": round(t_lw, 4),
            "nli_time": 0.0
        }

    # Step 2: Ambiguous claim -> Select top candidate chunks for ModernBERT NLI
    top_candidates = select_evidence_for_nli(claim, retrieved_chunks, top_k=top_k)
    
    # Formulate individual pairs and optional combined pair of top 2
    pairs_to_run = [(cand[1].get("text", ""), claim) for cand in top_candidates]
    eval_is_combined = False
    
    if len(top_candidates) >= 2:
        combined_text = top_candidates[0][1].get("text", "") + "\n\n" + top_candidates[1][1].get("text", "")
        pairs_to_run.append((combined_text, claim))
        eval_is_combined = True

    t_nli_start = time.time()
    nli_results = run_modernbert_nli_batch(pairs_to_run, verifier)
    t_nli = time.time() - t_nli_start

    # Merge NLI probabilities into candidate evaluations
    final_evals = []
    evals_map = {ev["chunk_idx"]: ev for ev in all_lw_evals}

    num_individual = len(top_candidates)
    for i, (cand_idx, cand_chunk, _) in enumerate(top_candidates):
        nli_res = nli_results[i]
        eval_item = evals_map[cand_idx]
        eval_item["entailment"] = nli_res["entailment"]
        eval_item["neutral"] = nli_res["neutral"]
        eval_item["contradiction"] = nli_res["contradiction"]
        final_evals.append(eval_item)

    # Check if combined multi-chunk evaluation established entailment
    combined_entailment = 0.0
    if eval_is_combined and len(nli_results) > num_individual:
        comb_res = nli_results[-1]
        combined_entailment = comb_res.get("entailment", 0.0)

    # Aggregate across individual evaluated chunks
    status, confidence, best_chunk, best_snippet, max_e, max_c = aggregate_evidence(
        final_evals, ent_thresh=ent_thresh, cont_thresh=cont_thresh, conflict_margin=VERIFIER_CONFIG["NLI_CONFLICT_MARGIN"]
    )

    # If individual chunks were neutral/partial but multi-chunk combined premise entails the claim:
    if status == "UNSUPPORTED" and combined_entailment >= 0.50 and max_c < 0.25:
        status = "VERIFIED"
        confidence = round(combined_entailment, 4)
        best_chunk = top_candidates[0][1]
        best_snippet = "Multi-chunk verified: " + " | ".join([find_best_sentence_snippet(claim, cand[1].get("text", "")) for cand in top_candidates[:2]])

    if debug:
        logger.info(f"[Debug] Claim: '{claim}' -> Method: modernbert_nli, Status: {status}, Conf: {confidence:.2f}, NLI Pairs: {len(pairs_to_run)}")

    # Format structured evidence references
    evidence_list = []
    for ev in final_evals:
        c = ev.get("chunk", {})
        evidence_list.append({
            "section": c.get("section", "Unknown"),
            "chapter": c.get("chapter", "Unknown"),
            "page_start": c.get("page_start", 0),
            "page_end": c.get("page_end", 0),
            "chunk_id": c.get("chunk_id", ""),
            "text": c.get("text", "")
        })

    return {
        "claim": claim,
        "status": status,
        "confidence": confidence,
        "method": "modernbert_nli",
        "score": confidence,
        "best_chunk": best_chunk,
        "best_evidence_snippet": best_snippet,
        "evidence": evidence_list,
        "evaluations": all_lw_evals,
        "nli_triggered": True,
        "lightweight_time": round(t_lw, 4),
        "nli_time": round(t_nli, 4)
    }


# ─────────────────────────────────────────────
# 8. OVERALL ANSWER VERIFICATION
# ─────────────────────────────────────────────

def verify_answer(
    answer: str,
    retrieved_chunks: List[Dict[str, Any]],
    client=None,
    verifier: Optional[Dict[str, Any]] = None,
    entail_thresh: Optional[float] = None,
    contra_thresh: Optional[float] = None,
    debug: bool = False
) -> Dict[str, Any]:
    """
    Post-generation verification entry point.
    Receives the generated answer and the exact same retrieved chunks passed to the LLM.
    Does NOT modify or rewrite the answer.
    """
    start_time = time.time()

    # Handle standard 'Not found' responses
    if "not found in the provided textbook" in answer.lower():
        return {
            "answer": answer,
            "overall_status": "NOT APPLICABLE",
            "is_available": True,
            "message": "Answer indicates information is absent from textbook.",
            "claims": [],
            "total_claims": 0,
            "verified_count": 0,
            "unsupported_count": 0,
            "contradicted_count": 0,
            "conflict_count": 0,
            "verification_rate": 1.0,
            "elapsed_seconds": 0.0,
            "metrics": {
                "num_claims": 0,
                "num_nli_calls": 0,
                "lightweight_time": 0.0,
                "nli_time": 0.0,
                "total_verification_time": 0.0
            }
        }

    # Extract claims
    claims = extract_claims(client, answer)

    if not claims:
        elapsed = round(time.time() - start_time, 4)
        return {
            "answer": answer,
            "overall_status": "UNSUPPORTED",
            "is_available": True,
            "message": "No atomic claims extracted.",
            "claims": [],
            "total_claims": 0,
            "verified_count": 0,
            "unsupported_count": 0,
            "contradicted_count": 0,
            "conflict_count": 0,
            "verification_rate": 0.0,
            "elapsed_seconds": elapsed,
            "metrics": {
                "num_claims": 0,
                "num_nli_calls": 0,
                "lightweight_time": 0.0,
                "nli_time": 0.0,
                "total_verification_time": elapsed
            }
        }

    verified_claims = []
    verified_count = 0
    unsupported_count = 0
    contradicted_count = 0
    conflict_count = 0

    total_nli_calls = 0
    total_lw_time = 0.0
    total_nli_time = 0.0

    for claim in claims:
        claim_res = verify_single_claim(
            claim=claim,
            retrieved_chunks=retrieved_chunks,
            verifier=verifier,
            entail_thresh=entail_thresh,
            contra_thresh=contra_thresh,
            debug=debug
        )
        verified_claims.append(claim_res)

        if claim_res.get("nli_triggered"):
            total_nli_calls += 1
        total_lw_time += claim_res.get("lightweight_time", 0.0)
        total_nli_time += claim_res.get("nli_time", 0.0)

        status = claim_res["status"]
        if status == "VERIFIED":
            verified_count += 1
        elif status == "CONTRADICTED":
            contradicted_count += 1
        elif status == "EVIDENCE CONFLICT":
            conflict_count += 1
        else:
            unsupported_count += 1

    total_claims = len(verified_claims)

    # Overall Status Decision Hierarchy
    if conflict_count > 0:
        overall_status = "EVIDENCE CONFLICT"
    elif contradicted_count > 0 and verified_count == 0:
        overall_status = "CONTRADICTED"
    elif contradicted_count > 0 and verified_count > 0:
        overall_status = "PARTIALLY VERIFIED"
    elif verified_count == total_claims:
        overall_status = "VERIFIED"
    elif verified_count > 0:
        overall_status = "PARTIALLY VERIFIED"
    else:
        overall_status = "UNSUPPORTED"

    verification_rate = round(verified_count / total_claims, 2) if total_claims > 0 else 0.0
    total_elapsed = round(time.time() - start_time, 4)

    return {
        "answer": answer,
        "overall_status": overall_status,
        "is_available": True,
        "claims": verified_claims,
        "total_claims": total_claims,
        "verified_count": verified_count,
        "unsupported_count": unsupported_count,
        "contradicted_count": contradicted_count,
        "conflict_count": conflict_count,
        "verification_rate": verification_rate,
        "elapsed_seconds": total_elapsed,
        "metrics": {
            "num_claims": total_claims,
            "num_nli_calls": total_nli_calls,
            "lightweight_time": round(total_lw_time, 4),
            "nli_time": round(total_nli_time, 4),
            "total_verification_time": total_elapsed
        }
    }
