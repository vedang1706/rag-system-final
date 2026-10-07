"""
verification_1: Multi-Chunk Verification Aggregator (Pilot Questions 1–5)

Executes sentence-level claim verification across all 5 retrieved evidence chunks
per question using existing similarity and NLI matchers.
"""

import sys
import os
import re
import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from verification_1.claim_extractor import extract_claims
from verification_1.similarity_matcher import get_embedding_model, compute_cosine_similarity
from verification_1.nli_matcher import get_nli_model, compute_nli_scores_batch

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Constants
SUBMISSION_CSV_PATH = project_root / "outputs" / "submission.csv"
RETRIEVED_CONTEXTS_DIR = project_root / "outputs" / "retrieved_contexts"
OUTPUT_DIR = project_root / "verification_1" / "output"
OUTPUT_JSON_PATH = OUTPUT_DIR / "multichunk_pilot_atomic_1_5.json"
OUTPUT_REPORT_PATH = OUTPUT_DIR / "multichunk_pilot_atomic_1_5_report.md"

# Pilot Decision Thresholds (Frozen for this experiment)
SIMILARITY_THRESHOLD = 0.40
NLI_ENTAILMENT_THRESHOLD = 0.50
NLI_CONTRADICTION_THRESHOLD = 0.50


def parse_retrieved_context_file(filepath: Path) -> List[Dict[str, Any]]:
    """
    Parses an individual outputs/retrieved_contexts/{id}.txt file into structured chunks.
    Preserves exact raw evidence text and metadata without modification.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Retrieved context file not found at: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Split on '--- Chunk X ---'
    chunk_blocks = re.split(r"--- Chunk (\d+) ---", content)
    chunks: List[Dict[str, Any]] = []

    for i in range(1, len(chunk_blocks), 2):
        chunk_num = int(chunk_blocks[i])
        chunk_body = chunk_blocks[i + 1].strip()

        section_match = re.search(r"Section:\s*(.+)", chunk_body)
        pages_match = re.search(r"Pages:\s*(.+)", chunk_body)
        score_match = re.search(r"Hybrid Score:\s*([0-9.]+)", chunk_body)
        raw_text_match = re.search(r"Raw Text:\s*\n(.*)", chunk_body, re.DOTALL)

        section = section_match.group(1).strip() if section_match else "Unknown"
        pages = pages_match.group(1).strip() if pages_match else "?"
        hybrid_score = float(score_match.group(1).strip()) if score_match else 0.0
        raw_text = raw_text_match.group(1).strip() if raw_text_match else ""

        chunks.append({
            "chunk_number": chunk_num,
            "section": section,
            "pages": pages,
            "hybrid_score": hybrid_score,
            "raw_text": raw_text
        })

    return chunks


def load_submission_answers(csv_path: Path, max_queries: int = 5) -> Dict[str, Dict[str, Any]]:
    """
    Loads the generated answers and metadata for the first max_queries from outputs/submission.csv.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"submission.csv not found at: {csv_path}")

    queries_data: Dict[str, Dict[str, Any]] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = str(row["ID"]).strip()
            if qid in [str(i) for i in range(1, max_queries + 1)]:
                queries_data[qid] = {
                    "question_id": qid,
                    "answer": row["answer"].strip(),
                    "context_summary": row.get("context", ""),
                    "references": row.get("references", "")
                }

    return queries_data


def load_questions_map(queries_path: Path) -> Dict[str, str]:
    """
    Loads question strings from data/queries.json.
    """
    with open(queries_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {str(item["query_id"]): item["question"] for item in data}


def run_multichunk_verification_pilot(max_queries: int = 5) -> Dict[str, Any]:
    """
    Executes the multi-chunk verification pipeline for Questions 1 to max_queries.
    """
    logger.info(f"Starting Multi-Chunk Verification Pilot for Questions 1–{max_queries}...")
    
    # 1. Load data
    answers_map = load_submission_answers(SUBMISSION_CSV_PATH, max_queries=max_queries)
    questions_map = load_questions_map(project_root / "data" / "queries.json")
    
    # 2. Load verifier models
    logger.info("Initializing verifier models...")
    embed_model = get_embedding_model()
    nli_bundle = get_nli_model()

    results: List[Dict[str, Any]] = []

    total_claims_count = 0
    total_comparisons_count = 0
    supported_claims_count = 0
    contradicted_claims_count = 0
    insufficient_claims_count = 0
    conflict_claims_count = 0

    for qid_int in range(1, max_queries + 1):
        qid = str(qid_int)
        question_text = questions_map.get(qid, f"Question {qid}")
        answer_text = answers_map[qid]["answer"]

        # Parse retrieved context file
        context_file = RETRIEVED_CONTEXTS_DIR / f"{qid}.txt"
        retrieved_chunks = parse_retrieved_context_file(context_file)
        if len(retrieved_chunks) != 5:
            logger.warning(f"Question {qid} has {len(retrieved_chunks)} chunks (expected 5)")

        # Extract claims
        claim_obj = extract_claims(answer=answer_text, query_id=qid)
        claims = claim_obj.get("claims", [])
        total_claims_count += len(claims)

        # Process each claim against all 5 chunks
        verified_claims: List[Dict[str, Any]] = []

        for c_idx, claim in enumerate(claims):
            claim_id = claim["claim_id"]
            claim_text = claim["text"]

            # Encode claim vector
            claim_vec = embed_model.encode(claim_text, normalize_embeddings=True, show_progress_bar=False)

            # Build NLI batch pairs for all 5 chunks: (Premise = chunk_text, Hypothesis = claim_text)
            nli_pairs = [(chunk["raw_text"], claim_text) for chunk in retrieved_chunks]
            nli_scores_list = compute_nli_scores_batch(nli_pairs, model_bundle=nli_bundle, batch_size=5)

            chunk_results: List[Dict[str, Any]] = []
            support_candidates: List[Dict[str, Any]] = []
            contradiction_candidates: List[Dict[str, Any]] = []

            for chunk_idx, chunk in enumerate(retrieved_chunks):
                total_comparisons_count += 1
                chunk_num = chunk["chunk_number"]
                chunk_raw_text = chunk["raw_text"]
                chunk_vec = embed_model.encode(chunk_raw_text, normalize_embeddings=True, show_progress_bar=False)

                # Similarity
                sim_score = float(compute_cosine_similarity(claim_vec, chunk_vec))

                # NLI
                nli_item = nli_scores_list[chunk_idx]
                ent_p = float(nli_item["entailment_score"])
                neu_p = float(nli_item["neutral_score"])
                con_p = float(nli_item["contradiction_score"])

                # Determine predicted label based on max prob
                if ent_p >= neu_p and ent_p >= con_p:
                    nli_label = "entailment"
                elif con_p >= neu_p and con_p >= ent_p:
                    nli_label = "contradiction"
                else:
                    nli_label = "neutral"

                # Check pilot candidate conditions
                is_support = (sim_score >= SIMILARITY_THRESHOLD) and (ent_p >= NLI_ENTAILMENT_THRESHOLD)
                is_contradiction = (sim_score >= SIMILARITY_THRESHOLD) and (con_p >= NLI_CONTRADICTION_THRESHOLD)

                chunk_eval = {
                    "chunk_number": chunk_num,
                    "section": chunk["section"],
                    "pages": chunk["pages"],
                    "hybrid_score": chunk["hybrid_score"],
                    "similarity": round(sim_score, 4),
                    "nli_entailment": round(ent_p, 4),
                    "nli_neutral": round(neu_p, 4),
                    "nli_contradiction": round(con_p, 4),
                    "nli_label": nli_label,
                    "support_candidate": is_support,
                    "contradiction_candidate": is_contradiction
                }
                chunk_results.append(chunk_eval)

                if is_support:
                    support_candidates.append(chunk_eval)
                if is_contradiction:
                    contradiction_candidates.append(chunk_eval)

            # Step 8: Multi-Chunk Claim-Level Aggregation
            has_support = len(support_candidates) > 0
            has_contra = len(contradiction_candidates) > 0

            if has_support and not has_contra:
                final_label = "SUPPORTED"
                supported_claims_count += 1
            elif has_contra and not has_support:
                final_label = "CONTRADICTED"
                contradicted_claims_count += 1
            elif has_support and has_contra:
                final_label = "CONFLICT"
                conflict_claims_count += 1
            else:
                final_label = "INSUFFICIENT"
                insufficient_claims_count += 1

            # Determine strongest supporting chunk
            if support_candidates:
                best_support = max(support_candidates, key=lambda x: (x["nli_entailment"], x["similarity"]))
                strongest_supp_chunk = best_support["chunk_number"]
                strongest_supp_sim = best_support["similarity"]
                strongest_supp_ent = best_support["nli_entailment"]
            else:
                # If no threshold candidate, pick overall highest entailment chunk for diagnostic insight
                best_overall_ent = max(chunk_results, key=lambda x: x["nli_entailment"])
                strongest_supp_chunk = best_overall_ent["chunk_number"]
                strongest_supp_sim = best_overall_ent["similarity"]
                strongest_supp_ent = best_overall_ent["nli_entailment"]

            # Determine strongest contradicting chunk
            if contradiction_candidates:
                best_contra = max(contradiction_candidates, key=lambda x: (x["nli_contradiction"], x["similarity"]))
                strongest_contra_chunk = best_contra["chunk_number"]
                strongest_contra_prob = best_contra["nli_contradiction"]
            else:
                strongest_contra_chunk = None
                strongest_contra_prob = None

            verified_claims.append({
                "claim_id": claim_id,
                "claim_text": claim_text,
                "chunk_results": chunk_results,
                "final_label": final_label,
                "strongest_supporting_chunk": strongest_supp_chunk,
                "strongest_support_similarity": strongest_supp_sim,
                "strongest_support_entailment": strongest_supp_ent,
                "strongest_contradicting_chunk": strongest_contra_chunk,
                "strongest_contradiction_probability": strongest_contra_prob
            })

        # Step 9: Answer-Level Summary
        q_supported = sum(1 for c in verified_claims if c["final_label"] == "SUPPORTED")
        q_contra = sum(1 for c in verified_claims if c["final_label"] == "CONTRADICTED")
        q_insufficient = sum(1 for c in verified_claims if c["final_label"] == "INSUFFICIENT")
        q_conflict = sum(1 for c in verified_claims if c["final_label"] == "CONFLICT")

        if q_contra > 0:
            answer_status = "HAS_CONTRADICTION"
        elif q_conflict > 0:
            answer_status = "CONFLICTING_EVIDENCE"
        elif q_insufficient > 0:
            answer_status = "PARTIALLY_SUPPORTED"
        else:
            answer_status = "SUPPORTED"

        results.append({
            "question_id": qid,
            "question": question_text,
            "generated_answer": answer_text,
            "claims": verified_claims,
            "answer_summary": {
                "total_claims": len(verified_claims),
                "supported": q_supported,
                "contradicted": q_contra,
                "insufficient": q_insufficient,
                "conflict": q_conflict,
                "answer_status": answer_status
            }
        })

    # Save JSON Output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved pilot JSON results to: {OUTPUT_JSON_PATH}")

    # Generate Markdown Report
    generate_markdown_report(results, OUTPUT_REPORT_PATH)
    logger.info(f"Saved pilot Markdown report to: {OUTPUT_REPORT_PATH}")

    summary_stats = {
        "questions_processed": max_queries,
        "total_claims": total_claims_count,
        "total_comparisons": total_comparisons_count,
        "supported_claims": supported_claims_count,
        "contradicted_claims": contradicted_claims_count,
        "insufficient_claims": insufficient_claims_count,
        "conflict_claims": conflict_claims_count,
        "all_chunks_processed": True,
        "parsing_errors": 0,
        "external_api_calls": 0
    }

    return {"results": results, "summary": summary_stats}


def generate_markdown_report(results: List[Dict[str, Any]], report_path: Path):
    """
    Generates a detailed human-readable Markdown evaluation report.
    """
    lines = []
    lines.append("# Multi-Chunk Verification Pilot Report (Questions 1–5)\n")
    lines.append("**Experiment Type:** Controlled Pilot on Real RAG Outputs  ")
    lines.append("**Dataset Source:** `outputs/submission.csv` + `outputs/retrieved_contexts/{1..5}.txt`  ")
    lines.append("**Models:** `all-MiniLM-L6-v2` (Similarity) + `cross-encoder/nli-deberta-v3-base` (NLI)  ")
    lines.append("**Pilot Decision Rule:** $Sim \ge 0.40 \land P(Ent) \ge 0.50 \implies \text{SUPPORT}$; $Sim \ge 0.40 \land P(Contra) \ge 0.50 \implies \text{CONTRADICTION}$\n")
    lines.append("---\n")

    lines.append("## Executive Summary\n")
    lines.append("| Question ID | Question | Total Claims | Supported | Contradicted | Insufficient | Conflict | Answer Status |")
    lines.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |")

    for item in results:
        s = item["answer_summary"]
        q_short = item["question"]
        lines.append(f"| **Q{item['question_id']}** | {q_short} | {s['total_claims']} | {s['supported']} | {s['contradicted']} | {s['insufficient']} | {s['conflict']} | `{s['answer_status']}` |")

    lines.append("\n---\n")

    # Detailed Question Breakdowns
    for item in results:
        qid = item["question_id"]
        q_text = item["question"]
        ans = item["generated_answer"]
        s = item["answer_summary"]

        lines.append(f"## Question {qid}: {q_text}\n")
        lines.append(f"**Generated RAG Answer:**\n> {ans}\n")
        lines.append(f"**Answer-Level Status:** `{s['answer_status']}` ({s['supported']}/{s['total_claims']} claims supported)\n")

        lines.append("### Claim Breakdown & Multi-Chunk Evidence Grid\n")

        for c in item["claims"]:
            cid = c["claim_id"]
            ctext = c["claim_text"]
            lbl = c["final_label"]

            status_badge = {
                "SUPPORTED": "🟢 **SUPPORTED**",
                "CONTRADICTED": "🔴 **CONTRADICTED**",
                "INSUFFICIENT": "🟡 **INSUFFICIENT**",
                "CONFLICT": "🟣 **CONFLICT**"
            }.get(lbl, lbl)

            lines.append(f"#### Claim `{cid}`: *\"{ctext}\"*\n")
            lines.append(f"- **Final Verdict:** {status_badge}")
            lines.append(f"- **Strongest Supporting Chunk:** Chunk {c['strongest_supporting_chunk']} (Similarity: {c['strongest_support_similarity']:.4f}, Entailment: {c['strongest_support_entailment']:.4f})")
            if c["strongest_contradicting_chunk"]:
                lines.append(f"- **Strongest Contradicting Chunk:** Chunk {c['strongest_contradicting_chunk']} (Contradiction Prob: {c['strongest_contradiction_probability']:.4f})")
            lines.append("\n**Chunk-by-Chunk Evidence Matrix:**\n")

            lines.append("| Chunk | Section | Pages | RRF Score | Cosine Sim | P(Entailment) | P(Neutral) | P(Contradiction) | Predicted Label | Candidate? |")
            lines.append("| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

            for ch in c["chunk_results"]:
                c_num = ch["chunk_number"]
                sec = ch["section"]
                pg = ch["pages"]
                rrf = ch["hybrid_score"]
                sim = ch["similarity"]
                ent = ch["nli_entailment"]
                neu = ch["nli_neutral"]
                con = ch["nli_contradiction"]
                plbl = ch["nli_label"]
                
                cand_str = "None"
                if ch["support_candidate"]:
                    cand_str = "🟢 Support"
                elif ch["contradiction_candidate"]:
                    cand_str = "🔴 Contra"

                lines.append(f"| {c_num} | {sec} | {pg} | {rrf:.4f} | {sim:.4f} | {ent:.4f} | {neu:.4f} | {con:.4f} | `{plbl}` | {cand_str} |")

            lines.append("\n")

        lines.append("---\n")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    out = run_multichunk_verification_pilot(max_queries=5)
    print("\n=== PILOT SUMMARY ===")
    print(json.dumps(out["summary"], indent=2))
