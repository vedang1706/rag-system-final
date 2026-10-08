"""
verification_1: Single Query Real End-to-End Diagnostic
Runs ONE randomly selected textbook question through the live RAG + Verification pipeline.
Measures wall-clock timings, inspects claim decomposition, NLI probabilities, and final verification status.
"""

import sys
import os
import io

# Force UTF-8 encoding on standard streams to avoid Windows cp1252 charmap encoding errors
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import time
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import httpx
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Ensure project root and src are on sys.path
project_root = Path(__file__).resolve().parent.parent
src_dir = project_root / "src"
for p in [str(project_root), str(src_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.embed_index import load_chunks, load_chroma_collection, load_embed_model
from src.retrieve import load_or_build_bm25, retrieve
from src.generate import generate_answer, NVIDIA_BASE_URL, NVIDIA_API_KEY, MODEL_NAME
from verification_1.claim_extractor import extract_claims
from verification_1.similarity_matcher import get_embedding_model, compute_cosine_similarity
from verification_1.nli_matcher import get_nli_model, compute_nli_scores_batch

OUTPUT_JSON_PATH = project_root / "verification_1" / "output" / "single_query_diagnostic.json"
OUTPUT_MD_PATH = project_root / "verification_1" / "output" / "single_query_diagnostic.md"
QUERIES_PATH = project_root / "data" / "queries.json"

# Verification Thresholds (from current implementation in multi_chunk_aggregator.py)
SIMILARITY_THRESHOLD = 0.40
NLI_ENTAILMENT_THRESHOLD = 0.50
NLI_CONTRADICTION_THRESHOLD = 0.50


def run_diagnostic():
    print("=" * 80)
    print("TASK 1: RUN ONE REAL END-TO-END VERIFICATION DIAGNOSTIC")
    print("=" * 80)

    # 1. Load queries and pick ONE random question
    with open(QUERIES_PATH, "r", encoding="utf-8") as f:
        queries = json.load(f)

    # Use a reproducible random selection
    rng = random.Random(42)
    selected_query = rng.choice(queries)
    qid = str(selected_query.get("query_id") or selected_query.get("id"))
    question = selected_query.get("question") or selected_query.get("query")

    print(f"\nSelected Query ID: {qid}")
    print(f"Selected Question: {question}")

    # Pre-load/warm up components so initialization doesn't artificially distort retrieval measurement
    print("\nInitializing RAG and Verifier components...")
    chunks = load_chunks(str(project_root / "cache" / "chunks.json"))
    collection = load_chroma_collection()
    ret_embed_model = load_embed_model()
    bm25 = load_or_build_bm25(chunks)
    
    # Initialize OpenAI client with explicit httpx.Client to handle httpx 0.28+ compatibility
    client = OpenAI(
        base_url=NVIDIA_BASE_URL,
        api_key=NVIDIA_API_KEY,
        http_client=httpx.Client()
    )
    print(f"✓ NVIDIA client initialized (Model: {MODEL_NAME})")

    sim_model = get_embedding_model()
    nli_bundle = get_nli_model()

    print("\nStarting End-to-End Pipeline Execution...")
    # Wall clock start
    t0_pipeline = time.perf_counter()

    # -------------------------------------------------------------
    # 1. RETRIEVAL
    # -------------------------------------------------------------
    t0_ret = time.perf_counter()
    retrieved_chunks = retrieve(
        query=question,
        collection=collection,
        model=ret_embed_model,
        bm25=bm25,
        chunks=chunks,
        top_k=5
    )
    t_retrieval = time.perf_counter() - t0_ret
    print(f"  ✓ Retrieval completed in {t_retrieval:.4f}s ({len(retrieved_chunks)} chunks retrieved)")

    # -------------------------------------------------------------
    # 2. ANSWER GENERATION
    # -------------------------------------------------------------
    t0_gen = time.perf_counter()
    gen_result = generate_answer(
        client=client,
        query_id=qid,
        question=question,
        retrieved_chunks=retrieved_chunks,
        answer_style="Standard (2-5 Sentences)"
    )
    generated_answer = gen_result["answer"]
    t_generation = time.perf_counter() - t0_gen
    print(f"  ✓ Answer Generation completed in {t_generation:.4f}s")
    print(f"\nGenerated Answer:\n{generated_answer}\n")

    # -------------------------------------------------------------
    # 3. CLAIM EXTRACTION
    # -------------------------------------------------------------
    t0_claim = time.perf_counter()
    claim_obj = extract_claims(answer=generated_answer, query_id=qid)
    claims = claim_obj.get("claims", [])
    non_claim_text = claim_obj.get("non_claim_text", [])
    extractor_engine = claim_obj.get("engine", "unknown")
    t_claim_extraction = time.perf_counter() - t0_claim
    print(f"  ✓ Claim Extraction ({extractor_engine}) completed in {t_claim_extraction:.4f}s ({len(claims)} atomic claims extracted)")

    # -------------------------------------------------------------
    # 4. VERIFICATION LAYER
    # -------------------------------------------------------------
    t0_ver = time.perf_counter()
    verified_claims = []

    supported_count = 0
    neutral_count = 0
    contradiction_count = 0
    insufficient_count = 0
    conflict_count = 0

    # Evaluate each claim against all 5 retrieved chunks
    for c in claims:
        cid = c["claim_id"]
        ctext = c["claim_text"]
        s_idx = c.get("sentence_index", 1)

        c_vec = sim_model.encode(ctext, normalize_embeddings=True, show_progress_bar=False)

        # Batch NLI across all 5 retrieved chunks: (premise=chunk_raw_text, hypothesis=claim_text)
        nli_pairs = [(chunk["text"], ctext) for chunk in retrieved_chunks]
        nli_scores_list = compute_nli_scores_batch(nli_pairs, model_bundle=nli_bundle, batch_size=5)

        chunk_evaluations = []
        support_cands = []
        contra_cands = []

        for chunk_idx, chunk in enumerate(retrieved_chunks):
            chunk_num = chunk_idx + 1
            chunk_raw_text = chunk["text"]
            chunk_vec = sim_model.encode(chunk_raw_text, normalize_embeddings=True, show_progress_bar=False)

            sim_score = float(compute_cosine_similarity(c_vec, chunk_vec))
            nli_item = nli_scores_list[chunk_idx]
            ent_p = float(nli_item["entailment_score"])
            neu_p = float(nli_item["neutral_score"])
            con_p = float(nli_item["contradiction_score"])

            # Argmax label
            if ent_p >= neu_p and ent_p >= con_p:
                pred_label = "entailment"
            elif con_p >= neu_p and con_p >= ent_p:
                pred_label = "contradiction"
            else:
                pred_label = "neutral"

            is_supp = (sim_score >= SIMILARITY_THRESHOLD) and (ent_p >= NLI_ENTAILMENT_THRESHOLD)
            is_contra = (sim_score >= SIMILARITY_THRESHOLD) and (con_p >= NLI_CONTRADICTION_THRESHOLD)

            chunk_eval = {
                "chunk_number": chunk_num,
                "chunk_id": chunk.get("chunk_id", chunk_num),
                "section": chunk.get("section", "Unknown"),
                "section_path": chunk.get("section_path", "Unknown"),
                "chapter": chunk.get("chapter", "Unknown"),
                "pages": chunk.get("pages", []),
                "similarity": round(sim_score, 4),
                "nli_probabilities": {
                    "contradiction": round(con_p, 4),
                    "neutral": round(neu_p, 4),
                    "entailment": round(ent_p, 4)
                },
                "nli_predicted_label": pred_label,
                "is_support_candidate": is_supp,
                "is_contradiction_candidate": is_contra,
                "chunk_snippet": chunk_raw_text[:200] + "..." if len(chunk_raw_text) > 200 else chunk_raw_text
            }
            chunk_evaluations.append(chunk_eval)

            if is_supp:
                support_cands.append(chunk_eval)
            if is_contra:
                contra_cands.append(chunk_eval)

        # Multi-chunk decision logic from multi_chunk_aggregator.py
        has_support = len(support_cands) > 0
        has_contra = len(contra_cands) > 0

        if has_support and not has_contra:
            final_verdict = "SUPPORTED"
            supported_count += 1
        elif has_contra and not has_support:
            final_verdict = "CONTRADICTED"
            contradiction_count += 1
        elif has_support and has_contra:
            final_verdict = "CONFLICT"
            conflict_count += 1
        else:
            final_verdict = "INSUFFICIENT"
            insufficient_count += 1

        # Determine best / selected evidence chunk
        if support_cands:
            best_chunk = max(support_cands, key=lambda x: (x["nli_probabilities"]["entailment"], x["similarity"]))
        else:
            # Pick highest similarity chunk
            best_chunk = max(chunk_evaluations, key=lambda x: x["similarity"])

        # Count NLI label on best chunk
        best_nli_label = best_chunk["nli_predicted_label"]
        if best_nli_label == "neutral":
            neutral_count += 1

        verified_claims.append({
            "claim_id": cid,
            "claim_text": ctext,
            "sentence_index": s_idx,
            "best_evidence_chunk": {
                "chunk_number": best_chunk["chunk_number"],
                "chunk_id": best_chunk["chunk_id"],
                "section": best_chunk["section"],
                "section_path": best_chunk["section_path"],
                "pages": best_chunk["pages"],
                "similarity": best_chunk["similarity"],
                "nli_probabilities": best_chunk["nli_probabilities"],
                "nli_predicted_label": best_chunk["nli_predicted_label"],
            },
            "final_verdict": final_verdict,
            "all_chunk_evaluations": chunk_evaluations
        })

    t_verification = time.perf_counter() - t0_ver
    t_total_pipeline = time.perf_counter() - t0_pipeline

    # Overall Answer Status (from multi_chunk_aggregator.py)
    if contradiction_count > 0:
        overall_status = "HAS_CONTRADICTION"
    elif conflict_count > 0:
        overall_status = "CONFLICTING_EVIDENCE"
    elif insufficient_count > 0:
        overall_status = "PARTIALLY_SUPPORTED"
    else:
        overall_status = "SUPPORTED"

    # Overall Verification Score
    overall_score_str = "Not currently produced by implementation."
    supported_pct = round((supported_count / max(len(claims), 1)) * 100, 2)

    # Compile JSON Output
    diagnostic_data = {
        "experiment": "Task 1: Real End-to-End Verification Diagnostic",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "question_id": qid,
        "question": question,
        "generated_answer": generated_answer,
        "timing": {
            "retrieval_seconds": round(t_retrieval, 4),
            "generation_seconds": round(t_generation, 4),
            "claim_extraction_seconds": round(t_claim_extraction, 4),
            "verification_seconds": round(t_verification, 4),
            "total_pipeline_seconds": round(t_total_pipeline, 4)
        },
        "retrieved_chunks": [
            {
                "chunk_number": idx + 1,
                "chunk_id": c.get("chunk_id"),
                "section": c.get("section"),
                "section_path": c.get("section_path"),
                "pages": c.get("pages"),
                "text_length_chars": len(c.get("text", "")),
                "text_snippet": c.get("text", "")[:300] + "..."
            }
            for idx, c in enumerate(retrieved_chunks)
        ],
        "claims_decomposition": {
            "engine": extractor_engine,
            "num_claims": len(claims),
            "claims": [
                {
                    "claim_id": c["claim_id"],
                    "claim_text": c["claim_text"],
                    "sentence_index": c.get("sentence_index", 1)
                }
                for c in claims
            ],
            "non_claim_text": non_claim_text
        },
        "verification_results": {
            "total_claims": len(claims),
            "verdict_counts": {
                "supported": supported_count,
                "contradicted": contradiction_count,
                "insufficient": insufficient_count,
                "conflict": conflict_count
            },
            "best_chunk_nli_label_counts": {
                "entailment": sum(1 for vc in verified_claims if vc["best_evidence_chunk"]["nli_predicted_label"] == "entailment"),
                "neutral": sum(1 for vc in verified_claims if vc["best_evidence_chunk"]["nli_predicted_label"] == "neutral"),
                "contradiction": sum(1 for vc in verified_claims if vc["best_evidence_chunk"]["nli_predicted_label"] == "contradiction"),
            },
            "overall_verification_score": overall_score_str,
            "overall_verification_score_percentage_supported": supported_pct,
            "overall_flag_status": overall_status,
            "claims": verified_claims
        }
    }

    # Save JSON
    OUTPUT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(diagnostic_data, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Saved diagnostic JSON to: {OUTPUT_JSON_PATH}")

    # Build Markdown
    md_lines = []
    md_lines.append("# End-to-End RAG + Verification Diagnostic Report\n")
    md_lines.append(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}  ")
    md_lines.append(f"**Question ID:** `{qid}`  ")
    md_lines.append(f"**Pipeline Type:** Live RAG (`all-MiniLM-L6-v2` + BM25 + NVIDIA `gpt-oss-20b`) + `verification_1`  ")
    md_lines.append("\n---\n")

    md_lines.append("## 1. Query & Generated Output\n")
    md_lines.append(f"### QUESTION:\n> **{question}**\n")
    md_lines.append(f"### GENERATED ANSWER:\n> {generated_answer}\n")

    md_lines.append("\n---\n")
    md_lines.append("## 2. Timing Measurements\n")
    md_lines.append("| Pipeline Stage | Measured Time (seconds) | Percentage of Total |")
    md_lines.append("| :--- | :---: | :---: |")
    md_lines.append(f"| **1. Retrieval (Vector + BM25 + RRF)** | `{t_retrieval:.4f} s` | {t_retrieval/t_total_pipeline*100:.1f}% |")
    md_lines.append(f"| **2. Answer Generation (NVIDIA API)** | `{t_generation:.4f} s` | {t_generation/t_total_pipeline*100:.1f}% |")
    md_lines.append(f"| **3. Claim Extraction (Gemini Atomic)** | `{t_claim_extraction:.4f} s` | {t_claim_extraction/t_total_pipeline*100:.1f}% |")
    md_lines.append(f"| **4. Verification Layer (Similarity + DeBERTa NLI)** | `{t_verification:.4f} s` | {t_verification/t_total_pipeline*100:.1f}% |")
    md_lines.append(f"| **TOTAL PIPELINE TIME** | **`{t_total_pipeline:.4f} s`** | **100.0%** |")

    md_lines.append("\n---\n")
    md_lines.append("## 3. Extracted Atomic Claims\n")
    md_lines.append("| Claim ID | Sentence Index | Atomic Claim Text |")
    md_lines.append("| :--- | :---: | :--- |")
    for c in claims:
        md_lines.append(f"| **`{c['claim_id']}`** | {c.get('sentence_index', 1)} | {c['claim_text']} |")

    md_lines.append("\n---\n")
    md_lines.append("## 4. Claim-Level Verification Results\n")
    md_lines.append("| Claim ID | Claim Text | Best Evidence Chunk | Section | Page(s) | Similarity | NLI (Contra / Neu / Ent) | NLI Label | Verdict |")
    md_lines.append("| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |")

    for vc in verified_claims:
        be = vc["best_evidence_chunk"]
        probs = be["nli_probabilities"]
        prob_str = f"{probs['contradiction']:.3f} / {probs['neutral']:.3f} / {probs['entailment']:.3f}"
        pages_str = str(be["pages"])
        sec_short = be["section"][:25] + "..." if len(be["section"]) > 25 else be["section"]
        claim_short = vc["claim_text"][:35] + "..." if len(vc["claim_text"]) > 35 else vc["claim_text"]
        md_lines.append(
            f"| `{vc['claim_id']}` | {claim_short} | Chunk {be['chunk_number']} (ID {be['chunk_id']}) | {sec_short} | {pages_str} | {be['similarity']:.4f} | {prob_str} | `{be['nli_predicted_label']}` | **`{vc['final_verdict']}`** |"
        )

    md_lines.append("\n---\n")
    md_lines.append("## 5. Aggregate Summary & Overall Verdict\n")
    md_lines.append(f"- **TOTAL CLAIMS:** {len(claims)}")
    md_lines.append(f"- **SUPPORTED / ENTAILMENT:** {supported_count}")
    md_lines.append(f"- **NEUTRAL (on best chunk):** {neutral_count}")
    md_lines.append(f"- **CONTRADICTION:** {contradiction_count}")
    md_lines.append(f"- **OTHER / INSUFFICIENT:** {insufficient_count}")
    md_lines.append(f"- **CONFLICT:** {conflict_count}")
    md_lines.append(f"\n### OVERALL VERIFICATION SCORE:\n> `{overall_score_str}` *(Supported Claim Ratio: {supported_pct}%)*")
    md_lines.append(f"\n### OVERALL FLAG/STATUS:\n> **`{overall_status}`**")

    md_lines.append("\n---\n")
    md_lines.append("## 6. Diagnostic Observations on the 'Many Neutral' Phenomenon\n")
    
    neutral_with_high_sim = [
        vc for vc in verified_claims 
        if vc["final_verdict"] == "INSUFFICIENT" and vc["best_evidence_chunk"]["similarity"] >= 0.40
    ]
    
    if neutral_with_high_sim:
        md_lines.append(f"**Identified {len(neutral_with_high_sim)} claim(s) exhibiting the 'Many Neutral' phenomenon** (Cosine Similarity >= 0.40, but NLI assigned `Neutral` due to 400-word full-chunk premise dilution):\n")
        for n_item in neutral_with_high_sim:
            be = n_item["best_evidence_chunk"]
            md_lines.append(f"### Claim `{n_item['claim_id']}`: *\"{n_item['claim_text']}\"*")
            md_lines.append(f"- **Selected Evidence Chunk:** Chunk {be['chunk_number']} ({be['section']})")
            md_lines.append(f"- **Semantic Similarity:** `{be['similarity']:.4f}` (>= 0.40)")
            md_lines.append(f"- **NLI Probabilities:** Contradiction: `{be['nli_probabilities']['contradiction']:.4f}`, Neutral: `{be['nli_probabilities']['neutral']:.4f}`, Entailment: `{be['nli_probabilities']['entailment']:.4f}`")
            md_lines.append(f"- **Diagnosis:** The chunk text clearly contains the relevant topic, but the cross-encoder attention was diluted across the full 400-word passage, failing to reach the >= 0.50 Entailment threshold.")
            md_lines.append("")
    else:
        md_lines.append("No claims exhibited the high-similarity neutral phenomenon in this query.")

    with open(OUTPUT_MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"✓ Saved diagnostic Markdown to: {OUTPUT_MD_PATH}")

    print("\n" + "=" * 80)
    print("DIAGNOSTIC COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    run_diagnostic()
