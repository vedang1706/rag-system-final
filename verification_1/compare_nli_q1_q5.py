"""
verification_1: Head-to-Head NLI Comparison on Questions 1–5
Compares cross-encoder/nli-deberta-v3-base (512 max tokens) vs
dleemiller/ModernCE-base-nli (2048 max tokens) on identical (claim, evidence) inputs.
"""

import sys
import time
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple
import torch

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from verification_1.nli_matcher import get_nli_model
from verification_1.modern_nli_matcher import get_modern_nli_model

PILOT_ATOMIC_JSON = project_root / "verification_1" / "output" / "multichunk_pilot_atomic_1_5.json"
RETRIEVED_CONTEXTS_DIR = project_root / "outputs" / "retrieved_contexts"
OUTPUT_JSON_PATH = project_root / "verification_1" / "output" / "q1_q5_nli_comparison.json"
OUTPUT_REPORT_PATH = project_root / "verification_1" / "output" / "q1_q5_nli_comparison_report.txt"


def parse_retrieved_context_file(filepath: Path) -> List[Dict[str, Any]]:
    """Parses outputs/retrieved_contexts/{qid}.txt into structured chunks."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    chunk_blocks = re.split(r"--- Chunk (\d+) ---", content)
    chunks = []
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
            "raw_text": raw_text,
        })
    return chunks


def main():
    print("=" * 80)
    print("STARTING Q1-Q5 HEAD-TO-HEAD NLI COMPARISON: DeBERTa vs ModernCE")
    print("=" * 80)

    # 1. Load existing atomic pilot JSON
    print(f"Loading existing atomic pilot claims from: {PILOT_ATOMIC_JSON}")
    with open(PILOT_ATOMIC_JSON, "r", encoding="utf-8") as f:
        pilot_data = json.load(f)

    # 2. Load Models
    print("\n[1/4] Loading DeBERTa-v3-base NLI model (max_length=512)...")
    deb_tokenizer, deb_model, deb_label_map = get_nli_model()
    deb_ent_idx = deb_label_map["entailment"]
    deb_neu_idx = deb_label_map["neutral"]
    deb_con_idx = deb_label_map["contradiction"]
    deb_id2label = {v: k for k, v in deb_label_map.items()}

    print("\n[2/4] Loading ModernCE-base-nli model (max_length=2048)...")
    mod_tokenizer, mod_model, mod_label_map = get_modern_nli_model()
    mod_ent_idx = mod_label_map["entailment"]
    mod_neu_idx = mod_label_map["neutral"]
    mod_con_idx = mod_label_map["contradiction"]
    mod_id2label = {v: k for k, v in mod_label_map.items()}

    # 3. Collect all comparisons
    comparisons = []
    total_claims = 0
    question_summaries = {}

    for q_entry in pilot_data:
        qid = str(q_entry["question_id"])
        q_text = q_entry["question"]
        gen_ans = q_entry["generated_answer"]
        claims = q_entry.get("claims", [])
        total_claims += len(claims)

        ctx_file = RETRIEVED_CONTEXTS_DIR / f"{qid}.txt"
        chunks = parse_retrieved_context_file(ctx_file)

        question_summaries[qid] = {
            "question_id": qid,
            "question": q_text,
            "answer": gen_ans,
            "num_claims": len(claims),
            "num_chunks": len(chunks),
            "num_comparisons": len(claims) * len(chunks),
            "deberta_counts": {"entailment": 0, "neutral": 0, "contradiction": 0},
            "modernce_counts": {"entailment": 0, "neutral": 0, "contradiction": 0},
        }

        for claim in claims:
            cid = claim["claim_id"]
            c_text = claim["claim_text"]

            for chunk in chunks:
                chunk_num = chunk["chunk_number"]
                chunk_raw = chunk["raw_text"]

                # Token lengths
                deb_enc = deb_tokenizer(chunk_raw, c_text, truncation=False)
                deb_tok_len = len(deb_enc["input_ids"])

                mod_enc = mod_tokenizer(chunk_raw, c_text, truncation=False)
                mod_tok_len = len(mod_enc["input_ids"])

                comparisons.append({
                    "question_id": qid,
                    "claim_id": cid,
                    "claim_text": c_text,
                    "chunk_number": chunk_num,
                    "chunk_section": chunk["section"],
                    "chunk_pages": chunk["pages"],
                    "chunk_raw_text": chunk_raw,
                    "deberta_token_length": deb_tok_len,
                    "deberta_truncated_at_512": deb_tok_len > 512,
                    "modernce_token_length": mod_tok_len,
                    "modernce_truncated_at_2048": mod_tok_len > 2048,
                })

    total_comparisons = len(comparisons)
    print(f"\n[3/4] Prepared {total_comparisons} identical (claim, evidence) comparisons across {total_claims} claims.")

    # 4. Run inference for DeBERTa
    print("\nRunning inference for DeBERTa-v3-base...")
    start_deb = time.perf_counter()
    deb_batch_pairs = [(c["chunk_raw_text"], c["claim_text"]) for c in comparisons]

    # Process in batches of 16
    deb_results = []
    for i in range(0, len(deb_batch_pairs), 16):
        batch = deb_batch_pairs[i : i + 16]
        enc = deb_tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            logits = deb_model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        for p in probs:
            p_ent = float(p[deb_ent_idx])
            p_neu = float(p[deb_neu_idx])
            p_con = float(p[deb_con_idx])
            # argmax
            pred_idx = int(p.argmax())
            pred_label = deb_id2label[pred_idx]
            deb_results.append({
                "entailment": round(p_ent, 4),
                "neutral": round(p_neu, 4),
                "contradiction": round(p_con, 4),
                "predicted_label": pred_label,
            })
    deb_time = time.perf_counter() - start_deb
    print(f"DeBERTa completed in {deb_time:.2f}s ({deb_time/max(total_comparisons,1)*1000:.1f} ms/pair)")

    # 5. Run inference for ModernCE
    print("\nRunning inference for ModernCE-base-nli...")
    start_mod = time.perf_counter()
    mod_batch_pairs = [(c["chunk_raw_text"], c["claim_text"]) for c in comparisons]

    mod_results = []
    for i in range(0, len(mod_batch_pairs), 16):
        batch = mod_batch_pairs[i : i + 16]
        enc = mod_tokenizer(batch, padding=True, truncation=True, max_length=2048, return_tensors="pt")
        with torch.no_grad():
            logits = mod_model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        for p in probs:
            p_ent = float(p[mod_ent_idx])
            p_neu = float(p[mod_neu_idx])
            p_con = float(p[mod_con_idx])
            pred_idx = int(p.argmax())
            pred_label = mod_id2label[pred_idx]
            mod_results.append({
                "entailment": round(p_ent, 4),
                "neutral": round(p_neu, 4),
                "contradiction": round(p_con, 4),
                "predicted_label": pred_label,
            })
    mod_time = time.perf_counter() - start_mod
    print(f"ModernCE completed in {mod_time:.2f}s ({mod_time/max(total_comparisons,1)*1000:.1f} ms/pair)")

    # 6. Aggregate results
    transitions = {
        "neutral_to_entailment": 0,
        "neutral_to_contradiction": 0,
        "neutral_to_neutral": 0,
        "entailment_to_neutral": 0,
        "entailment_to_contradiction": 0,
        "entailment_to_entailment": 0,
        "contradiction_to_neutral": 0,
        "contradiction_to_entailment": 0,
        "contradiction_to_contradiction": 0,
        "other": 0,
    }

    deberta_totals = {"entailment": 0, "neutral": 0, "contradiction": 0}
    modernce_totals = {"entailment": 0, "neutral": 0, "contradiction": 0}

    detailed_records = []

    for comp, d_res, m_res in zip(comparisons, deb_results, mod_results):
        qid = comp["question_id"]
        d_pred = d_res["predicted_label"]
        m_pred = m_res["predicted_label"]

        deberta_totals[d_pred] += 1
        modernce_totals[m_pred] += 1

        question_summaries[qid]["deberta_counts"][d_pred] += 1
        question_summaries[qid]["modernce_counts"][m_pred] += 1

        # Transition tracking
        trans_key = f"{d_pred}_to_{m_pred}"
        if trans_key in transitions:
            transitions[trans_key] += 1
        else:
            transitions["other"] += 1

        rec = {
            "question_id": qid,
            "claim_id": comp["claim_id"],
            "claim_text": comp["claim_text"],
            "chunk_number": comp["chunk_number"],
            "chunk_section": comp["chunk_section"],
            "chunk_pages": comp["chunk_pages"],
            "deberta_token_length": comp["deberta_token_length"],
            "deberta_truncated_at_512": comp["deberta_truncated_at_512"],
            "modernce_token_length": comp["modernce_token_length"],
            "modernce_truncated_at_2048": comp["modernce_truncated_at_2048"],
            "deberta": d_res,
            "modernce": m_res,
            "prediction_transition": f"{d_pred.upper()} -> {m_pred.upper()}",
            "is_transition": d_pred != m_pred,
            "chunk_excerpt": comp["chunk_raw_text"][:300] + ("..." if len(comp["chunk_raw_text"]) > 300 else ""),
            "full_chunk_text": comp["chunk_raw_text"],
        }
        detailed_records.append(rec)

    # 7. Identify Key Transition Cases
    neutral_to_non_neutral = [r for r in detailed_records if r["deberta"]["predicted_label"] == "neutral" and r["modernce"]["predicted_label"] != "neutral"]
    non_neutral_to_neutral = [r for r in detailed_records if r["deberta"]["predicted_label"] != "neutral" and r["modernce"]["predicted_label"] == "neutral"]

    # 8. Token length / Truncation statistics
    deb_lengths = [c["deberta_token_length"] for c in comparisons]
    mod_lengths = [c["modernce_token_length"] for c in comparisons]
    deb_truncated_count = sum(1 for c in comparisons if c["deberta_truncated_at_512"])
    mod_truncated_count = sum(1 for c in comparisons if c["modernce_truncated_at_2048"])

    # 9. Build output JSON
    output_json_data = {
        "experiment_metadata": {
            "experiment_name": "Q1-Q5 Head-to-Head NLI Comparison",
            "models_compared": {
                "deberta": "cross-encoder/nli-deberta-v3-base",
                "modernce": "dleemiller/ModernCE-base-nli",
            },
            "num_questions": len(question_summaries),
            "num_claims": total_claims,
            "num_comparisons": total_comparisons,
            "deberta_inference_seconds": round(deb_time, 2),
            "modernce_inference_seconds": round(mod_time, 2),
        },
        "truncation_statistics": {
            "deberta_max_length": 512,
            "deberta_mean_token_length": round(sum(deb_lengths) / max(len(deb_lengths), 1), 1),
            "deberta_max_token_length": max(deb_lengths) if deb_lengths else 0,
            "deberta_truncated_pairs_count": deb_truncated_count,
            "deberta_truncated_pairs_percentage": round(deb_truncated_count / max(total_comparisons, 1) * 100, 1),
            "modernce_max_length": 2048,
            "modernce_mean_token_length": round(sum(mod_lengths) / max(len(mod_lengths), 1), 1),
            "modernce_max_token_length": max(mod_lengths) if mod_lengths else 0,
            "modernce_truncated_pairs_count": mod_truncated_count,
            "modernce_truncated_pairs_percentage": round(mod_truncated_count / max(total_comparisons, 1) * 100, 1),
        },
        "overall_predictions": {
            "deberta": deberta_totals,
            "modernce": modernce_totals,
        },
        "transitions": transitions,
        "question_summaries": question_summaries,
        "neutral_to_non_neutral_cases": neutral_to_non_neutral,
        "non_neutral_to_neutral_cases": non_neutral_to_neutral,
        "all_comparisons": detailed_records,
    }

    # Save JSON
    OUTPUT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(output_json_data, f, indent=2, ensure_ascii=False)
    print(f"\n[4/4] Saved machine-readable results to: {OUTPUT_JSON_PATH}")

    # Build and Save Human-Readable Text Report
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("Q1-Q5 HEAD-TO-HEAD NLI COMPARISON REPORT")
    report_lines.append("Baseline: cross-encoder/nli-deberta-v3-base (512 tokens)")
    report_lines.append("Target:   dleemiller/ModernCE-base-nli (2048 tokens)")
    report_lines.append("=" * 80)
    report_lines.append("")

    report_lines.append("A. INPUTS AUDITED")
    report_lines.append(f"- Questions: 5 questions (Q1 to Q5 from data/queries.json)")
    report_lines.append(f"- Atomic Claims: {total_claims} claims (from multichunk_pilot_atomic_1_5.json)")
    report_lines.append(f"- Chunks per question: 5 retrieved chunks (from outputs/retrieved_contexts/)")
    report_lines.append(f"- Total Claim x Chunk Comparisons: {total_comparisons}")
    report_lines.append("")

    report_lines.append("B. TOKEN LENGTH & TRUNCATION BEHAVIOR")
    report_lines.append(f"- DeBERTa (512 limit):  Mean length = {output_json_data['truncation_statistics']['deberta_mean_token_length']} tokens | Max = {output_json_data['truncation_statistics']['deberta_max_token_length']} tokens")
    report_lines.append(f"  * Truncated at 512 tokens: {deb_truncated_count} / {total_comparisons} pairs ({output_json_data['truncation_statistics']['deberta_truncated_pairs_percentage']}%)")
    report_lines.append(f"- ModernCE (2048 limit): Mean length = {output_json_data['truncation_statistics']['modernce_mean_token_length']} tokens | Max = {output_json_data['truncation_statistics']['modernce_max_token_length']} tokens")
    report_lines.append(f"  * Truncated at 2048 tokens: {mod_truncated_count} / {total_comparisons} pairs ({output_json_data['truncation_statistics']['modernce_truncated_pairs_percentage']}%)")
    report_lines.append("")

    report_lines.append("C. OVERALL RAW PREDICTIONS")
    report_lines.append(f"| Model    | Entailment | Neutral | Contradiction | Total |")
    report_lines.append(f"|----------|------------|---------|---------------|-------|")
    report_lines.append(f"| DeBERTa  | {deberta_totals['entailment']:10d} | {deberta_totals['neutral']:7d} | {deberta_totals['contradiction']:13d} | {total_comparisons:5d} |")
    report_lines.append(f"| ModernCE | {modernce_totals['entailment']:10d} | {modernce_totals['neutral']:7d} | {modernce_totals['contradiction']:13d} | {total_comparisons:5d} |")
    report_lines.append("")

    report_lines.append("D. PREDICTION TRANSITIONS (DeBERTa -> ModernCE)")
    report_lines.append(f"| Transition                      | Count | Percentage |")
    report_lines.append(f"|---------------------------------|-------|------------|")
    for t_name, t_count in transitions.items():
        pct = (t_count / max(total_comparisons, 1)) * 100
        report_lines.append(f"| {t_name:31s} | {t_count:5d} | {pct:9.1f}% |")
    report_lines.append("")

    report_lines.append("E. QUESTION-WISE BREAKDOWN")
    for qid in sorted(question_summaries.keys(), key=lambda x: int(x)):
        qs = question_summaries[qid]
        report_lines.append(f"--- Question {qid} ({qs['num_claims']} claims, {qs['num_comparisons']} comparisons) ---")
        report_lines.append(f"Q: {qs['question']}")
        report_lines.append(f"  DeBERTa:  Entailment={qs['deberta_counts']['entailment']}, Neutral={qs['deberta_counts']['neutral']}, Contradiction={qs['deberta_counts']['contradiction']}")
        report_lines.append(f"  ModernCE: Entailment={qs['modernce_counts']['entailment']}, Neutral={qs['modernce_counts']['neutral']}, Contradiction={qs['modernce_counts']['contradiction']}")
        report_lines.append("")

    report_lines.append("F. IMPORTANT TRANSITION CASES: NEUTRAL -> NON-NEUTRAL (DeBERTa was Neutral, ModernCE became non-Neutral)")
    report_lines.append(f"Total such cases: {len(neutral_to_non_neutral)}")
    report_lines.append("-" * 80)
    for idx, c in enumerate(neutral_to_non_neutral, 1):
        report_lines.append(f"Case {idx}: Q{c['question_id']} / Claim {c['claim_id']} / Chunk {c['chunk_number']} ({c['chunk_section']})")
        report_lines.append(f"  Claim: \"{c['claim_text']}\"")
        report_lines.append(f"  DeBERTa:  Pred={c['deberta']['predicted_label'].upper()} (Ent={c['deberta']['entailment']}, Neu={c['deberta']['neutral']}, Con={c['deberta']['contradiction']}) [TokLen: {c['deberta_token_length']}]")
        report_lines.append(f"  ModernCE: Pred={c['modernce']['predicted_label'].upper()} (Ent={c['modernce']['entailment']}, Neu={c['modernce']['neutral']}, Con={c['modernce']['contradiction']}) [TokLen: {c['modernce_token_length']}]")
        report_lines.append(f"  Evidence Excerpt: \"{c['chunk_excerpt']}\"")
        report_lines.append("")

    report_lines.append("G. IMPORTANT TRANSITION CASES: NON-NEUTRAL -> NEUTRAL (DeBERTa was non-Neutral, ModernCE became Neutral)")
    report_lines.append(f"Total such cases: {len(non_neutral_to_neutral)}")
    report_lines.append("-" * 80)
    for idx, c in enumerate(non_neutral_to_neutral, 1):
        report_lines.append(f"Case {idx}: Q{c['question_id']} / Claim {c['claim_id']} / Chunk {c['chunk_number']} ({c['chunk_section']})")
        report_lines.append(f"  Claim: \"{c['claim_text']}\"")
        report_lines.append(f"  DeBERTa:  Pred={c['deberta']['predicted_label'].upper()} (Ent={c['deberta']['entailment']}, Neu={c['deberta']['neutral']}, Con={c['deberta']['contradiction']}) [TokLen: {c['deberta_token_length']}]")
        report_lines.append(f"  ModernCE: Pred={c['modernce']['predicted_label'].upper()} (Ent={c['modernce']['entailment']}, Neu={c['modernce']['neutral']}, Con={c['modernce']['contradiction']}) [TokLen: {c['modernce_token_length']}]")
        report_lines.append(f"  Evidence Excerpt: \"{c['chunk_excerpt']}\"")
        report_lines.append("")

    report_text = "\n".join(report_lines)
    with open(OUTPUT_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"Saved human-readable report to: {OUTPUT_REPORT_PATH}")

    print("\n" + report_text[:3000])


if __name__ == "__main__":
    main()
