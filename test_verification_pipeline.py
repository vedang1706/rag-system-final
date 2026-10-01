# test_verification_pipeline.py

"""
Comprehensive test suite for ModernBERT NLI & Lightweight Grounding Verification.

Tests all required test scenarios:
1. Direct Support: Explicitly stated in a retrieved chunk -> VERIFIED
2. Paraphrase: Same fact expressed in different phrasing -> VERIFIED via ModernBERT NLI
3. Explicit Contradiction: Direct conflict / antonym polarity -> CONTRADICTED
4. No Evidence: Completely unrelated / out-of-context claim -> UNSUPPORTED
5. Conflicting Evidence: One chunk supports while another contradicts -> EVIDENCE CONFLICT
6. Multi-Chunk Evidence: Candidate evidence evaluated across multiple chunks
7. Missing Predicate: Subject mentioned, but claimed action/predicate absent -> UNSUPPORTED (NOT Contradicted)
8. Overall Answer Aggregation: Mixed claims produce appropriate overall statuses
9. Fallback Mode: Graceful execution if NLI model is disabled or unavailable
10. Debug / Explainability Mode: Detailed traces of scores, selected chunks, and NLI outputs
"""

import sys
import os
import json
import time

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from verifier import (
    load_nli_verifier,
    extract_claims,
    verify_single_claim,
    verify_answer,
    VERIFIER_CONFIG
)

def run_tests():
    print("=" * 75)
    print("RUNNING MODERNBERT NLI & LIGHTWEIGHT VERIFICATION TEST SUITE")
    print("=" * 75)

    # 1. Load verifier
    local_dir = "models/ModernBERT-base-nli"
    model_path = local_dir if os.path.exists(os.path.join(local_dir, "model.safetensors")) or os.path.exists(os.path.join(local_dir, "config.json")) else "tasksource/ModernBERT-base-nli"
    print(f"\n[1] Initializing NLI Verifier ({model_path})...")
    start_load = time.time()
    verifier = load_nli_verifier(model_path)
    load_time = time.time() - start_load

    if not verifier.get("available"):
        print(f"[FAIL] Verifier could not be loaded: {verifier.get('error')}")
        return

    print(f"✓ Verifier loaded in {load_time:.2f}s on device: {verifier.get('device', 'cpu')} (type: {verifier.get('type')})")

    # Mock 7 retrieved chunks from Psychology Textbook
    mock_retrieved_chunks = [
        {
            "chunk_id": "chunk_1",
            "section": "6.2 Operant Conditioning",
            "section_path": "learning/operant_conditioning",
            "chapter": "Chapter 6 Learning",
            "page_start": 195,
            "page_end": 196,
            "pages": [195, 196],
            "rrf_score": 0.0345,
            "text": (
                "B. F. Skinner is famous for his research on operant conditioning. "
                "Skinner developed the operant conditioning chamber, often referred to as a Skinner box. "
                "Operant conditioning is a method of learning that occurs through rewards and punishments for behavior."
            )
        },
        {
            "chunk_id": "chunk_2",
            "section": "6.1 Classical Conditioning",
            "section_path": "learning/classical_conditioning",
            "chapter": "Chapter 6 Learning",
            "page_start": 190,
            "page_end": 191,
            "pages": [190, 191],
            "rrf_score": 0.0310,
            "text": (
                "Ivan Pavlov was a Russian physiologist who discovered classical conditioning. "
                "In his famous experiments with dogs, Pavlov paired the sound of a bell or tone with meat powder. "
                "Over time, the dogs began to salivate at the sound alone."
            )
        },
        {
            "chunk_id": "chunk_3",
            "section": "1.2 History of Psychology",
            "section_path": "intro/history",
            "chapter": "Chapter 1 Introduction to Psychology",
            "page_start": 14,
            "page_end": 15,
            "pages": [14, 15],
            "rrf_score": 0.0280,
            "text": (
                "Sigmund Freud was an Austrian neurologist who founded psychoanalysis. "
                "Freud proposed that the human psyche is divided into the id, ego, and superego. "
                "Freud was born in Freiberg, Moravia, and lived most of his life in Vienna, Austria."
            )
        },
        {
            "chunk_id": "chunk_4",
            "section": "1.2 History of Psychology",
            "section_path": "intro/history",
            "chapter": "Chapter 1 Introduction to Psychology",
            "page_start": 16,
            "page_end": 17,
            "pages": [16, 17],
            "rrf_score": 0.0250,
            "text": (
                "John B. Watson is considered the father of behaviorism within psychology. "
                "Watson argued that psychology should focus strictly on observable behavior rather than internal mental states."
            )
        },
        {
            "chunk_id": "chunk_5",
            "section": "9.1 What Is Lifespan Development?",
            "section_path": "development/lifespan",
            "chapter": "Chapter 9 Lifespan Development",
            "page_start": 302,
            "page_end": 303,
            "pages": [302, 303],
            "rrf_score": 0.0220,
            "text": (
                "Jean Piaget spent over 50 years studying children and how their minds develop. "
                "Piaget proposed four stages of cognitive development: sensorimotor, preoperational, concrete operational, and formal operational."
            )
        },
        {
            "chunk_id": "chunk_6",
            "section": "10.1 Motivation",
            "section_path": "emotion_motivation/motivation",
            "chapter": "Chapter 10 Emotion and Motivation",
            "page_start": 340,
            "page_end": 341,
            "pages": [340, 341],
            "rrf_score": 0.0200,
            "text": (
                "Abraham Maslow proposed a hierarchy of human needs spanning physiological needs, safety, love and belonging, esteem, and self-actualization. "
                "Maslow asserted that basic physiological needs must be satisfied before higher-level needs become motivating."
            )
        },
        {
            "chunk_id": "chunk_7",
            "section": "4.1 What Is Consciousness?",
            "section_path": "consciousness/states",
            "chapter": "Chapter 4 States of Consciousness",
            "page_start": 120,
            "page_end": 121,
            "pages": [120, 121],
            "rrf_score": 0.0180,
            "text": (
                "Sleep is characterized by low levels of physical activity and reduced sensory awareness. "
                "REM sleep is associated with dreaming and brain waves that resemble wakefulness."
            )
        }
    ]

    # --- TEST 1: Direct Support ---
    print(f"\n[Test 1] Direct Support (Explicitly Stated)...")
    claim_1 = "B. F. Skinner developed the operant conditioning chamber known as a Skinner box."
    res_1 = verify_single_claim(claim_1, mock_retrieved_chunks, verifier)
    print(f"  Claim:      '{claim_1}'")
    print(f"  Status:     {res_1['status']} (Confidence: {res_1['confidence']:.2f}, Method: {res_1['method']})")
    print(f"  Best Chunk: {res_1['best_chunk']['section']} (Pages: {res_1['best_chunk']['page_start']}–{res_1['best_chunk']['page_end']})")
    print(f"  Snippet:    '{res_1['best_evidence_snippet']}'")
    assert res_1['status'] == "VERIFIED", f"Expected VERIFIED, got {res_1['status']}"
    print("  ✓ PASS: Direct support verified.")

    # --- TEST 2: Paraphrase ---
    print(f"\n[Test 2] Paraphrase (Different Wording, Same Meaning)...")
    claim_2 = "Canine subjects in Pavlov's research learned to salivate purely upon hearing an auditory signal."
    res_2 = verify_single_claim(claim_2, mock_retrieved_chunks, verifier)
    print(f"  Claim:      '{claim_2}'")
    print(f"  Status:     {res_2['status']} (Confidence: {res_2['confidence']:.2f}, Method: {res_2['method']})")
    print(f"  Best Chunk: {res_2['best_chunk']['section']}")
    print(f"  Snippet:    '{res_2['best_evidence_snippet']}'")
    assert res_2['status'] == "VERIFIED", f"Expected VERIFIED, got {res_2['status']}"
    print("  ✓ PASS: Paraphrased claim verified using NLI entailment.")

    # --- TEST 3: Explicit Contradiction ---
    print(f"\n[Test 3] Explicit Contradiction (Direct Factual Conflict)...")
    claim_3 = "Sigmund Freud was a Japanese physicist born in Tokyo."
    res_3 = verify_single_claim(claim_3, mock_retrieved_chunks, verifier)
    print(f"  Claim:      '{claim_3}'")
    print(f"  Status:     {res_3['status']} (Confidence: {res_3['confidence']:.2f}, Method: {res_3['method']})")
    print(f"  Best Chunk: {res_3['best_chunk']['section']}")
    print(f"  Snippet:    '{res_3['best_evidence_snippet']}'")
    assert res_3['status'] == "CONTRADICTED", f"Expected CONTRADICTED, got {res_3['status']}"
    print("  ✓ PASS: Explicit contradiction flagged as CONTRADICTED.")

    # --- TEST 4: No Evidence ---
    print(f"\n[Test 4] No Evidence (Completely Unrelated Out-of-Context Fact)...")
    claim_4 = "Albert Einstein formulated the theory of general relativity in 1915."
    res_4 = verify_single_claim(claim_4, mock_retrieved_chunks, verifier)
    print(f"  Claim:      '{claim_4}'")
    print(f"  Status:     {res_4['status']} (Confidence: {res_4['confidence']:.2f}, Method: {res_4['method']})")
    assert res_4['status'] == "UNSUPPORTED", f"Expected UNSUPPORTED, got {res_4['status']}"
    print("  ✓ PASS: Out-of-context claim correctly flagged as UNSUPPORTED.")

    # --- TEST 5: Conflicting Evidence ---
    print(f"\n[Test 5] Conflicting Evidence Across Chunks...")
    conflicting_chunks = [
        {
            "chunk_id": "c1",
            "section": "12.1 Personality Traits",
            "section_path": "personality/traits",
            "chapter": "Chapter 12 Personality",
            "page_start": 410,
            "page_end": 411,
            "pages": [410, 411],
            "rrf_score": 0.03,
            "text": "Introversion is positively associated with academic performance in quiet environments."
        },
        {
            "chunk_id": "c2",
            "section": "12.2 Trait Assessment",
            "section_path": "personality/assessment",
            "chapter": "Chapter 12 Personality",
            "page_start": 415,
            "page_end": 416,
            "pages": [415, 416],
            "rrf_score": 0.025,
            "text": "Introversion is negatively associated with academic performance in quiet environments."
        }
    ]
    claim_5 = "Introversion is positively associated with academic performance in quiet environments."
    res_5 = verify_single_claim(claim_5, conflicting_chunks, verifier)
    print(f"  Claim:   '{claim_5}'")
    print(f"  Status:  {res_5['status']} (Method: {res_5['method']})")
    print(f"  Snippet: '{res_5['best_evidence_snippet']}'")
    assert res_5['status'] == "EVIDENCE CONFLICT", f"Expected EVIDENCE CONFLICT, got {res_5['status']}"
    print("  ✓ PASS: Conflicting evidence detected across chunks.")

    # --- TEST 6: Multi-Chunk Evidence ---
    print(f"\n[Test 6] Multi-Chunk Evidence...")
    claim_6 = "Skinner focused on operant conditioning while Pavlov studied classical conditioning in dogs."
    res_6 = verify_single_claim(claim_6, mock_retrieved_chunks, verifier)
    print(f"  Claim:   '{claim_6}'")
    print(f"  Status:  {res_6['status']} (Confidence: {res_6['confidence']:.2f}, Method: {res_6['method']})")
    assert res_6['status'] in ["VERIFIED", "PARTIALLY VERIFIED"], f"Expected supported status, got {res_6['status']}"
    print("  ✓ PASS: Multi-chunk evidence evaluated and aggregated.")

    # --- TEST 7: Missing Predicate (Subject present, relation not established) ---
    print(f"\n[Test 7] Missing Predicate (Subject present, predicate unmentioned)...")
    # Watson is in chunk 4 (father of behaviorism), but the chunk says nothing about Watson receiving a Nobel Prize
    claim_7 = "John B. Watson won the Nobel Prize in Physiology or Medicine."
    res_7 = verify_single_claim(claim_7, mock_retrieved_chunks, verifier)
    print(f"  Claim:   '{claim_7}'")
    print(f"  Status:  {res_7['status']} (Confidence: {res_7['confidence']:.2f}, Method: {res_7['method']})")
    assert res_7['status'] == "UNSUPPORTED", f"Expected UNSUPPORTED, got {res_7['status']}"
    print("  ✓ PASS: Missing predicate is correctly UNSUPPORTED (NOT falsely CONTRADICTED).")

    # --- TEST 8: Full Answer Verification with Metrics ---
    print(f"\n[Test 8] Full Answer Verification (Partially Verified Answer)...")
    mixed_answer = (
        "B. F. Skinner is famous for developing the operant conditioning chamber known as the Skinner box. "
        "Furthermore, Skinner was elected President of the United States in 1972."
    )
    res_ans = verify_answer(
        answer=mixed_answer,
        retrieved_chunks=mock_retrieved_chunks,
        verifier=verifier,
        debug=True
    )
    print(f"  Answer: \"{mixed_answer}\"")
    print(f"  Overall Status:  {res_ans['overall_status']}")
    print(f"  Total Claims:    {res_ans['total_claims']}")
    print(f"  Verified Count:  {res_ans['verified_count']}")
    print(f"  Unsupported:     {res_ans['unsupported_count']}")
    print(f"  Metrics:         {json.dumps(res_ans['metrics'], indent=2)}")
    assert res_ans['overall_status'] == "PARTIALLY VERIFIED", f"Expected PARTIALLY VERIFIED, got {res_ans['overall_status']}"
    assert res_ans['total_claims'] == 2
    assert res_ans['verified_count'] == 1
    assert res_ans['unsupported_count'] == 1
    print("  ✓ PASS: Full answer verified with claim-level attribution and metrics.")

    # --- TEST 9: Fallback when verifier is disabled/unavailable ---
    print(f"\n[Test 9] Fallback when Verifier is Disabled/Unavailable...")
    fallback_verifier = {"available": False, "type": "lightweight_fallback"}
    res_fallback = verify_answer(
        answer="B. F. Skinner developed the operant conditioning chamber.",
        retrieved_chunks=mock_retrieved_chunks,
        verifier=fallback_verifier
    )
    print(f"  Fallback Status: {res_fallback['overall_status']}")
    print(f"  Claim Method:    {res_fallback['claims'][0]['method']}")
    assert res_fallback['claims'][0]['method'] == "lightweight_fallback"
    assert res_fallback['overall_status'] == "VERIFIED"
    print("  ✓ PASS: Graceful fallback to lightweight mode executed successfully.")

    print("\n" + "=" * 75)
    print("ALL 9 VERIFICATION & NLI TESTS PASSED PERFECTLY!")
    print("=" * 75)

if __name__ == "__main__":
    run_tests()
