# verification_1 — Post-Generation Hallucination & Evidence Verification Framework

`verification_1` is the official post-generation verification layer for the RAG pipeline. It evaluates whether generated answers are faithfully entailed by retrieved textbook evidence without modifying or perturbing the frozen upstream RAG pipeline.

---

## 1. System Architecture & ModernCE Verification Layer

The production verification architecture uses **Complete-Answer Multi-Chunk Cross-Encoder NLI** (`dleemiller/ModernCE-base-nli`, ModernBERT architecture, 2048-token max sequence length).

```
                            USER QUERY
                                ↓
                      FROZEN RAG RETRIEVAL
                                ↓
                     FROZEN ANSWER GENERATION
                                ↓
                5 RETRIEVED CHUNKS + COMPLETE ANSWER
                                │
                    [Explicit Abstention Check]
                   /                           \
              (Yes: "Not found")           (No: Content Answer)
                     ↓                              ↓
             STATUS: NOT_FOUND              MODERNCE INDIVIDUAL
             VERDICT: ABSTENTION             CHUNK SCORING (5x)
             (NLI Bypassed)                         ↓
                                            RANK CHUNKS BY
                                            ENTAILMENT PROBABILITY
                                                    ↓
                                            CONSTRUCT TOP-2 PREMISE
                                            (In retrieval order)
                                                    ↓
                                            MODERNCE EVALUATION:
                                            Top-2 vs Complete Answer
                                           /                        \
                                  (ENTAILMENT)                   (NOT ENTAILMENT)
                                       ↓                                ↓
                               STATUS: SUPPORTED              CONSTRUCT TOP-3 PREMISE
                               VERDICT: ENTAILED_BY_TOP2      (In retrieval order)
                               (Stop & Return)                          ↓
                                                              MODERNCE EVALUATION:
                                                              Top-3 vs Complete Answer
                                                             /                        \
                                                    (ENTAILMENT)                (NEUTRAL / CONTRADICTION)
                                                         ↓                                  ↓
                                                 STATUS: SUPPORTED                  STATUS: INSUFFICIENT_EVIDENCE
                                                 VERDICT: ENTAILED_BY_TOP3          VERDICT: NOT_ENTAILED
```

---

## 2. Core Architectural Principles & Rationale

### 2.1 Why Complete-Answer Verification?
- **Premise Dilution & Fragmentation Elimination**: Previous experiments showed that decomposing answers into atomic claims caused massive premise dilution when evaluated against large chunks (~80–97% Neutral). Complete answers retain discourse context and grammatical cohesion.
- **Zero LLM Overhead**: Evaluates the raw generated answer directly, eliminating the latency, prompt drift, and token costs of external LLM-based claim extraction.

### 2.2 Why Combine Multiple Retrieved Chunks?
- Complex domain answers (e.g. psychology, biology) synthesize distinct sub-concepts that naturally span adjacent or distinct textbook sections.
- On single 400-word chunks, ModernCE achieved only **18.4% entailment** across 250 evaluations. Combining Top-2 chunks yielded an immediate **76.0% entailment signal**.

### 2.3 Why Top-2 First, Top-3 as Fallback?
- **Latency Optimization**: Top-2 evaluation runs in ~2.2 seconds on CPU. 76% of all benchmark queries are fully resolved at Top-2 without running Top-3.
- **Selective High-Recall Fallback**: For queries requiring broader contextual coverage, Top-3 evaluation successfully recovered **58.3% of Top-2 Neutral queries**, raising the content answer entailment rate to **93.75%**.

### 2.4 Handling Explicit Abstentions
- Answers such as *"Not found in the provided textbook."* reflect appropriate abstention on out-of-scope queries.
- Evaluating an abstention string against unrelated retrieved chunks causes cross-encoders to predict `CONTRADICTION` ($C > 0.90$). The verifier detects abstentions first, safely bypassing NLI and returning `STATUS_NOT_FOUND` (`ABSTENTION`).

---

## 3. Evidence-Entailment vs. Factual Truth

> **CRITICAL SCIENTIFIC DISTINCTION**:
> ModernCE verification measures **directional natural language inference (entailment)** between the retrieved textbook premise and the generated answer hypothesis:
> $$\text{NLI}(\text{Premise}=\text{Retrieved Evidence},\; \text{Hypothesis}=\text{Generated Answer})$$
>
> - **What It Proves**: The generated answer is grounded in and logically follows from the retrieved textbook context.
> - **What It Does NOT Prove**: It does not prove real-world absolute factual accuracy if the underlying textbook text contains domain errors or if retrieval surfaces incomplete context.

---

## 4. Main Limitation

Complete-answer verification evaluates the holistic premise-hypothesis pair. If an answer contains 4 sentences where 3 are strongly supported and 1 is a minor unsupported detail, the cross-encoder may still assign a high overall entailment score. For applications requiring strict sentence-by-sentence attribution, complete-answer verification serves as a high-throughput primary filter.

---

## 5. Production API Reference (`verification_1/verifier.py`)

### 5.1 Basic Usage

```python
from verification_1.verifier import verify_answer

# Inputs from frozen RAG pipeline
answer = "The scientific method in psychology is an empirical, cyclical process that begins with a theory..."
retrieved_chunks = [
    {"chunk_number": 1, "text": "The scientific method is a circular process..."},
    {"chunk_number": 2, "text": "Psychological research tests hypotheses..."},
    {"chunk_number": 3, "text": "..."},
]

result = verify_answer(answer=answer, retrieved_chunks=retrieved_chunks)

print("Status          :", result.status)            # SUPPORTED
print("Verdict         :", result.verdict)           # ENTAILED_BY_TOP2
print("Entailment Prob :", result.entailment_prob)    # 0.9140
print("Selected Chunks :", result.selected_chunk_numbers) # [1, 3]
print("Explanation     :", result.explanation)
```

### 5.2 Structured Result Schema (`VerificationResult`)

| Field | Type | Description |
| :--- | :--- | :--- |
| `status` | `str` | Conceptual user-facing status (`SUPPORTED`, `INSUFFICIENT_EVIDENCE`, `CONTRADICTED`, `NOT_FOUND`). |
| `verdict` | `str` | Detailed internal verdict (`ENTAILED_BY_TOP2`, `ENTAILED_BY_TOP3`, `NOT_ENTAILED`, `ABSTENTION`). |
| `is_abstention` | `bool` | True if the answer was identified as an explicit abstention. |
| `combination_used` | `Optional[str]` | `"TOP_2"`, `"TOP_3"`, or `None`. |
| `entailment_prob` | `float` | Entailment probability from the final ModernCE evaluation. |
| `neutral_prob` | `float` | Neutral probability from the final ModernCE evaluation. |
| `contradiction_prob` | `float` | Contradiction probability from the final ModernCE evaluation. |
| `predicted_label` | `str` | Predicted NLI label (`"entailment"`, `"neutral"`, `"contradiction"`). |
| `selected_chunk_numbers` | `List[int]` | Indices of the chunks combined in the final premise. |
| `ranked_chunk_numbers` | `List[int]` | All chunk indices ordered by individual entailment probability descending. |
| `individual_chunk_evaluations`| `List[dict]` | Breakdown of scores and logits for each of the 5 candidate chunks. |
| `total_nli_time_sec` | `float` | Total CPU/GPU inference time spent on NLI forward passes. |
| `explanation` | `str` | Human-readable attribution and verdict explanation. |

---

## 6. Directory Structure

```
verification_1/
├── README.md                                  # Comprehensive architecture documentation
├── verifier.py                                # Main production verifier (ModernCE complete-answer)
├── test_verifier.py                           # Unit & integration test suite for verifier
├── modern_nli_matcher.py                      # ModernBERT NLI cross-encoder model manager
├── test_modern_nli_matcher.py                 # Tests for ModernBERT matcher
├── live_verification_demo.py                  # End-to-end verification demo script
├── context_snapshot.py                        # Safe snapshot utility for frozen RAG contexts
├── claim_extractor.py                         # Sentence/claim decomposition tool
├── similarity_matcher.py                      # MiniLM semantic similarity baseline
├── output/
│   └── modernce_50q_complete_answer/          # 50-question empirical experiment audit artifacts
│       ├── raw_results.json                   # Full evaluation traces
│       ├── results_table.csv                  # Tabular spreadsheet
│       ├── experiment_report.md               # Diagnostic report
│       ├── conclusion.md                      # Decisive experiment conclusion
│       ├── timing_report.json                 # Latency breakdowns
│       ├── model_info.json                    # Model configuration & verified label map
│       ├── input_manifest.json                # SHA256 input checksums
│       └── per_question/                      # Individual query traces (Q01.json...Q50.json)
```

---

## 7. Running Verification & Tests

### Run Full Verifier Test Suite
```bash
python -m unittest verification_1/test_verifier.py
```

### Run Live End-to-End Demo
```bash
python verification_1/live_verification_demo.py
```

### Run All Verification Tests
```bash
python -m unittest discover -s verification_1 -p "test_*.py"
```
