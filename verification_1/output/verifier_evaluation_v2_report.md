# Verification Layer Evaluation & Calibration Report (Dataset v2)

- **Dataset File:** `C:\Users\HP\OneDrive\Desktop\RAG_SYSTEM-main_Ritesh\RAG_SYSTEM-main\verification_1\input\verification_dataset_v2.json`
- **Total Examples Evaluated:** 60
- **Class Distribution:** Entailment = 21, Neutral = 17, Contradiction = 22
- **Models Evaluated:**
  - Semantic Similarity: `all-MiniLM-L6-v2` (cosine similarity)
  - Natural Language Inference: `cross-encoder/nli-deberta-v3-base` (cross-encoder)

> **Note:** This is an evaluation and calibration experiment on the verification testbed. It does not claim full hallucination detection in production without end-to-end integration.

---

## 1. Score Distributions by Gold Label

### A. Semantic Similarity (`all-MiniLM-L6-v2`)

| Gold Label | Count | Min | Max | Mean | Median | Std Dev | Q1 (25%) | Q3 (75%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Entailment** | 21 | 0.6404 | 0.8877 | **0.7740** | 0.7845 | 0.0677 | 0.7278 | 0.8251 |
| **Neutral** | 17 | 0.5208 | 0.8486 | **0.6894** | 0.7025 | 0.0925 | 0.6422 | 0.7469 |
| **Contradiction** | 22 | 0.5513 | 0.9520 | **0.8122** | 0.8308 | 0.0961 | 0.7644 | 0.8844 |
| **All Examples** | 60 | 0.5208 | 0.9520 | **0.7640** | 0.7735 | 0.0983 | 0.7011 | 0.8376 |

### B. DeBERTa NLI Predicted Probabilities (`cross-encoder/nli-deberta-v3-base`)

| Gold Label | Metric | Min | Max | Mean | Median | Std Dev | Q1 (25%) | Q3 (75%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Entailment** | Entailment Score | 0.0724 | 0.9985 | **0.9527** | 0.9975 | 0.2017 | 0.9958 | 0.9980 |
| | Neutral Score | 0.0014 | 0.0817 | 0.0069 | 0.0025 | 0.0173 | 0.0019 | 0.0042 |
| | Contradiction Score | 0.0000 | 0.8459 | 0.0403 | 0.0000 | 0.1846 | 0.0000 | 0.0001 |
| **Neutral** | Entailment Score | 0.0000 | 0.9918 | **0.0651** | 0.0001 | 0.2404 | 0.0000 | 0.0002 |
| | Neutral Score | 0.0081 | 0.9998 | 0.9346 | 0.9997 | 0.2403 | 0.9994 | 0.9998 |
| | Contradiction Score | 0.0000 | 0.0013 | 0.0003 | 0.0002 | 0.0003 | 0.0001 | 0.0003 |
| **Contradiction** | Entailment Score | 0.0000 | 0.9955 | **0.0454** | 0.0000 | 0.2122 | 0.0000 | 0.0001 |
| | Neutral Score | 0.0000 | 0.9866 | 0.0521 | 0.0002 | 0.2112 | 0.0001 | 0.0006 |
| | Contradiction Score | 0.0001 | 1.0000 | 0.9025 | 0.9998 | 0.2920 | 0.9992 | 0.9999 |

---

## 2. NLI Signal Alone (Multiclass Evaluation)

- **Overall Accuracy:** **93.33%**
- **Macro F1 Score:** **0.9339**
- **Weighted F1 Score:** **0.9333**

### Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Entailment** | 0.9091 | 0.9524 | **0.9302** | 21 |
| **Neutral** | 0.9412 | 0.9412 | **0.9412** | 17 |
| **Contradiction** | 0.9524 | 0.9091 | **0.9302** | 22 |

### NLI Confusion Matrix

| Gold \ Predicted | Contradiction | Entailment | Neutral | Total |
| :--- | :---: | :---: | :---: | :---: |
| **Contradiction** | **20** | 1 | 1 | 22 |
| **Entailment** | 1 | **20** | 0 | 21 |
| **Neutral** | 0 | 1 | **16** | 17 |

---

## 3. Semantic Similarity Signal Alone (Threshold Sweep)

Semantic similarity was evaluated on the binary task of separating **Entailment (Supported)** from **Non-Entailment (Neutral + Contradiction)**.

### Optimal Thresholds Identified:
- **Best F1 Threshold:** $T = 0.67$ -> **F1: 0.5714** (Precision: 0.4082, Recall: 0.9524, Accuracy: 50.00%)
- **Best Accuracy Threshold:** $T = 0.93$ -> **Accuracy: 63.33%** (F1: 0.0000)

### Threshold Sweep Progression Sample

| Threshold | Accuracy | Precision | Recall | F1 Score | TP | FP | FN | TN |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.30 | 35.0% | 0.3500 | 1.0000 | **0.5185** | 21 | 39 | 0 | 0 |
| 0.35 | 35.0% | 0.3500 | 1.0000 | **0.5185** | 21 | 39 | 0 | 0 |
| 0.40 | 35.0% | 0.3500 | 1.0000 | **0.5185** | 21 | 39 | 0 | 0 |
| 0.45 | 35.0% | 0.3500 | 1.0000 | **0.5185** | 21 | 39 | 0 | 0 |
| 0.50 | 35.0% | 0.3500 | 1.0000 | **0.5185** | 21 | 39 | 0 | 0 |
| 0.55 | 36.7% | 0.3559 | 1.0000 | **0.5250** | 21 | 38 | 0 | 1 |
| 0.60 | 41.7% | 0.3750 | 1.0000 | **0.5455** | 21 | 35 | 0 | 4 |
| 0.65 | 45.0% | 0.3846 | 0.9524 | **0.5479** | 20 | 32 | 1 | 7 |
| 0.70 | 46.7% | 0.3778 | 0.8095 | **0.5152** | 17 | 28 | 4 | 11 |
| 0.75 | 50.0% | 0.3714 | 0.6190 | **0.4643** | 13 | 22 | 8 | 17 |
| 0.80 | 51.7% | 0.3333 | 0.3810 | **0.3556** | 8 | 16 | 13 | 23 |
| 0.85 | 53.3% | 0.1818 | 0.0952 | **0.1250** | 2 | 9 | 19 | 30 |
| 0.90 | 60.0% | 0.0000 | 0.0000 | **0.0000** | 0 | 3 | 21 | 36 |
| 0.95 | 63.3% | 0.0000 | 0.0000 | **0.0000** | 0 | 1 | 21 | 38 |

---

## 4. Hybrid Similarity + NLI Signal Evaluation

### Dual-Threshold Gating (Entailment Verification)
- **Optimal Parameters:** `Similarity >= 0.40` AND `Entailment >= 0.40`
- **Binary Entailment F1:** **0.9302** (Precision: 0.9091, Recall: 0.9524, Accuracy: 95.00%)
- **Confusion Matrix:** TP = 20, FP = 2, FN = 1, TN = 37

### 3-Class Similarity Gating
- **Optimal Gating Threshold:** `Similarity Gate = 0.40`
- **Overall Accuracy:** **93.33%**
- **Macro F1:** **0.9339**

---

## 5. Key Findings & Insights

1. **MiniLM Similarity Limitations:** Contradictions and Entailments continue to exhibit high semantic overlap (mean ~0.76–0.78), meaning cosine similarity alone suffers from high false-positive rates when verifying facts against evidence.
2. **DeBERTa Directional Power:** Cross-encoder NLI provides high precision and recall on factual contradictions and ungrounded statements.
3. **Hybrid Signal Value:** Applying similarity as an initial relevance filter before invoking cross-encoder NLI yields strong performance while enabling computational savings.

---
*Report generated automatically by `verification_1/evaluate_verifier.py`.*