# Verification Layer: 280-Example Benchmark Executive Summary

**Dataset:** `final_dataset_280eg_metadata_fixed.json` ($N=280$, Balanced: 94 Entailment, 94 Neutral, 92 Contradiction)  
**Models:** `all-MiniLM-L6-v2` (Similarity) vs. `cross-encoder/nli-deberta-v3-base` (NLI)  
**Date:** 2026-10-06 16:05:04 | **Status:** 100% Deterministic Benchmark Complete  

---

## 1. Key Numerical Findings

| Metric | Semantic Similarity (`all-MiniLM-L6-v2`) | NLI Cross-Encoder (`nli-deberta-v3-base`) | Hybrid Candidate (`Sim>=0.40 & Ent>=0.40`) |
| :--- | :---: | :---: | :---: |
| **Overall Accuracy** | 69.6% | **96.4%** | **97.9%** |
| **Macro F1** | 0.6320 (Binary) | **0.9645** (3-Class) | **0.9674** (Binary) |
| **Mean Entailment Score** | 0.8093 | 98.9% Precision | 90.0%+ Precision |
| **Mean Contradiction Score** | **0.7686 (High Overlap!)** | 98.9% Precision | High Precision |

---

## 2. Core Takeaways for Project Defense

1. **Similarity $\neq$ Factual Grounding:**
   Factual contradictions have a mean similarity of **0.7686** compared to **0.8093** for true entailments. Cosine similarity alone cannot detect hallucinations or factual inversions.
2. **NLI Delivers High Directional Accuracy:**
   DeBERTa-v3 cross-encoder achieves **96.4% accuracy** and correctly classifies factual support, subtle neutral extrapolations, and contradictions.
3. **Hybrid Architecture is Optimal:**
   Cosine similarity serves as a fast $O(1)$ relevance filter to reject obviously unrelated text, while NLI acts as the precision verification engine.

---
*Artifact generated for viva defense and project documentation.*