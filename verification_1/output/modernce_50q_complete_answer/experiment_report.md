# 50-Question Complete-Answer Multi-Chunk ModernCE Verification Experiment

> **Diagnostic Evaluation Report**: Comprehensive evaluation of long-context ModernBERT cross-encoder (`dleemiller/ModernCE-base-nli`) performing complete-answer verification via adaptive Top-2 / Top-3 retrieved chunk aggregation across all 50 original RAG benchmark queries.

## 1. Executive Summary & Core Results

- **Total Questions Evaluated**: 50 (Questions 1 to 50)
- **Generated Content Answers**: 48
- **Abstention Answers ("Not found in textbook")**: 2
- **Total Chunks Evaluated**: 250 ($50 \times 5$)
- **Total ModernCE Evaluations**: 312 (250 Individual + 50 Top-2 + 12 Top-3)
- **Total Verification Inference Time**: 416.525 s (Average 8.33 s/question)
- **Total Experiment Wall-Clock Time**: 427.249 s

### Final Verdict Distribution:

| Verdict / Outcome | Count | % of All 50 Questions | % of Content Answers ($N=48$) |
| :--- | :---: | :---: | :---: |
| **ENTAILED_BY_TOP2** | **38** | **76.0%** | **79.2%** |
| **ENTAILED_BY_TOP3** | **7** | **14.0%** | **14.6%** |
| **TOTAL ENTAILED** | **45** | **90.0%** | **93.75%** |
| **NOT_ENTAILED** | **5** | **10.0%** | **10.4%** |
| *Abstention Subset* | 2 | 4.0% | -- |

---

## 2. Immutable Inputs & Alignment Verification

- **Questions Source**: `data/queries.json` (50 queries)
- **Generated Answers Source**: `outputs/submission.csv` (50 rows)
- **Retrieved Evidence Source**: `outputs/retrieved_contexts/{1..50}.txt` (50 files, exactly 5 chunks each)
- **Chunk Cache Metadata**: `cache/chunks.json`
- **Proof of Immutability**: All input files were SHA256 checksummed prior to execution and saved in `input_manifest.json`.

---

## 3. Model Configuration

- **Model Identifier**: `dleemiller/ModernCE-base-nli`
- **Architecture**: ModernBERT Cross-Encoder (`max_length=2048` tokens)
- **Inference Direction**: Premise = Retrieved Evidence Chunks; Hypothesis = Complete Generated Answer
- **Label Mapping (Empirically Verified)**: `0` $\rightarrow$ `contradiction`, `1` $\rightarrow$ `entailment`, `2` $\rightarrow$ `neutral`
- **Sequence Truncation**: **0.0%** truncation occurred across all 250 individual and multi-chunk evaluations.

---

## 4. Multi-Stage Execution Breakdown

### 4.1 Stage A — Individual Chunk Evaluations ($N=250$)
- Across all 250 individual chunk evaluations, single chunks rarely contain sufficient multi-sentence context to entail a complete 4-sentence answer.
- **Individual Chunk Label Distribution**: Entailment = 46 (18.4%), Neutral = 190 (76.0%), Contradiction = 14 (5.6%).
- **Best Individual Chunk Per Question ($N=50$)**: Entailment = 27 (54.0%), Neutral = 21 (42.0%), Contradiction = 2 (4.0%).

### 4.2 Stage B — Top-2 Chunk Combination ($N=50$)
- Top-2 chunks combined in original retrieval order produced a massive synergistic entailment jump.
- **Top-2 Entailment Count**: **38 / 50 (76.0%)**
- **Top-2 Neutral Count**: 10 / 50 (20.0%)
- **Top-2 Contradiction Count**: 2 / 50 (4.0%)

### 4.3 Stage C — Top-3 Chunk Combination ($N=12$)
- Only 12 questions required Top-3 evaluation.
- **Top-3 Entailment Recoveries**: **7** question(s) recovered into Entailment.
- **Top-3 Final Neutral**: 3
- **Top-3 Final Contradiction**: 2

---

## 5. Timing & Verification Latency

| Metric | Value |
| :--- | :---: |
| **Model Load Time** | 10.141 s |
| **Model Warm-up Time** | 0.209 s |
| **Total ModernCE NLI Inference Time** | **416.525 s** |
| **Average Inference Time Per Question** | **8.33 s** |
| **Average Inference Time Per Evaluation** | **1335.01 ms** |
| **Average Individual Evaluation Latency** | 1033.9 ms |
| **Average Top-2 Evaluation Latency** | 2250.33 ms |
| **Average Top-3 Evaluation Latency** | 3794.48 ms |
| **Average Evaluations per Question** | 6.24 evals |
| **Total Wall-Clock Time (including I/O & JSON serialization)** | **427.249 s** |

---

## 6. Full 50-Question Summary Table

| QID | Type | Best Indiv. (E) | Top-2 (E) | Top-2 Label | Top-3 (E) | Top-3 Label | Final Verdict | NLI Time (s) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Q1** | CONTENT | C3 (0.348) | 0.914 | `ENT` | -- | -- | **ENTAILED_BY_TOP2** | 5.58 |
| **Q2** | ABSTAIN | C5 (0.000) | 0.002 | `CON` | 0.002 | `CON` | **NOT_ENTAILED** | 8.49 |
| **Q3** | CONTENT | C1 (0.699) | 0.970 | `ENT` | -- | -- | **ENTAILED_BY_TOP2** | 6.11 |
| **Q4** | CONTENT | C4 (0.157) | 0.937 | `ENT` | -- | -- | **ENTAILED_BY_TOP2** | 5.79 |
| **Q5** | CONTENT | C3 (0.193) | 0.706 | `ENT` | -- | -- | **ENTAILED_BY_TOP2** | 6.67 |
| **Q6** | CONTENT | C1 (0.986) | 0.959 | `ENT` | -- | -- | **ENTAILED_BY_TOP2** | 5.99 |
| **Q7** | CONTENT | C1 (0.716) | 0.864 | `ENT` | -- | -- | **ENTAILED_BY_TOP2** | 5.91 |
| **Q8** | CONTENT | C1 (0.990) | 0.978 | `ENT` | -- | -- | **ENTAILED_BY_TOP2** | 7.32 |
| **Q9** | CONTENT | C2 (0.988) | 0.953 | `ENT` | -- | -- | **ENTAILED_BY_TOP2** | 5.49 |
| **Q10** | CONTENT | C3 (0.982) | 0.965 | `ENT` | -- | -- | **ENTAILED_BY_TOP2** | 6.98 |
| **Q11** | CONTENT | C2 (0.049) | 0.094 | `NEU` | 0.359 | `NEU` | **NOT_ENTAILED** | 9.92 |
| **Q12** | CONTENT | C1 (0.851) | 0.991 | `ENT` | -- | -- | **ENTAILED_BY_TOP2** | 7.51 |
| **Q13** | CONTENT | C2 (0.961) | 0.971 | `ENT` | -- | -- | **ENTAILED_BY_TOP2** | 7.37 |
| **Q14** | CONTENT | C3 (0.005) | 0.005 | `NEU` | 0.845 | `ENT` | **ENTAILED_BY_TOP3** | 10.99 |
| **Q15** | CONTENT | C4 (0.046) | 0.341 | `NEU` | 0.930 | `ENT` | **ENTAILED_BY_TOP3** | 10.58 |
| **Q16** | CONTENT | C5 (0.973) | 0.980 | `ENT` | -- | -- | **ENTAILED_BY_TOP2** | 7.92 |
| **Q17** | CONTENT | C1 (0.295) | 0.681 | `ENT` | -- | -- | **ENTAILED_BY_TOP2** | 7.94 |
| **Q18** | CONTENT | C1 (0.984) | 0.968 | `ENT` | -- | -- | **ENTAILED_BY_TOP2** | 6.05 |
| **Q19** | CONTENT | C1 (0.877) | 0.961 | `ENT` | -- | -- | **ENTAILED_BY_TOP2** | 6.96 |
| **Q20** | CONTENT | C3 (0.302) | 0.496 | `NEU` | 0.852 | `ENT` | **ENTAILED_BY_TOP3** | 11.44 |
| **Q21** | CONTENT | C3 (0.970) | 0.941 | `ENT` | -- | -- | **ENTAILED_BY_TOP2** | 7.00 |
| **Q22** | CONTENT | C2 (0.993) | 0.989 | `ENT` | -- | -- | **ENTAILED_BY_TOP2** | 8.51 |
| **Q23** | CONTENT | C1 (0.977) | 0.638 | `ENT` | -- | -- | **ENTAILED_BY_TOP2** | 6.68 |
| **Q24** | CONTENT | C1 (0.384) | 0.741 | `ENT` | -- | -- | **ENTAILED_BY_TOP2** | 6.80 |
| **Q25** | CONTENT | C5 (0.989) | 0.993 | `ENT` | -- | -- | **ENTAILED_BY_TOP2** | 7.24 |
| **Q26** | CONTENT | C3 (0.353) | 0.884 | `ENT` | -- | -- | **ENTAILED_BY_TOP2** | 9.56 |
| **Q27** | CONTENT | C4 (0.972) | 0.975 | `ENT` | -- | -- | **ENTAILED_BY_TOP2** | 8.03 |
| **Q28** | CONTENT | C1 (0.016) | 0.896 | `ENT` | -- | -- | **ENTAILED_BY_TOP2** | 7.97 |
| **Q29** | CONTENT | C3 (0.018) | 0.593 | `ENT` | -- | -- | **ENTAILED_BY_TOP2** | 7.69 |
| **Q30** | CONTENT | C4 (0.807) | 0.263 | `NEU` | 0.901 | `ENT` | **ENTAILED_BY_TOP3** | 11.62 |
| **Q31** | CONTENT | C2 (0.006) | 0.007 | `NEU` | 0.005 | `NEU` | **NOT_ENTAILED** | 13.37 |
| **Q32** | CONTENT | C2 (0.307) | 0.797 | `ENT` | -- | -- | **ENTAILED_BY_TOP2** | 7.77 |
| **Q33** | CONTENT | C1 (0.183) | 0.550 | `ENT` | -- | -- | **ENTAILED_BY_TOP2** | 9.20 |
| **Q34** | CONTENT | C1 (0.978) | 0.897 | `ENT` | -- | -- | **ENTAILED_BY_TOP2** | 4.59 |
| **Q35** | CONTENT | C3 (0.982) | 0.979 | `ENT` | -- | -- | **ENTAILED_BY_TOP2** | 5.75 |
| **Q36** | CONTENT | C1 (0.030) | 0.299 | `NEU` | 0.471 | `NEU` | **NOT_ENTAILED** | 13.47 |
| **Q37** | CONTENT | C1 (0.322) | 0.839 | `ENT` | -- | -- | **ENTAILED_BY_TOP2** | 8.22 |
| **Q38** | CONTENT | C2 (0.894) | 0.846 | `ENT` | -- | -- | **ENTAILED_BY_TOP2** | 10.32 |
| **Q39** | CONTENT | C1 (0.984) | 0.982 | `ENT` | -- | -- | **ENTAILED_BY_TOP2** | 7.68 |
| **Q40** | CONTENT | C5 (0.017) | 0.012 | `NEU` | 0.846 | `ENT` | **ENTAILED_BY_TOP3** | 12.28 |
| **Q41** | CONTENT | C2 (0.417) | 0.376 | `NEU` | 0.908 | `ENT` | **ENTAILED_BY_TOP3** | 11.78 |
| **Q42** | CONTENT | C4 (0.568) | 0.900 | `ENT` | -- | -- | **ENTAILED_BY_TOP2** | 8.50 |
| **Q43** | CONTENT | C3 (0.800) | 0.826 | `ENT` | -- | -- | **ENTAILED_BY_TOP2** | 7.62 |
| **Q44** | CONTENT | C1 (0.931) | 0.833 | `ENT` | -- | -- | **ENTAILED_BY_TOP2** | 7.97 |
| **Q45** | CONTENT | C4 (0.091) | 0.189 | `NEU` | 0.859 | `ENT` | **ENTAILED_BY_TOP3** | 12.76 |
| **Q46** | CONTENT | C3 (0.487) | 0.538 | `ENT` | -- | -- | **ENTAILED_BY_TOP2** | 8.02 |
| **Q47** | CONTENT | C3 (0.713) | 0.906 | `ENT` | -- | -- | **ENTAILED_BY_TOP2** | 7.46 |
| **Q48** | ABSTAIN | C2 (0.001) | 0.000 | `CON` | 0.000 | `CON` | **NOT_ENTAILED** | 11.64 |
| **Q49** | CONTENT | C5 (0.817) | 0.956 | `ENT` | -- | -- | **ENTAILED_BY_TOP2** | 9.88 |
| **Q50** | CONTENT | C1 (0.895) | 0.910 | `ENT` | -- | -- | **ENTAILED_BY_TOP2** | 6.16 |

---

## 7. Failure Analysis & Category Breakdown

### 7.1 Strong Multi-Chunk Synergistic Entailment (Individual Neutral $\rightarrow$ Top-2 ENTAILMENT)
- **Count**: **12 questions** demonstrated synergistic entailment, where no single chunk entailed the answer, but Top-2 chunks together produced strong entailment.
- **Representative Examples**:
  - **Q1** (*"What is the scientific method in psychology?"*): Best Indiv Chunk 3 ($E=0.3480$, `NEUTRAL`) $\rightarrow$ Top-2 Chunks [1, 3] ($E=0.9140$, `ENTAILMENT`).
  - **Q4** (*"What is operant conditioning?"*): Best Indiv Chunk 4 ($E=0.1565$, `NEUTRAL`) $\rightarrow$ Top-2 Chunks [3, 4] ($E=0.9373$, `ENTAILMENT`).
  - **Q5** (*"What is problem-solving in psychology?"*): Best Indiv Chunk 3 ($E=0.1931$, `NEUTRAL`) $\rightarrow$ Top-2 Chunks [2, 3] ($E=0.7060$, `ENTAILMENT`).

### 7.2 Top-3 Recoveries (Top-2 NOT Entailment $\rightarrow$ Top-3 ENTAILMENT)
- **Count**: **7 question(s)** required Top-3 chunks to complete the evidence set.
  - **Q14** (*"What are Freud's contributions to psychology?"*): Top-2 Chunks [3, 5] ($E=0.0055$, `NEUTRAL`) $\rightarrow$ Top-3 Chunks [2, 3, 5] ($E=0.8446$, `ENTAILMENT`).
  - **Q15** (*"What are Gestalt principles?"*): Top-2 Chunks [4, 5] ($E=0.3406$, `NEUTRAL`) $\rightarrow$ Top-3 Chunks [1, 4, 5] ($E=0.9300$, `ENTAILMENT`).
  - **Q20** (*"What is cognitive psychology?"*): Top-2 Chunks [3, 4] ($E=0.4957$, `NEUTRAL`) $\rightarrow$ Top-3 Chunks [1, 3, 4] ($E=0.8519$, `ENTAILMENT`).
  - **Q30** (*"What are psychological disorders?"*): Top-2 Chunks [1, 4] ($E=0.2633$, `NEUTRAL`) $\rightarrow$ Top-3 Chunks [1, 2, 4] ($E=0.9013$, `ENTAILMENT`).
  - **Q40** (*"What are different types of psychological therapies?"*): Top-2 Chunks [1, 5] ($E=0.0122$, `NEUTRAL`) $\rightarrow$ Top-3 Chunks [1, 4, 5] ($E=0.8463$, `ENTAILMENT`).
  - **Q41** (*"What is the role of neurotransmitters in mental health?"*): Top-2 Chunks [2, 3] ($E=0.3762$, `NEUTRAL`) $\rightarrow$ Top-3 Chunks [1, 2, 3] ($E=0.9085$, `ENTAILMENT`).
  - **Q45** (*"What are key takeaways from the Brown v. Board of Education research?"*): Top-2 Chunks [4, 5] ($E=0.1890$, `NEUTRAL`) $\rightarrow$ Top-3 Chunks [1, 4, 5] ($E=0.8591$, `ENTAILMENT`).

### 7.3 Persistent Neutral Cases (Top-2 & Top-3 Failed to Entail)
- **Count**: **3 question(s)**
  - **Q11** (*"What is the history of psychology?"*): Top-2 ($E=0.0938$, `NEUTRAL`), Top-3 ($E=0.3595$, `NEUTRAL`). Root cause: Partial evidence coverage in top retrieved chunks.
  - **Q31** (*"What is the DSM-5?"*): Top-2 ($E=0.0074$, `NEUTRAL`), Top-3 ($E=0.0053$, `NEUTRAL`). Root cause: Partial evidence coverage in top retrieved chunks.
  - **Q36** (*"What are the challenges of cross-cultural psychology?"*): Top-2 ($E=0.2989$, `NEUTRAL`), Top-3 ($E=0.4711$, `NEUTRAL`). Root cause: Partial evidence coverage in top retrieved chunks.

### 7.4 Contradiction & Abstention Analysis
- **Total Contradictions**: 2 questions.
- **Abstention Answers**: 2 questions (e.g. Q2, Q48) explicitly stated *"Not found in the provided textbook"* because the question was out-of-scope. When evaluated against unrelated biology/psychology chunks, ModernCE accurately predicted `CONTRADICTION` ($C > 0.90$), confirming that the model does not hallucinate entailment on missing context.

---

## 8. Comparison with Previous Verification Experiments

| Verification Paradigm | Granularity | Context Window | Entailment Signal Rate | Avg Latency / Query | Practical Feasibility |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Claim-Level DeBERTa Baseline** | Atomic Claims $\times$ Single Chunks | 512 tokens | ~13.6% of pairs (severe premise dilution) | ~57.0 s | Poor (Slow, high neutral rate) |
| **DeBERTa Micro-Units ($k=3$)** | Atomic Claims $\times$ 3-Sentence Units | 512 tokens | Improved (~30% recoveries) | ~12.0 s | Moderate (Requires claim decomposition) |
| **ModernCE Complete-Answer Multi-Chunk** | Complete Answer $\times$ Top-2/Top-3 Chunks | 2048 tokens | **93.75% on content answers** | **8.33 s** | **High (Fast, zero decomposition overhead, strong signal)** |

---
