# Multi-Chunk Verification Pilot Report (Questions 1–5)

**Experiment Type:** Controlled Pilot on Real RAG Outputs  
**Dataset Source:** `outputs/submission.csv` + `outputs/retrieved_contexts/{1..5}.txt`  
**Models:** `all-MiniLM-L6-v2` (Similarity) + `cross-encoder/nli-deberta-v3-base` (NLI)  
**Pilot Decision Rule:** $Sim \ge 0.40 \land P(Ent) \ge 0.50 \implies 	ext{SUPPORT}$; $Sim \ge 0.40 \land P(Contra) \ge 0.50 \implies 	ext{CONTRADICTION}$

---

## Executive Summary

| Question ID | Question | Total Claims | Supported | Contradicted | Insufficient | Conflict | Answer Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Q1** | What is the scientific method in psychology? | 4 | 0 | 0 | 4 | 0 | `PARTIALLY_SUPPORTED` |
| **Q2** | What are the basic parts of a neuron? | 1 | 0 | 0 | 1 | 0 | `PARTIALLY_SUPPORTED` |
| **Q3** | What are the stages of sleep? | 2 | 0 | 0 | 2 | 0 | `PARTIALLY_SUPPORTED` |
| **Q4** | What is operant conditioning? | 4 | 0 | 0 | 4 | 0 | `PARTIALLY_SUPPORTED` |
| **Q5** | What is problem-solving in psychology? | 4 | 0 | 0 | 4 | 0 | `PARTIALLY_SUPPORTED` |

---

## Question 1: What is the scientific method in psychology?

**Generated RAG Answer:**
> The scientific method in psychology is an empirical, cyclical process that begins with a theory—a broad, evidence‑based explanation of a phenomenon. From that theory a testable, often “if‑then” hypothesis is derived. Researchers then design experiments or observational studies to collect measurable data, analyze the results, and publish them so others can replicate or challenge the findings. Consistent results support the theory, while inconsistencies prompt revision or new hypotheses, making the method self‑correcting and iterative.

**Answer-Level Status:** `PARTIALLY_SUPPORTED` (0/4 claims supported)

### Claim Breakdown & Multi-Chunk Evidence Grid

#### Claim `1_C1`: *"The scientific method in psychology is an empirical, cyclical process that begins with a theory—a broad, evidence‑based explanation of a phenomenon."*

- **Final Verdict:** 🟡 **INSUFFICIENT**
- **Strongest Supporting Chunk:** Chunk 3 (Similarity: 0.6016, Entailment: 0.0018)

**Chunk-by-Chunk Evidence Matrix:**

| Chunk | Section | Pages | RRF Score | Cosine Sim | P(Entailment) | P(Neutral) | P(Contradiction) | Predicted Label | Candidate? |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 1.1 What Is Psychology? | 20-20 | 0.0331 | 0.5691 | 0.0003 | 0.9484 | 0.0513 | `neutral` | None |
| 2 | 1.1 What Is Psychology? | 20-20 | 0.0297 | 0.4452 | 0.0006 | 0.9993 | 0.0001 | `neutral` | None |
| 3 | 2.1 Why Is Research Important? | 52-52 | 0.0294 | 0.6016 | 0.0018 | 0.9970 | 0.0012 | `neutral` | None |
| 4 | 1.3 Contemporary Psychology | 30-31 | 0.0156 | 0.5275 | 0.0006 | 0.9990 | 0.0004 | `neutral` | None |
| 5 | Critical Thinking Questions | 81-81 | 0.0156 | 0.4830 | 0.0007 | 0.9989 | 0.0005 | `neutral` | None |


#### Claim `1_C2`: *"From that theory a testable, often “if‑then” hypothesis is derived."*

- **Final Verdict:** 🟡 **INSUFFICIENT**
- **Strongest Supporting Chunk:** Chunk 3 (Similarity: 0.5089, Entailment: 0.0024)

**Chunk-by-Chunk Evidence Matrix:**

| Chunk | Section | Pages | RRF Score | Cosine Sim | P(Entailment) | P(Neutral) | P(Contradiction) | Predicted Label | Candidate? |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 1.1 What Is Psychology? | 20-20 | 0.0331 | 0.2919 | 0.0008 | 0.9982 | 0.0010 | `neutral` | None |
| 2 | 1.1 What Is Psychology? | 20-20 | 0.0297 | 0.0484 | 0.0006 | 0.9993 | 0.0001 | `neutral` | None |
| 3 | 2.1 Why Is Research Important? | 52-52 | 0.0294 | 0.5089 | 0.0024 | 0.9972 | 0.0004 | `neutral` | None |
| 4 | 1.3 Contemporary Psychology | 30-31 | 0.0156 | 0.0821 | 0.0006 | 0.9990 | 0.0004 | `neutral` | None |
| 5 | Critical Thinking Questions | 81-81 | 0.0156 | 0.1212 | 0.0007 | 0.9988 | 0.0005 | `neutral` | None |


#### Claim `1_C3`: *"Researchers then design experiments or observational studies to collect measurable data, analyze the results, and publish them so others can replicate or challenge the findings."*

- **Final Verdict:** 🟡 **INSUFFICIENT**
- **Strongest Supporting Chunk:** Chunk 3 (Similarity: 0.4467, Entailment: 0.0010)

**Chunk-by-Chunk Evidence Matrix:**

| Chunk | Section | Pages | RRF Score | Cosine Sim | P(Entailment) | P(Neutral) | P(Contradiction) | Predicted Label | Candidate? |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 1.1 What Is Psychology? | 20-20 | 0.0331 | 0.3976 | 0.0005 | 0.9279 | 0.0716 | `neutral` | None |
| 2 | 1.1 What Is Psychology? | 20-20 | 0.0297 | 0.1894 | 0.0005 | 0.9994 | 0.0001 | `neutral` | None |
| 3 | 2.1 Why Is Research Important? | 52-52 | 0.0294 | 0.4467 | 0.0010 | 0.9980 | 0.0010 | `neutral` | None |
| 4 | 1.3 Contemporary Psychology | 30-31 | 0.0156 | 0.3216 | 0.0005 | 0.9991 | 0.0003 | `neutral` | None |
| 5 | Critical Thinking Questions | 81-81 | 0.0156 | 0.3966 | 0.0004 | 0.9991 | 0.0005 | `neutral` | None |


#### Claim `1_C4`: *"Consistent results support the theory, while inconsistencies prompt revision or new hypotheses, making the method self‑correcting and iterative."*

- **Final Verdict:** 🟡 **INSUFFICIENT**
- **Strongest Supporting Chunk:** Chunk 3 (Similarity: 0.4412, Entailment: 0.0041)

**Chunk-by-Chunk Evidence Matrix:**

| Chunk | Section | Pages | RRF Score | Cosine Sim | P(Entailment) | P(Neutral) | P(Contradiction) | Predicted Label | Candidate? |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 1.1 What Is Psychology? | 20-20 | 0.0331 | 0.2198 | 0.0008 | 0.9979 | 0.0014 | `neutral` | None |
| 2 | 1.1 What Is Psychology? | 20-20 | 0.0297 | 0.0234 | 0.0006 | 0.9993 | 0.0001 | `neutral` | None |
| 3 | 2.1 Why Is Research Important? | 52-52 | 0.0294 | 0.4412 | 0.0041 | 0.9953 | 0.0007 | `neutral` | None |
| 4 | 1.3 Contemporary Psychology | 30-31 | 0.0156 | 0.1689 | 0.0005 | 0.9991 | 0.0004 | `neutral` | None |
| 5 | Critical Thinking Questions | 81-81 | 0.0156 | 0.2322 | 0.0007 | 0.9988 | 0.0005 | `neutral` | None |


---

## Question 2: What are the basic parts of a neuron?

**Generated RAG Answer:**
> Not found in the provided textbook.

**Answer-Level Status:** `PARTIALLY_SUPPORTED` (0/1 claims supported)

### Claim Breakdown & Multi-Chunk Evidence Grid

#### Claim `2_C1`: *"Not found in the provided textbook."*

- **Final Verdict:** 🟡 **INSUFFICIENT**
- **Strongest Supporting Chunk:** Chunk 1 (Similarity: 0.0491, Entailment: 0.0557)

**Chunk-by-Chunk Evidence Matrix:**

| Chunk | Section | Pages | RRF Score | Cosine Sim | P(Entailment) | P(Neutral) | P(Contradiction) | Predicted Label | Candidate? |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 6.1 What Is Learning? | 194-194 | 0.0314 | 0.0491 | 0.0557 | 0.9409 | 0.0034 | `neutral` | None |
| 2 | Chapter Outline | 83-83 | 0.0309 | 0.0717 | 0.0003 | 0.9970 | 0.0026 | `neutral` | None |
| 3 | Chapter 3 Biopsychology | 83-83 | 0.0302 | 0.0717 | 0.0003 | 0.9970 | 0.0026 | `neutral` | None |
| 4 | 3.1 Human Genetics | 83-84 | 0.0295 | 0.0717 | 0.0083 | 0.9914 | 0.0002 | `neutral` | None |
| 5 | 3.2 Cells of the Nervous System | 94-95 | 0.0164 | 0.0151 | 0.0005 | 0.9991 | 0.0003 | `neutral` | None |


---

## Question 3: What are the stages of sleep?

**Generated RAG Answer:**
> Sleep consists of two main phases: non‑REM (NREM) and REM.  
NREM is divided into three stages:  
- **Stage 1** is a brief transitional period between wakefulness and sleep.  
- **Stage 2** features sleep spindles and K‑complexes, indicating light sleep.  
- **Stage 3** is deep, slow‑wave sleep with the lowest brain‑wave frequency and highest amplitude.  
After the NREM cycle, REM sleep follows, characterized by rapid eye movements and brain‑wave patterns similar to wakefulness.

**Answer-Level Status:** `PARTIALLY_SUPPORTED` (0/2 claims supported)

### Claim Breakdown & Multi-Chunk Evidence Grid

#### Claim `3_C1`: *"Sleep consists of two main phases: non‑REM (NREM) and REM.  
NREM is divided into three stages:  
- **Stage 1** is a brief transitional period between wakefulness and sleep.  
- **Stage 2** features sleep spindles and K‑complexes, indicating light sleep.  
- **Stage 3** is deep, slow‑wave sleep with the lowest brain‑wave frequency and highest amplitude."*

- **Final Verdict:** 🟡 **INSUFFICIENT**
- **Strongest Supporting Chunk:** Chunk 1 (Similarity: 0.6607, Entailment: 0.0072)

**Chunk-by-Chunk Evidence Matrix:**

| Chunk | Section | Pages | RRF Score | Cosine Sim | P(Entailment) | P(Neutral) | P(Contradiction) | Predicted Label | Candidate? |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 4.3 Stages of Sleep | 129-129 | 0.0320 | 0.6607 | 0.0072 | 0.9925 | 0.0002 | `neutral` | None |
| 2 | 4.2 Sleep and Why We Sleep | 126-127 | 0.0315 | 0.4328 | 0.0007 | 0.9986 | 0.0006 | `neutral` | None |
| 3 | Chapter 4 States of Consciousness | 121-121 | 0.0304 | 0.4707 | 0.0022 | 0.9968 | 0.0010 | `neutral` | None |
| 4 | Chapter Outline | 121-121 | 0.0303 | 0.4707 | 0.0022 | 0.9968 | 0.0010 | `neutral` | None |
| 5 | 4.3 Stages of Sleep | 132-132 | 0.0272 | 0.4667 | 0.0011 | 0.9988 | 0.0001 | `neutral` | None |


#### Claim `3_C2`: *"After the NREM cycle, REM sleep follows, characterized by rapid eye movements and brain‑wave patterns similar to wakefulness."*

- **Final Verdict:** 🟡 **INSUFFICIENT**
- **Strongest Supporting Chunk:** Chunk 1 (Similarity: 0.6278, Entailment: 0.0064)

**Chunk-by-Chunk Evidence Matrix:**

| Chunk | Section | Pages | RRF Score | Cosine Sim | P(Entailment) | P(Neutral) | P(Contradiction) | Predicted Label | Candidate? |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 4.3 Stages of Sleep | 129-129 | 0.0320 | 0.6278 | 0.0064 | 0.9931 | 0.0004 | `neutral` | None |
| 2 | 4.2 Sleep and Why We Sleep | 126-127 | 0.0315 | 0.3789 | 0.0012 | 0.9986 | 0.0003 | `neutral` | None |
| 3 | Chapter 4 States of Consciousness | 121-121 | 0.0304 | 0.4172 | 0.0016 | 0.9980 | 0.0004 | `neutral` | None |
| 4 | Chapter Outline | 121-121 | 0.0303 | 0.4172 | 0.0016 | 0.9980 | 0.0004 | `neutral` | None |
| 5 | 4.3 Stages of Sleep | 132-132 | 0.0272 | 0.4198 | 0.0011 | 0.9988 | 0.0001 | `neutral` | None |


---

## Question 4: What is operant conditioning?

**Generated RAG Answer:**
> Operant conditioning is a type of associative learning in which an organism learns to link a behavior with its consequence. A consequence that is pleasant (reinforcement) increases the likelihood of the behavior being repeated, while a consequence that is unpleasant (punishment) decreases that likelihood. The consequence can be added (positive) or removed (negative), and the timing of the consequence—soon after the behavior—helps strengthen or weaken the behavior. This process explains how new behaviors, such as riding a bike, can be acquired.

**Answer-Level Status:** `PARTIALLY_SUPPORTED` (0/4 claims supported)

### Claim Breakdown & Multi-Chunk Evidence Grid

#### Claim `4_C1`: *"Operant conditioning is a type of associative learning in which an organism learns to link a behavior with its consequence."*

- **Final Verdict:** 🟡 **INSUFFICIENT**
- **Strongest Supporting Chunk:** Chunk 4 (Similarity: 0.5094, Entailment: 0.0023)

**Chunk-by-Chunk Evidence Matrix:**

| Chunk | Section | Pages | RRF Score | Cosine Sim | P(Entailment) | P(Neutral) | P(Contradiction) | Predicted Label | Candidate? |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 6.2 Classical Conditioning | 195-195 | 0.0312 | 0.7384 | 0.0012 | 0.9986 | 0.0003 | `neutral` | None |
| 2 | 6.1 What Is Learning? | 194-194 | 0.0302 | 0.7174 | 0.0004 | 0.9995 | 0.0001 | `neutral` | None |
| 3 | 6.3 Operant Conditioning | 205-206 | 0.0299 | 0.5581 | 0.0008 | 0.9987 | 0.0005 | `neutral` | None |
| 4 | 6.3 Operant Conditioning | 204-204 | 0.0295 | 0.5094 | 0.0023 | 0.7718 | 0.2259 | `neutral` | None |
| 5 | Personal Application Questions | 223-223 | 0.0167 | 0.4872 | 0.0011 | 0.9773 | 0.0216 | `neutral` | None |


#### Claim `4_C2`: *"A consequence that is pleasant (reinforcement) increases the likelihood of the behavior being repeated, while a consequence that is unpleasant (punishment) decreases that likelihood."*

- **Final Verdict:** 🟡 **INSUFFICIENT**
- **Strongest Supporting Chunk:** Chunk 5 (Similarity: 0.3920, Entailment: 0.0011)

**Chunk-by-Chunk Evidence Matrix:**

| Chunk | Section | Pages | RRF Score | Cosine Sim | P(Entailment) | P(Neutral) | P(Contradiction) | Predicted Label | Candidate? |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 6.2 Classical Conditioning | 195-195 | 0.0312 | 0.3864 | 0.0006 | 0.9991 | 0.0003 | `neutral` | None |
| 2 | 6.1 What Is Learning? | 194-194 | 0.0302 | 0.3163 | 0.0005 | 0.9994 | 0.0001 | `neutral` | None |
| 3 | 6.3 Operant Conditioning | 205-206 | 0.0299 | 0.5674 | 0.0010 | 0.9987 | 0.0004 | `neutral` | None |
| 4 | 6.3 Operant Conditioning | 204-204 | 0.0295 | 0.3153 | 0.0003 | 0.0199 | 0.9797 | `contradiction` | None |
| 5 | Personal Application Questions | 223-223 | 0.0167 | 0.3920 | 0.0011 | 0.9858 | 0.0132 | `neutral` | None |


#### Claim `4_C3`: *"The consequence can be added (positive) or removed (negative), and the timing of the consequence—soon after the behavior—helps strengthen or weaken the behavior."*

- **Final Verdict:** 🟡 **INSUFFICIENT**
- **Strongest Supporting Chunk:** Chunk 4 (Similarity: 0.2901, Entailment: 0.0013)

**Chunk-by-Chunk Evidence Matrix:**

| Chunk | Section | Pages | RRF Score | Cosine Sim | P(Entailment) | P(Neutral) | P(Contradiction) | Predicted Label | Candidate? |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 6.2 Classical Conditioning | 195-195 | 0.0312 | 0.3472 | 0.0006 | 0.9990 | 0.0004 | `neutral` | None |
| 2 | 6.1 What Is Learning? | 194-194 | 0.0302 | 0.2650 | 0.0004 | 0.9993 | 0.0002 | `neutral` | None |
| 3 | 6.3 Operant Conditioning | 205-206 | 0.0299 | 0.5002 | 0.0007 | 0.9989 | 0.0004 | `neutral` | None |
| 4 | 6.3 Operant Conditioning | 204-204 | 0.0295 | 0.2901 | 0.0013 | 0.9725 | 0.0263 | `neutral` | None |
| 5 | Personal Application Questions | 223-223 | 0.0167 | 0.3024 | 0.0007 | 0.9932 | 0.0061 | `neutral` | None |


#### Claim `4_C4`: *"This process explains how new behaviors, such as riding a bike, can be acquired."*

- **Final Verdict:** 🟡 **INSUFFICIENT**
- **Strongest Supporting Chunk:** Chunk 4 (Similarity: 0.2929, Entailment: 0.0008)

**Chunk-by-Chunk Evidence Matrix:**

| Chunk | Section | Pages | RRF Score | Cosine Sim | P(Entailment) | P(Neutral) | P(Contradiction) | Predicted Label | Candidate? |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 6.2 Classical Conditioning | 195-195 | 0.0312 | 0.3872 | 0.0005 | 0.9990 | 0.0005 | `neutral` | None |
| 2 | 6.1 What Is Learning? | 194-194 | 0.0302 | 0.3447 | 0.0003 | 0.9994 | 0.0003 | `neutral` | None |
| 3 | 6.3 Operant Conditioning | 205-206 | 0.0299 | 0.2604 | 0.0007 | 0.9985 | 0.0008 | `neutral` | None |
| 4 | 6.3 Operant Conditioning | 204-204 | 0.0295 | 0.2929 | 0.0008 | 0.4253 | 0.5738 | `contradiction` | None |
| 5 | Personal Application Questions | 223-223 | 0.0167 | 0.2875 | 0.0005 | 0.9862 | 0.0133 | `neutral` | None |


---

## Question 5: What is problem-solving in psychology?

**Generated RAG Answer:**
> Problem‑solving in psychology is the mental process of identifying a problem and then applying a strategy to find a solution. It involves selecting from various approaches—such as algorithms (step‑by‑step procedures), heuristics (mental shortcuts), or trial and error—and executing them to reach a desired outcome. Successful problem‑solving requires clear problem definition, consideration of possible solutions, and evaluation of their costs and benefits. Common roadblocks include mental set and functional fixedness, which cause people to persist with ineffective approaches.

**Answer-Level Status:** `PARTIALLY_SUPPORTED` (0/4 claims supported)

### Claim Breakdown & Multi-Chunk Evidence Grid

#### Claim `5_C1`: *"Problem‑solving in psychology is the mental process of identifying a problem and then applying a strategy to find a solution."*

- **Final Verdict:** 🟡 **INSUFFICIENT**
- **Strongest Supporting Chunk:** Chunk 5 (Similarity: 0.4914, Entailment: 0.0172)

**Chunk-by-Chunk Evidence Matrix:**

| Chunk | Section | Pages | RRF Score | Cosine Sim | P(Entailment) | P(Neutral) | P(Contradiction) | Predicted Label | Candidate? |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 7.3 Problem Solving | 234-234 | 0.0331 | 0.6041 | 0.0009 | 0.9985 | 0.0007 | `neutral` | None |
| 2 | 7.3 Problem Solving | 234-235 | 0.0309 | 0.4727 | 0.0008 | 0.9988 | 0.0004 | `neutral` | None |
| 3 | 7.3 Problem Solving | 236-238 | 0.0159 | 0.5239 | 0.0043 | 0.9946 | 0.0011 | `neutral` | None |
| 4 | 7.3 Problem Solving | 235-236 | 0.0159 | 0.3659 | 0.0049 | 0.9950 | 0.0000 | `neutral` | None |
| 5 | 14.4 Regulation of Stress | 524-524 | 0.0156 | 0.4914 | 0.0172 | 0.9828 | 0.0000 | `neutral` | None |


#### Claim `5_C2`: *"It involves selecting from various approaches—such as algorithms (step‑by‑step procedures), heuristics (mental shortcuts), or trial and error—and executing them to reach a desired outcome."*

- **Final Verdict:** 🟡 **INSUFFICIENT**
- **Strongest Supporting Chunk:** Chunk 5 (Similarity: 0.2166, Entailment: 0.0107)

**Chunk-by-Chunk Evidence Matrix:**

| Chunk | Section | Pages | RRF Score | Cosine Sim | P(Entailment) | P(Neutral) | P(Contradiction) | Predicted Label | Candidate? |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 7.3 Problem Solving | 234-234 | 0.0331 | 0.3468 | 0.0016 | 0.9980 | 0.0005 | `neutral` | None |
| 2 | 7.3 Problem Solving | 234-235 | 0.0309 | 0.6946 | 0.0006 | 0.9989 | 0.0004 | `neutral` | None |
| 3 | 7.3 Problem Solving | 236-238 | 0.0159 | 0.3276 | 0.0024 | 0.9952 | 0.0024 | `neutral` | None |
| 4 | 7.3 Problem Solving | 235-236 | 0.0159 | 0.4652 | 0.0063 | 0.9936 | 0.0001 | `neutral` | None |
| 5 | 14.4 Regulation of Stress | 524-524 | 0.0156 | 0.2166 | 0.0107 | 0.9892 | 0.0000 | `neutral` | None |


#### Claim `5_C3`: *"Successful problem‑solving requires clear problem definition, consideration of possible solutions, and evaluation of their costs and benefits."*

- **Final Verdict:** 🟡 **INSUFFICIENT**
- **Strongest Supporting Chunk:** Chunk 5 (Similarity: 0.3332, Entailment: 0.1005)

**Chunk-by-Chunk Evidence Matrix:**

| Chunk | Section | Pages | RRF Score | Cosine Sim | P(Entailment) | P(Neutral) | P(Contradiction) | Predicted Label | Candidate? |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 7.3 Problem Solving | 234-234 | 0.0331 | 0.4997 | 0.0007 | 0.9985 | 0.0008 | `neutral` | None |
| 2 | 7.3 Problem Solving | 234-235 | 0.0309 | 0.4505 | 0.0007 | 0.9988 | 0.0005 | `neutral` | None |
| 3 | 7.3 Problem Solving | 236-238 | 0.0159 | 0.3848 | 0.0036 | 0.9949 | 0.0015 | `neutral` | None |
| 4 | 7.3 Problem Solving | 235-236 | 0.0159 | 0.3518 | 0.0042 | 0.9957 | 0.0001 | `neutral` | None |
| 5 | 14.4 Regulation of Stress | 524-524 | 0.0156 | 0.3332 | 0.1005 | 0.8995 | 0.0000 | `neutral` | None |


#### Claim `5_C4`: *"Common roadblocks include mental set and functional fixedness, which cause people to persist with ineffective approaches."*

- **Final Verdict:** 🟡 **INSUFFICIENT**
- **Strongest Supporting Chunk:** Chunk 4 (Similarity: 0.3127, Entailment: 0.0061)

**Chunk-by-Chunk Evidence Matrix:**

| Chunk | Section | Pages | RRF Score | Cosine Sim | P(Entailment) | P(Neutral) | P(Contradiction) | Predicted Label | Candidate? |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 7.3 Problem Solving | 234-234 | 0.0331 | 0.3846 | 0.0006 | 0.9983 | 0.0011 | `neutral` | None |
| 2 | 7.3 Problem Solving | 234-235 | 0.0309 | 0.3484 | 0.0008 | 0.9980 | 0.0012 | `neutral` | None |
| 3 | 7.3 Problem Solving | 236-238 | 0.0159 | 0.5485 | 0.0060 | 0.9932 | 0.0008 | `neutral` | None |
| 4 | 7.3 Problem Solving | 235-236 | 0.0159 | 0.3127 | 0.0061 | 0.9939 | 0.0000 | `neutral` | None |
| 5 | 14.4 Regulation of Stress | 524-524 | 0.0156 | 0.4053 | 0.0027 | 0.9973 | 0.0000 | `neutral` | None |


---
