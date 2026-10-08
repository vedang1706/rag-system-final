# Complete-Answer Multi-Chunk ModernCE Experiment on Questions 1–5

> **Diagnostic Verification Report**: Evaluating whether long-context ModernBERT NLI (`dleemiller/ModernCE-base-nli`) can verify complete generated RAG answers by combining multiple retrieved evidence chunks (Top-2 / Top-3) in original retrieval order without claim decomposition.

## 1. Experiment Objective & Setup

- **Objective**: Test whether multi-chunk evidence combination directly entails complete RAG generated answers under a long-context NLI cross-encoder.
- **NLI Model**: `dleemiller/ModernCE-base-nli` (ModernBERT architecture, 2048 max sequence tokens)
- **Label Mapping (Empirically Verified)**:
  - `Index 0` $\rightarrow$ **Contradiction**
  - `Index 1` $\rightarrow$ **Entailment**
  - `Index 2` $\rightarrow$ **Neutral**
- **Input Sources**:
  - Q1–Q5 Generated Answers: `verification_1/output/multichunk_pilot_atomic_1_5.json`
  - Q1–Q5 Retrieved Chunks: `outputs/retrieved_contexts/{1..5}.txt` (5 chunks per question)
  - Chunk Metadata Cache: `cache/chunks.json`
- **Methodology**:
  1. **Phase A (Individual Chunks)**: Run ModernCE on `complete_answer + chunk_i` for all 5 chunks.
  2. **Phase B (Top-2 Combination)**: Select the two chunks with highest individual entailment probabilities, concatenate in **original retrieval order**, and evaluate. If predicted label is **ENTAILMENT**, stop.
  3. **Phase C (Top-3 Combination)**: If Top-2 is not entailment, concatenate the top 3 chunks in **original retrieval order** and evaluate as final verdict.

---

## 2. Summary Results Table

| Question | Individual Best Entailment | Top-2 Entailment | Top-2 Label | Top-3 Entailment | Top-3 Label | Final Verdict | Total Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Q1** | Chunk 3 (0.3480) | 0.9140 | `ENTAILMENT` | NOT RUN | NOT RUN | **ENTAILMENT** (TOP_2) | 5.33 |
| **Q2** | Chunk 5 (0.0004) | 0.0020 | `CONTRADICTION` | 0.0018 | `CONTRADICTION` | **CONTRADICTION** (TOP_3) | 8.48 |
| **Q3** | Chunk 1 (0.6990) | 0.9701 | `ENTAILMENT` | NOT RUN | NOT RUN | **ENTAILMENT** (TOP_2) | 5.71 |
| **Q4** | Chunk 4 (0.1565) | 0.9373 | `ENTAILMENT` | NOT RUN | NOT RUN | **ENTAILMENT** (TOP_2) | 5.48 |
| **Q5** | Chunk 3 (0.1931) | 0.7060 | `ENTAILMENT` | NOT RUN | NOT RUN | **ENTAILMENT** (TOP_2) | 6.08 |

---

## 3. Overall Execution Statistics

- **Number of Questions**: 5
- **Individual Chunk Evaluations (Phase A)**: 25
- **Top-2 Evaluations (Phase B)**: 5
- **Top-3 Evaluations (Phase C)**: 1
- **Total ModernCE Evaluations**: 31
- **Model Load Time**: 10.677 s
- **Model Warm-up Time**: 0.153 s
- **Total NLI Inference Time**: 31.072 s
- **Average Evaluation Time**: 1.0023 s (1002.31 ms)
- **Total Experiment Wall-Clock Time**: 42.216 s

---

## 4. Question-by-Question Detailed Results

### Question 1: "What is the scientific method in psychology?"

**Complete Generated Answer**:
> "The scientific method in psychology is an empirical, cyclical process that begins with a theory—a broad, evidence‑based explanation of a phenomenon. From that theory a testable, often “if‑then” hypothesis is derived. Researchers then design experiments or observational studies to collect measurable data, analyze the results, and publish them so others can replicate or challenge the findings. Consistent results support the theory, while inconsistencies prompt revision or new hypotheses, making the method self‑correcting and iterative."

#### Phase A: Individual Chunk Evaluations

| Chunk # | Chunk ID | Section | Pages | Tokens | Entailment | Neutral | Contradiction | Predicted Label | Time (s) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `introduction_to_psychology_what_is_` | *1.1 What Is Psychology?* | 20-20 | 612 | 0.2633 | 0.7361 | 0.0006 | `NEUTRAL` | 0.868 |
| 2 | `introduction_to_psychology_what_is_` | *1.1 What Is Psychology?* | 20-20 | 428 | 0.0054 | 0.9928 | 0.0018 | `NEUTRAL` | 0.543 |
| 3 | `psychological_research_why_is_resea` | *2.1 Why Is Research Impor* | 52-52 | 477 | 0.3480 | 0.6512 | 0.0009 | `NEUTRAL` | 0.606 |
| 4 | `introduction_to_psychology_contempo` | *1.3 Contemporary Psycholo* | 30-31 | 653 | 0.0131 | 0.9559 | 0.0309 | `NEUTRAL` | 0.861 |
| 5 | `psychological_research_critical_thi` | *Critical Thinking Questio* | 81-81 | 592 | 0.0013 | 0.9964 | 0.0023 | `NEUTRAL` | 0.786 |

**Ranked Chunks by Entailment Probability**:
1. Rank 1: **Chunk 3** (`psychological_research_why_is_research_important_chunk_6`) — Entailment: `0.3480` (`NEUTRAL`)
1. Rank 2: **Chunk 1** (`introduction_to_psychology_what_is_psychology_chunk_0`) — Entailment: `0.2633` (`NEUTRAL`)
1. Rank 3: **Chunk 4** (`introduction_to_psychology_contemporary_psychology_chunk_1`) — Entailment: `0.0131` (`NEUTRAL`)
1. Rank 4: **Chunk 2** (`introduction_to_psychology_what_is_psychology_chunk_1`) — Entailment: `0.0054` (`NEUTRAL`)
1. Rank 5: **Chunk 5** (`psychological_research_critical_thinking_questions_chunk_0`) — Entailment: `0.0013` (`NEUTRAL`)

#### Phase B: Top-2 Chunk Combination
- **Selected Chunks (Original Retrieval Order)**: Chunks [1, 3] (`introduction_to_psychology_what_is_psychology_chunk_0, psychological_research_why_is_research_important_chunk_6`)
- **Combined Premise Tokens**: 990
- **Probabilities**: Entailment = `0.9140` | Neutral = `0.0849` | Contradiction = `0.0011`
- **Predicted Label**: `ENTAILMENT`
- **Inference Time**: 1.668 s

#### Phase C: Top-3 Combination
- **Status**: `NOT RUN` (Top-2 combination successfully achieved `ENTAILMENT`).

**Final Verdict for Q1**: `ENTAILMENT` via TOP_2 (Chunks [1, 3], Entailment = `0.9140`) in 5.33s total.

---

### Question 2: "What are the basic parts of a neuron?"

**Complete Generated Answer**:
> "Not found in the provided textbook."

#### Phase A: Individual Chunk Evaluations

| Chunk # | Chunk ID | Section | Pages | Tokens | Entailment | Neutral | Contradiction | Predicted Label | Time (s) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `learning_what_is_learning_chunk_1` | *6.1 What Is Learning?* | 194-194 | 558 | 0.0001 | 0.0086 | 0.9913 | `CONTRADICTION` | 0.814 |
| 2 | `biopsychology_human_genetics_chunk_` | *Chapter Outline* | 83-83 | 456 | 0.0001 | 0.0012 | 0.9987 | `CONTRADICTION` | 0.669 |
| 3 | `biopsychology_human_genetics_chunk_` | *Chapter 3 Biopsychology* | 83-83 | 456 | 0.0001 | 0.0012 | 0.9987 | `CONTRADICTION` | 0.659 |
| 4 | `biopsychology_human_genetics_chunk_` | *3.1 Human Genetics* | 83-84 | 555 | 0.0001 | 0.0015 | 0.9984 | `CONTRADICTION` | 0.818 |
| 5 | `biopsychology_cells_of_the_nervous_` | *3.2 Cells of the Nervous * | 94-95 | 575 | 0.0004 | 0.0517 | 0.9479 | `CONTRADICTION` | 0.837 |

**Ranked Chunks by Entailment Probability**:
1. Rank 1: **Chunk 5** (`biopsychology_cells_of_the_nervous_system_chunk_6`) — Entailment: `0.0004` (`CONTRADICTION`)
1. Rank 2: **Chunk 1** (`learning_what_is_learning_chunk_1`) — Entailment: `0.0001` (`CONTRADICTION`)
1. Rank 3: **Chunk 2** (`biopsychology_human_genetics_chunk_0`) — Entailment: `0.0001` (`CONTRADICTION`)
1. Rank 4: **Chunk 3** (`biopsychology_human_genetics_chunk_0`) — Entailment: `0.0001` (`CONTRADICTION`)
1. Rank 5: **Chunk 4** (`biopsychology_human_genetics_chunk_0`) — Entailment: `0.0001` (`CONTRADICTION`)

#### Phase B: Top-2 Chunk Combination
- **Selected Chunks (Original Retrieval Order)**: Chunks [1, 5] (`learning_what_is_learning_chunk_1, biopsychology_cells_of_the_nervous_system_chunk_6`)
- **Combined Premise Tokens**: 1125
- **Probabilities**: Entailment = `0.0020` | Neutral = `0.0714` | Contradiction = `0.9266`
- **Predicted Label**: `CONTRADICTION`
- **Inference Time**: 1.757 s

#### Phase C: Top-3 Chunk Combination (Top-2 was NOT Entailment)
- **Selected Chunks (Original Retrieval Order)**: Chunks [1, 2, 5] (`learning_what_is_learning_chunk_1, biopsychology_human_genetics_chunk_0, biopsychology_cells_of_the_nervous_system_chunk_6`)
- **Combined Premise Tokens**: 1573
- **Probabilities**: Entailment = `0.0018` | Neutral = `0.0563` | Contradiction = `0.9418`
- **Predicted Label**: `CONTRADICTION`
- **Inference Time**: 2.920 s

**Final Verdict for Q2**: `CONTRADICTION` via TOP_3 (Chunks [1, 2, 5], Entailment = `0.0018`) in 8.48s total.

---

### Question 3: "What are the stages of sleep?"

**Complete Generated Answer**:
> "Sleep consists of two main phases: non‑REM (NREM) and REM.  
NREM is divided into three stages:  
- **Stage 1** is a brief transitional period between wakefulness and sleep.  
- **Stage 2** features sleep spindles and K‑complexes, indicating light sleep.  
- **Stage 3** is deep, slow‑wave sleep with the lowest brain‑wave frequency and highest amplitude.  
After the NREM cycle, REM sleep follows, characterized by rapid eye movements and brain‑wave patterns similar to wakefulness."

#### Phase A: Individual Chunk Evaluations

| Chunk # | Chunk ID | Section | Pages | Tokens | Entailment | Neutral | Contradiction | Predicted Label | Time (s) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `states_of_consciousness_stages_of_s` | *4.3 Stages of Sleep* | 129-129 | 658 | 0.6990 | 0.1918 | 0.1092 | `ENTAILMENT` | 0.864 |
| 2 | `states_of_consciousness_sleep_and_w` | *4.2 Sleep and Why We Slee* | 126-127 | 724 | 0.0415 | 0.9542 | 0.0043 | `NEUTRAL` | 0.973 |
| 3 | `states_of_consciousness_introductio` | *Chapter 4 States of Consc* | 121-121 | 381 | 0.0007 | 0.8889 | 0.1104 | `NEUTRAL` | 0.491 |
| 4 | `states_of_consciousness_introductio` | *Chapter Outline* | 121-121 | 381 | 0.0007 | 0.8889 | 0.1104 | `NEUTRAL` | 0.481 |
| 5 | `states_of_consciousness_stages_of_s` | *4.3 Stages of Sleep* | 132-132 | 686 | 0.0004 | 0.9896 | 0.0099 | `NEUTRAL` | 0.915 |

**Ranked Chunks by Entailment Probability**:
1. Rank 1: **Chunk 1** (`states_of_consciousness_stages_of_sleep_chunk_0`) — Entailment: `0.6990` (`ENTAILMENT`)
1. Rank 2: **Chunk 2** (`states_of_consciousness_sleep_and_why_we_sleep_chunk_1`) — Entailment: `0.0415` (`NEUTRAL`)
1. Rank 3: **Chunk 3** (`states_of_consciousness_introduction_chunk_0`) — Entailment: `0.0007` (`NEUTRAL`)
1. Rank 4: **Chunk 4** (`states_of_consciousness_introduction_chunk_0`) — Entailment: `0.0007` (`NEUTRAL`)
1. Rank 5: **Chunk 5** (`states_of_consciousness_stages_of_sleep_chunk_3`) — Entailment: `0.0004` (`NEUTRAL`)

#### Phase B: Top-2 Chunk Combination
- **Selected Chunks (Original Retrieval Order)**: Chunks [1, 2] (`states_of_consciousness_stages_of_sleep_chunk_0, states_of_consciousness_sleep_and_why_we_sleep_chunk_1`)
- **Combined Premise Tokens**: 1259
- **Probabilities**: Entailment = `0.9701` | Neutral = `0.0283` | Contradiction = `0.0016`
- **Predicted Label**: `ENTAILMENT`
- **Inference Time**: 1.981 s

#### Phase C: Top-3 Combination
- **Status**: `NOT RUN` (Top-2 combination successfully achieved `ENTAILMENT`).

**Final Verdict for Q3**: `ENTAILMENT` via TOP_2 (Chunks [1, 2], Entailment = `0.9701`) in 5.71s total.

---

### Question 4: "What is operant conditioning?"

**Complete Generated Answer**:
> "Operant conditioning is a type of associative learning in which an organism learns to link a behavior with its consequence. A consequence that is pleasant (reinforcement) increases the likelihood of the behavior being repeated, while a consequence that is unpleasant (punishment) decreases that likelihood. The consequence can be added (positive) or removed (negative), and the timing of the consequence—soon after the behavior—helps strengthen or weaken the behavior. This process explains how new behaviors, such as riding a bike, can be acquired."

#### Phase A: Individual Chunk Evaluations

| Chunk # | Chunk ID | Section | Pages | Tokens | Entailment | Neutral | Contradiction | Predicted Label | Time (s) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `learning_classical_conditioning_chu` | *6.2 Classical Conditionin* | 195-195 | 630 | 0.0008 | 0.9783 | 0.0209 | `NEUTRAL` | 0.888 |
| 2 | `learning_what_is_learning_chunk_2` | *6.1 What Is Learning?* | 194-194 | 499 | 0.0016 | 0.9971 | 0.0013 | `NEUTRAL` | 0.627 |
| 3 | `learning_operant_conditioning_chunk` | *6.3 Operant Conditioning* | 205-206 | 636 | 0.0026 | 0.9963 | 0.0011 | `NEUTRAL` | 0.820 |
| 4 | `learning_operant_conditioning_chunk` | *6.3 Operant Conditioning* | 204-204 | 617 | 0.1565 | 0.8433 | 0.0001 | `NEUTRAL` | 0.782 |
| 5 | `learning_personal_application_quest` | *Personal Application Ques* | 223-223 | 434 | 0.0003 | 0.9971 | 0.0026 | `NEUTRAL` | 0.550 |

**Ranked Chunks by Entailment Probability**:
1. Rank 1: **Chunk 4** (`learning_operant_conditioning_chunk_0`) — Entailment: `0.1565` (`NEUTRAL`)
1. Rank 2: **Chunk 3** (`learning_operant_conditioning_chunk_2`) — Entailment: `0.0026` (`NEUTRAL`)
1. Rank 3: **Chunk 2** (`learning_what_is_learning_chunk_2`) — Entailment: `0.0016` (`NEUTRAL`)
1. Rank 4: **Chunk 1** (`learning_classical_conditioning_chunk_0`) — Entailment: `0.0008` (`NEUTRAL`)
1. Rank 5: **Chunk 5** (`learning_personal_application_questions_chunk_0`) — Entailment: `0.0003` (`NEUTRAL`)

#### Phase B: Top-2 Chunk Combination
- **Selected Chunks (Original Retrieval Order)**: Chunks [3, 4] (`learning_operant_conditioning_chunk_2, learning_operant_conditioning_chunk_0`)
- **Combined Premise Tokens**: 1148
- **Probabilities**: Entailment = `0.9373` | Neutral = `0.0623` | Contradiction = `0.0004`
- **Predicted Label**: `ENTAILMENT`
- **Inference Time**: 1.811 s

#### Phase C: Top-3 Combination
- **Status**: `NOT RUN` (Top-2 combination successfully achieved `ENTAILMENT`).

**Final Verdict for Q4**: `ENTAILMENT` via TOP_2 (Chunks [3, 4], Entailment = `0.9373`) in 5.48s total.

---

### Question 5: "What is problem-solving in psychology?"

**Complete Generated Answer**:
> "Problem‑solving in psychology is the mental process of identifying a problem and then applying a strategy to find a solution. It involves selecting from various approaches—such as algorithms (step‑by‑step procedures), heuristics (mental shortcuts), or trial and error—and executing them to reach a desired outcome. Successful problem‑solving requires clear problem definition, consideration of possible solutions, and evaluation of their costs and benefits. Common roadblocks include mental set and functional fixedness, which cause people to persist with ineffective approaches."

#### Phase A: Individual Chunk Evaluations

| Chunk # | Chunk ID | Section | Pages | Tokens | Entailment | Neutral | Contradiction | Predicted Label | Time (s) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `thinking_and_intelligence_problem_s` | *7.3 Problem Solving* | 234-234 | 678 | 0.0008 | 0.9966 | 0.0026 | `NEUTRAL` | 0.903 |
| 2 | `thinking_and_intelligence_problem_s` | *7.3 Problem Solving* | 234-235 | 595 | 0.0016 | 0.9967 | 0.0017 | `NEUTRAL` | 0.757 |
| 3 | `thinking_and_intelligence_problem_s` | *7.3 Problem Solving* | 236-238 | 664 | 0.1931 | 0.8035 | 0.0033 | `NEUTRAL` | 0.871 |
| 4 | `thinking_and_intelligence_problem_s` | *7.3 Problem Solving* | 235-236 | 679 | 0.0014 | 0.9959 | 0.0027 | `NEUTRAL` | 0.886 |
| 5 | `stress_lifestyle_and_health_regulat` | *14.4 Regulation of Stress* | 524-524 | 660 | 0.0008 | 0.9980 | 0.0013 | `NEUTRAL` | 0.859 |

**Ranked Chunks by Entailment Probability**:
1. Rank 1: **Chunk 3** (`thinking_and_intelligence_problem_solving_chunk_3`) — Entailment: `0.1931` (`NEUTRAL`)
1. Rank 2: **Chunk 2** (`thinking_and_intelligence_problem_solving_chunk_1`) — Entailment: `0.0016` (`NEUTRAL`)
1. Rank 3: **Chunk 4** (`thinking_and_intelligence_problem_solving_chunk_2`) — Entailment: `0.0014` (`NEUTRAL`)
1. Rank 4: **Chunk 1** (`thinking_and_intelligence_problem_solving_chunk_0`) — Entailment: `0.0008` (`NEUTRAL`)
1. Rank 5: **Chunk 5** (`stress_lifestyle_and_health_regulation_of_stress_chunk_0`) — Entailment: `0.0008` (`NEUTRAL`)

#### Phase B: Top-2 Chunk Combination
- **Selected Chunks (Original Retrieval Order)**: Chunks [2, 3] (`thinking_and_intelligence_problem_solving_chunk_1, thinking_and_intelligence_problem_solving_chunk_3`)
- **Combined Premise Tokens**: 1150
- **Probabilities**: Entailment = `0.7060` | Neutral = `0.2927` | Contradiction = `0.0013`
- **Predicted Label**: `ENTAILMENT`
- **Inference Time**: 1.807 s

#### Phase C: Top-3 Combination
- **Status**: `NOT RUN` (Top-2 combination successfully achieved `ENTAILMENT`).

**Final Verdict for Q5**: `ENTAILMENT` via TOP_2 (Chunks [2, 3], Entailment = `0.7060`) in 6.08s total.

---

## 5. Key Findings and Analysis

1. **Multi-Chunk Synergistic Entailment**: Combining relevant retrieved chunks allows ModernCE to verify multi-faceted generated answers in a single forward pass without claim extraction.
2. **Context Capacity**: ModernBERT's 2048-token context window accommodated all Top-2 and Top-3 combined chunks without any sequence truncation.
3. **Question 2 Behavior (Unanswerable/Missing Context)**: For unanswerable queries where the RAG response was *"Not found in the provided textbook"*, ModernCE appropriately predicted `CONTRADICTION` / `NEUTRAL` against the retrieved passages, avoiding hallucinated entailment.
4. **Inference Latency**: The adaptive Top-2 $\rightarrow$ Top-3 combination strategy minimized unnecessary evaluations, stopping at Top-2 whenever sufficient evidence was found.
