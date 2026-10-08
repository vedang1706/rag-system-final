# Controlled Evidence Granularity Test on Query 41 (DeBERTa)

**Date:** 2026-10-07 13:08:28  
**Question ID:** `41`  
**Question:** *"What is the role of neurotransmitters in mental health?"*  
**NLI Model:** `cross-encoder/nli-deberta-v3-base` (512 max length)  
**Similarity Model:** `sentence-transformers/all-MiniLM-L6-v2`  
**Condition A:** Full Retrieved Parent Chunk (~400 words) as Premise  
**Condition B:** Top-1 3-Sentence Micro-Unit (~60 words) as Premise  

---

## 1. Executive Summary & Core Results

| Metric | Condition A (Full Chunk Baseline) | Condition B (Top-1 Micro-Unit) | Delta / Change |
| :--- | :---: | :---: | :---: |
| **Entailment Count** | **0** (0.0%) | **3** (30.0%) | **+3 Entailment Recoveries** |
| **Neutral Count** | **10** (100.0%) | **7** (70.0%) | **-3 Neutrals** |
| **Contradiction Count** | **0** (0.0%) | **0** (0.0%) | **+0** |
| **Total NLI Time** | `7.2528 s` (725.3 ms/eval) | `2.5322 s` (253.2 ms/eval) | **2.86x Faster per NLI call** |
| **Lossless Reconstruction** | 5/5 (100.0%) | 5/5 (100.0%) | **100.0% Exact Match** |

---

## 2. Claim-by-Claim Comparison Table

| Claim ID | Claim Text | Parent Chunk | Full / Micro Chars | Full / Micro Sim | Full DeBERTa (Con/Neu/Ent) [Label] | Micro DeBERTa (Con/Neu/Ent) [Label] | Label Changed? |
| :--- | :--- | :---: | :---: | :---: | :--- | :--- | :---: |
| `41_C1` | Neurotransmitters mediate the ... | Chunk 1 | 3040 / 329 | 0.69 / 0.65 | `0.00/1.00/0.00` **`neutral`** | `0.00/0.04/0.96` **`entailment`** | **`YES`** |
| `41_C2` | Neurotransmitters are essentia... | Chunk 1 | 3040 / 637 | 0.56 / 0.60 | `0.00/1.00/0.00` **`neutral`** | `0.00/1.00/0.00` **`neutral`** | **`NO`** |
| `41_C3` | Imbalances in key neurotransmi... | Chunk 2 | 1649 / 460 | 0.48 / 0.45 | `0.00/1.00/0.00` **`neutral`** | `0.00/1.00/0.00` **`neutral`** | **`NO`** |
| `41_C4` | Low serotonin is linked to dep... | Chunk 2 | 1649 / 460 | 0.48 / 0.57 | `0.00/0.99/0.01` **`neutral`** | `0.00/0.01/0.99` **`entailment`** | **`YES`** |
| `41_C5` | Excess dopamine is linked to s... | Chunk 2 | 1649 / 348 | 0.56 / 0.72 | `0.00/1.00/0.00` **`neutral`** | `0.00/0.00/1.00` **`entailment`** | **`YES`** |
| `41_C6` | Psychotropic medications work ... | Chunk 2 | 1649 / 224 | 0.61 / 0.66 | `0.00/1.00/0.00` **`neutral`** | `0.00/1.00/0.00` **`neutral`** | **`NO`** |
| `41_C7` | Agonists mimic neurotransmitte... | Chunk 2 | 1649 / 460 | 0.54 / 0.53 | `0.00/1.00/0.00` **`neutral`** | `0.00/1.00/0.00` **`neutral`** | **`NO`** |
| `41_C8` | Antagonists block the action o... | Chunk 2 | 1649 / 460 | 0.62 / 0.60 | `0.00/1.00/0.00` **`neutral`** | `0.03/0.97/0.00` **`neutral`** | **`NO`** |
| `41_C9` | Reuptake inhibitors prolong th... | Chunk 2 | 1649 / 460 | 0.56 / 0.67 | `0.00/1.00/0.00` **`neutral`** | `0.00/1.00/0.00` **`neutral`** | **`NO`** |
| `41_C10` | Targeting neurotransmitter sys... | Chunk 2 | 1649 / 224 | 0.57 / 0.59 | `0.00/1.00/0.00` **`neutral`** | `0.00/1.00/0.00` **`neutral`** | **`NO`** |

---

## 3. Transition Matrix & Entailment Recovery Analysis

| Transition Type | Count | % of Claims | Interpretation |
| :--- | :---: | :---: | :--- |
| **`Neutral` $\rightarrow$ `Entailment`** | **3** | **30.0%** | **Entailment Recovery (Dilution Removed)** |
| **`Neutral` $\rightarrow$ `Neutral`** | **7** | **70.0%** | Persistent Neutral (High DeBERTa neutral prior) |
| **`Neutral` $\rightarrow$ `Contradiction`** | **0** | **0.0%** | Shift to Contradiction |
| **`Entailment` $\rightarrow$ `Neutral`** | **0** | **0.0%** | Context Loss |
| **`Entailment` $\rightarrow$ `Entailment`** | **0** | **0.0%** | Stable Entailment |

---

## 4. In-Depth Case Studies

### Claim `41_C1`: *"Neurotransmitters mediate the chemical portion of neuronal communication."*
- **Transition:** `neutral_to_entailment` (Entailment Recovery (Premise Dilution Eliminated))
- **Parent Chunk:** `biopsychology_cells_of_the_nervous_system_chunk_6`
- **Condition A (Full Chunk, 3040 chars):**
  - Label: **`neutral`** (Contra: `0.0002`, Neu: `0.9991`, Ent: `0.0007`)
- **Condition B (Top-1 Micro-Unit, Sents 0-2, 329 chars):**
  - Text: *"the length of the axon is an electrical event, and movement of the neurotransmitter across the synaptic space represents the chemical portion of the process. However, there are some specialized connections between neurons that are entirely electrical. In such cases, the neurons are said to communicate via an electrical synapse."*
  - Similarity: `0.6482`
  - Label: **`entailment`** (Contra: `0.0001`, Neu: `0.0371`, Ent: `0.9628`)

### Claim `41_C2`: *"Neurotransmitters are essential for normal brain function."*
- **Transition:** `neutral_to_neutral` (Persistent Neutral (Unresolved Evidence or High Neutral Prior))
- **Parent Chunk:** `biopsychology_cells_of_the_nervous_system_chunk_6`
- **Condition A (Full Chunk, 3040 chars):**
  - Label: **`neutral`** (Contra: `0.0003`, Neu: `0.9991`, Ent: `0.0006`)
- **Condition B (Top-1 Micro-Unit, Sents 6-8, 637 chars):**
  - Text: *"82 3 • Biopsychology Neurotransmitters and Drugs There are several different types of neurotransmitters released by different neurons, and we can speak in broad terms about the kinds of functions associated with different neurotransmitters (Table 3.1). Much of what psychologists know about the functions of neurotransmitters comes from research on the effects of drugs in psychological disorders. Psychologists who take a biological perspective and focus on the physiological causes of behavior assert that psychological disorders like depression and schizophrenia are associated with imbalances in one or more neurotransmitter systems."*
  - Similarity: `0.6005`
  - Label: **`neutral`** (Contra: `0.0005`, Neu: `0.9991`, Ent: `0.0004`)

### Claim `41_C3`: *"Imbalances in key neurotransmitter systems are linked to the symptoms of many psychiatric disorders."*
- **Transition:** `neutral_to_neutral` (Persistent Neutral (Unresolved Evidence or High Neutral Prior))
- **Parent Chunk:** `biopsychology_cells_of_the_nervous_system_chunk_7`
- **Condition A (Full Chunk, 1649 chars):**
  - Label: **`neutral`** (Contra: `0.0007`, Neu: `0.9992`, Ent: `0.0002`)
- **Condition B (Top-1 Micro-Unit, Sents 6-8, 460 chars):**
  - Text: *"In contrast to agonists and antagonists, which both operate by binding to receptor sites, reuptake inhibitors prevent unused neurotransmitters from being transported back to the neuron. This allows neurotransmitters to remain active in the synaptic cleft for longer durations, increasing their effectiveness. Depression, which has been consistently linked with reduced serotonin levels, is commonly treated with selective serotonin reuptake inhibitors (SSRIs)."*
  - Similarity: `0.4528`
  - Label: **`neutral`** (Contra: `0.0001`, Neu: `0.9996`, Ent: `0.0002`)

### Claim `41_C4`: *"Low serotonin is linked to depression."*
- **Transition:** `neutral_to_entailment` (Entailment Recovery (Premise Dilution Eliminated))
- **Parent Chunk:** `biopsychology_cells_of_the_nervous_system_chunk_7`
- **Condition A (Full Chunk, 1649 chars):**
  - Label: **`neutral`** (Contra: `0.0005`, Neu: `0.9939`, Ent: `0.0056`)
- **Condition B (Top-1 Micro-Unit, Sents 6-8, 460 chars):**
  - Text: *"In contrast to agonists and antagonists, which both operate by binding to receptor sites, reuptake inhibitors prevent unused neurotransmitters from being transported back to the neuron. This allows neurotransmitters to remain active in the synaptic cleft for longer durations, increasing their effectiveness. Depression, which has been consistently linked with reduced serotonin levels, is commonly treated with selective serotonin reuptake inhibitors (SSRIs)."*
  - Similarity: `0.5711`
  - Label: **`entailment`** (Contra: `0.0000`, Neu: `0.0060`, Ent: `0.9940`)

### Claim `41_C5`: *"Excess dopamine is linked to schizophrenia."*
- **Transition:** `neutral_to_entailment` (Entailment Recovery (Premise Dilution Eliminated))
- **Parent Chunk:** `biopsychology_cells_of_the_nervous_system_chunk_7`
- **Condition A (Full Chunk, 1649 chars):**
  - Label: **`neutral`** (Contra: `0.0011`, Neu: `0.9988`, Ent: `0.0002`)
- **Condition B (Top-1 Micro-Unit, Sents 3-5, 348 chars):**
  - Text: *"Certain symptoms of schizophrenia are associated with overactive dopamine neurotransmission. The antipsychotics used to treat these symptoms are antagonists for dopamine—they block dopamine’s effects by binding its receptors without activating them. Thus, they prevent dopamine released by one neuron from signaling information to adjacent neurons."*
  - Similarity: `0.7215`
  - Label: **`entailment`** (Contra: `0.0001`, Neu: `0.0027`, Ent: `0.9972`)

### Claim `41_C6`: *"Psychotropic medications work by restoring balance in neurotransmitter systems."*
- **Transition:** `neutral_to_neutral` (Persistent Neutral (Unresolved Evidence or High Neutral Prior))
- **Parent Chunk:** `biopsychology_cells_of_the_nervous_system_chunk_7`
- **Condition A (Full Chunk, 1649 chars):**
  - Label: **`neutral`** (Contra: `0.0014`, Neu: `0.9984`, Ent: `0.0002`)
- **Condition B (Top-1 Micro-Unit, Sents 12-13, 224 chars):**
  - Text: *"Psychotropic drugs are not instant solutions for people suffering from psychological disorders. Often, an individual must take a drug for several weeks before seeing improvement, and many 3.2 • Cells of the Nervous System 83"*
  - Similarity: `0.6564`
  - Label: **`neutral`** (Contra: `0.0000`, Neu: `0.9996`, Ent: `0.0003`)

### Claim `41_C7`: *"Agonists mimic neurotransmitters."*
- **Transition:** `neutral_to_neutral` (Persistent Neutral (Unresolved Evidence or High Neutral Prior))
- **Parent Chunk:** `biopsychology_cells_of_the_nervous_system_chunk_7`
- **Condition A (Full Chunk, 1649 chars):**
  - Label: **`neutral`** (Contra: `0.0012`, Neu: `0.9985`, Ent: `0.0003`)
- **Condition B (Top-1 Micro-Unit, Sents 6-8, 460 chars):**
  - Text: *"In contrast to agonists and antagonists, which both operate by binding to receptor sites, reuptake inhibitors prevent unused neurotransmitters from being transported back to the neuron. This allows neurotransmitters to remain active in the synaptic cleft for longer durations, increasing their effectiveness. Depression, which has been consistently linked with reduced serotonin levels, is commonly treated with selective serotonin reuptake inhibitors (SSRIs)."*
  - Similarity: `0.5343`
  - Label: **`neutral`** (Contra: `0.0008`, Neu: `0.9982`, Ent: `0.0010`)

### Claim `41_C8`: *"Antagonists block the action of neurotransmitters."*
- **Transition:** `neutral_to_neutral` (Persistent Neutral (Unresolved Evidence or High Neutral Prior))
- **Parent Chunk:** `biopsychology_cells_of_the_nervous_system_chunk_7`
- **Condition A (Full Chunk, 1649 chars):**
  - Label: **`neutral`** (Contra: `0.0013`, Neu: `0.9985`, Ent: `0.0002`)
- **Condition B (Top-1 Micro-Unit, Sents 6-8, 460 chars):**
  - Text: *"In contrast to agonists and antagonists, which both operate by binding to receptor sites, reuptake inhibitors prevent unused neurotransmitters from being transported back to the neuron. This allows neurotransmitters to remain active in the synaptic cleft for longer durations, increasing their effectiveness. Depression, which has been consistently linked with reduced serotonin levels, is commonly treated with selective serotonin reuptake inhibitors (SSRIs)."*
  - Similarity: `0.5989`
  - Label: **`neutral`** (Contra: `0.0250`, Neu: `0.9746`, Ent: `0.0004`)

### Claim `41_C9`: *"Reuptake inhibitors prolong the presence of neurotransmitters in the synaptic cleft."*
- **Transition:** `neutral_to_neutral` (Persistent Neutral (Unresolved Evidence or High Neutral Prior))
- **Parent Chunk:** `biopsychology_cells_of_the_nervous_system_chunk_7`
- **Condition A (Full Chunk, 1649 chars):**
  - Label: **`neutral`** (Contra: `0.0005`, Neu: `0.9992`, Ent: `0.0002`)
- **Condition B (Top-1 Micro-Unit, Sents 6-8, 460 chars):**
  - Text: *"In contrast to agonists and antagonists, which both operate by binding to receptor sites, reuptake inhibitors prevent unused neurotransmitters from being transported back to the neuron. This allows neurotransmitters to remain active in the synaptic cleft for longer durations, increasing their effectiveness. Depression, which has been consistently linked with reduced serotonin levels, is commonly treated with selective serotonin reuptake inhibitors (SSRIs)."*
  - Similarity: `0.6717`
  - Label: **`neutral`** (Contra: `0.0000`, Neu: `0.9968`, Ent: `0.0031`)

### Claim `41_C10`: *"Targeting neurotransmitter systems is a primary strategy for treating mental-health conditions."*
- **Transition:** `neutral_to_neutral` (Persistent Neutral (Unresolved Evidence or High Neutral Prior))
- **Parent Chunk:** `biopsychology_cells_of_the_nervous_system_chunk_7`
- **Condition A (Full Chunk, 1649 chars):**
  - Label: **`neutral`** (Contra: `0.0033`, Neu: `0.9964`, Ent: `0.0002`)
- **Condition B (Top-1 Micro-Unit, Sents 12-13, 224 chars):**
  - Text: *"Psychotropic drugs are not instant solutions for people suffering from psychological disorders. Often, an individual must take a drug for several weeks before seeing improvement, and many 3.2 • Cells of the Nervous System 83"*
  - Similarity: `0.5916`
  - Label: **`neutral`** (Contra: `0.0001`, Neu: `0.9996`, Ent: `0.0003`)


---

## 5. Answers to Mandatory Research Questions

1. **Did splitting the SAME retrieved chunks reduce the Neutral problem?**  
   **YES.** Diagnostic evidence shows that narrowing the premise to a 3-sentence micro-unit altered the prediction distribution and increased the entailment probability for focused propositions.

2. **How many Neutral $\rightarrow$ Entailment recoveries occurred?**  
   **3 claim(s)** successfully transitioned from `Neutral` to `Entailment`.

3. **How many Neutral $\rightarrow$ Contradiction cases were introduced?**  
   **0 claim(s)** transitioned from `Neutral` to `Contradiction`.

4. **Did splitting cause context-loss cases (`Entailment` $\rightarrow$ `Neutral`)?**  
   **0 claim(s)** experienced context loss.

5. **Was the micro-unit approach faster or slower for DeBERTa?**  
   **Faster (2.86x speedup).** Full chunk NLI took `725.3 ms/eval` vs. `253.2 ms/eval` for micro-units.

6. **Based on this single Query 41 experiment, is there enough evidence to justify testing smaller evidence passages on more questions?**  
   **YES.** Diagnostic evidence indicates that splitting the parent chunk into 3-sentence micro-units successfully reduced premise dilution, recovering 3 Entailment predictions from previously Neutral baselines without introducing any context-loss cases. This strongly justifies testing smaller evidence passages across a wider benchmark.
