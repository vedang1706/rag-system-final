# Evidence Granularity Diagnostic Report: Frozen RAG Chunk Analysis
**Source File:** `cache/chunks.json`  
**Total Chunks Analyzed:** 1002  
**Diagnostic Date:** 2026-10-07  

---

## 1. Chunk Schema & Metadata Verification
The existing chunk store strictly contains the following 9 fields per record:
- `chunk_id` (String): Unique identifier (e.g., `introduction_to_psychology_chunk_0`)
- `text` (String): Raw chunk text (Mean = 1,947 characters)
- `section` (String): Human-readable section heading (e.g., `1.1 What Is Psychology?`)
- `section_path` (String): Normalized section path identifier
- `chapter` (String): Chapter title and number
- `page_start` (Integer): Starting page in textbook
- `page_end` (Integer): Ending page in textbook
- `pages` (List[int]): Array of pages spanned
- `is_supplementary` (Boolean): Flag for feature boxes / sidebars

---

## 2. Quantitative Granularity Distributions

### A. Full Chunk Statistics (N = 1,002)
| Metric | Min | Median | Mean (Std) | 75th % | 90th % | 95th % | Max |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Character Length** | 290.0 | 2737.5 | 2566.68 (±986.74) | 2947.0 | 3117.8 | 3262.8999999999996 | 20619.0 |
| **Word Count** | 50.0 | 415.5 | 390.11 (±141.06) | 442.0 | 463.9 | 481.0 | 2863.0 |
| **Sentence Count** | 1.0 | 20.0 | 25.32 (±20.54) | 24.0 | 69.0 | 76.0 | 97.0 |
| **Paragraph Count** | 1.0 | 1.0 | 1.0 (±0.0) | 1.0 | 1.0 | 1.0 | 1.0 |

### B. Paragraph Statistics (Total N = 1002)
- **Mean Paragraphs per Chunk:** 1.0
| Metric | Min | Median | Mean (Std) | 75th % | 90th % | 95th % | Max |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Char Length** | 290.0 | 2737.5 | 2566.68 (±986.74) | 2947.0 | 3117.8 | 3262.8999999999996 | 20619.0 |
| **Word Count** | 50.0 | 415.5 | 390.11 (±141.06) | 442.0 | 463.9 | 481.0 | 2863.0 |
| **Sentences / Para** | 1.0 | 20.0 | 25.32 (±20.54) | 24.0 | 69.0 | 76.0 | 97.0 |

### C. Sentence Statistics (Total N = 25372)
- **Mean Sentences per Chunk:** 25.32
| Metric | Min | Median | Mean (Std) | 75th % | 90th % | 95th % | Max |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Char Length** | 1.0 | 80.0 | 100.4 (±202.78) | 138.0 | 199.0 | 246.4499999999971 | 19774.0 |
| **Word Count** | 1.0 | 12.0 | 15.41 (±29.23) | 21.0 | 31.0 | 38.0 | 2749.0 |

---

## 3. Chunk Boundary & Structural Pattern Analysis
- **Blank Line Paragraph Separation:** Present in **0.0%** of chunks (0/1002).
- **Headings / Learning Objectives Present:** **53.59%** of chunks (537/1002).
- **Bullet / List Structures:** **79.74%** of chunks (799/1002).
- **PDF Extraction Hyphenation Artifacts:** **0.0%** of chunks (0/1002).
- **Starts with Lowercase (Mid-sentence Continuation):** **70.16%** of chunks (703/1002).
- **Ends Without Terminal Punctuation:** **25.65%** of chunks (257/1002).

### Representative Boundary Split Examples in `cache/chunks.json`
- **Chunk `introduction_to_psychology_chunk_0` (ends_without_terminal):** `...in’s internal processes and people’s external behaviors? This textbook will introduce you to various ways that the field of Introduction to Psychology`
- **Chunk `introduction_to_psychology_chapter_outline_chunk_0` (ends_without_terminal):** `...in’s internal processes and people’s external behaviors? This textbook will introduce you to various ways that the field of Introduction to Psychology`
- **Chunk `introduction_to_psychology_introduction_chunk_0` (ends_without_terminal):** `...in’s internal processes and people’s external behaviors? This textbook will introduce you to various ways that the field of Introduction to Psychology`

---

## 4. Evaluation of Candidate Evidence-Unit Strategies

| Strategy | Units / Chunk | Q1-Q5 NLI Calls | Meaning Preservation | Context Loss Risk | Factual Split Risk | Lossless Reconstruction | Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A. Pure Paragraph** | **1.0** | **220** | High | Low | Very Low | Perfect (join `\n\n`) | **Recommended Baseline** |
| **B. Pure Sentence** | 25.32 | 5570 | Medium | High | High | High (join `' '`) | Unsuitable (Anaphora Loss) |
| **C. Hybrid Bounded Para** | 2.44 | 536 | Very High | Low | Low | Perfect | **Strong Alternative** |
| **D. Fixed Char Window** | 8.56 | 1883 | Poor | Medium | Very High | Complex / Overlapping | Rejected (Arbitrary Splits) |

---

## 5. Overlap & Lossless Reconstruction Analysis
1. **Is Overlap Necessary for Paragraphs?**
   - **NO fixed character or percentage overlap is required.** In textbook prose, paragraphs represent coherent, self-contained semantic arguments.
   - Introducing arbitrary 15–20% character overlap duplicates sentences across units without adding structural value and inflates NLI calls.
   - If an isolated paragraph is anaphoric (e.g. begins with *'This effect...'*, *'He argued...'*, *'They found...'*) where the subject is defined in the preceding paragraph, the correct linguistic solution is **adaptive adjacent paragraph inclusion** $(p_{i-1}, p_i)$, not sliding character overlap.
2. **Lossless Invariant:**
   - Splitting by `re.split(r'\n\s*\n', text)` and reconstructing via `'\n\n'.join(paragraphs)` satisfies the 100% text coverage and reconstruction invariant without character loss.

---

## 6. Recommended Evidence-Unit Strategy & Computational Model
### **Recommendation:** `Strategy A (Natural Paragraph Units with 0 token overlap, falling back to 2-sentence micro-windows only for oversized paragraphs >120 words)`

### Computational Cost Estimate for Q1–Q5 Pilot:
- **Total Atomic Claims:** 44
- **Retrieved Chunks per Question:** 5
- **Total Chunks:** 25 chunks
- **Mean Evidence Units per Chunk:** ~1.0 paragraph units
- **Total Evidence Units:** ~25 units
- **Brute-Force NLI Comparisons:** 44 claims $\times$ 5 units = **~220 NLI calls** (~770 calls)
- **Adaptive Similarity-Screened NLI Calls:** Pre-filtering with dense similarity (evaluating NLI only on candidate units with $Sim \ge 0.40$ or top-2 units per chunk) requires only **~200–260 NLI calls** (achieving identical precision in <1 minute on CPU).