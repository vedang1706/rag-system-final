# verification_1 — Claim Decomposition, Evidence Preservation & Semantic Matching

## 1. Overview & Purpose

`verification_1` is an isolated experimental framework for post-generation verification in RAG systems.

It encompasses three foundational steps:
1. **Step 1 — Sentence-Level Claim Decomposition:** Systematically decomposes a generated answer into sentence-level claim units using deterministic sentence segmentation (spaCy).
2. **Step 2 — Original Evidence Preservation:** Preserves the exact retrieval and prompt context evidence that the LLM received during generation.
3. **Step 3 — Claim-to-Evidence Semantic Matching:** Computes raw semantic similarity scores between each extracted claim and all retrieved/context evidence chunks using the existing `all-MiniLM-L6-v2` embedding infrastructure.

```
Generated Answer ─────────→ Claim Extractor (spaCy) ──→ Sentence-Level Claims
                                                                 │
Original RAG Generation ──→ Context Snapshot ────────→ RAG Snapshots (Exact Evidence)
                                                                 │
                                                       Semantic Matcher (MiniLM)
                                                                 ↓
                                                       Raw Similarity Scores
```

---

## 2. Step 1: Sentence-Level Claim Decomposition

- **Unit of Extraction:** Every grammatical sentence in the generated answer is extracted as a single claim unit.
- **Original Text Preservation:** Preserves sentence text verbatim without summarization, rewriting, or semantic modification.
- **Deterministic Tokenization:** Uses spaCy (`en_core_web_sm` / rule-based `sentencizer`) for reproducible sentence segmentation.
- **Terminology:** We use the term **"sentence-level claim unit"** rather than asserting that every sentence is necessarily an atomic factual claim.

---

## 3. Step 2: Preservation of Original RAG Evidence

### Why Preserve the Original Evidence?
To evaluate whether a generated claim is hallucinated or supported, the verification layer must check the claim against the **exact evidence that the generator originally saw**, without rerunning the retrieval pipeline.

### `retrieved_chunks` vs. `context_chunks`
The RAG pipeline enforces a context limit of `MAX_CONTEXT_CHARS = 20,000` during prompt construction:
- **`retrieved_chunks`:** The full list of candidate chunks returned by the hybrid retrieval system (vector + BM25 + RRF).
- **`context_chunks`:** The exact subset (prefix) of chunks that fit within the 20,000-character context window and were actually passed into the LLM prompt. Later chunks in `retrieved_chunks` that exceed this threshold are excluded.
- **`final_context`:** The exact formatted text block `[Source i] Section: ... | Pages: ...\n<text>` delivered to the LLM.

### Invariant Established
> "For every chunk listed in `context_chunks`, that chunk was actually included in the evidence context supplied to the LLM. No chunk listed in `context_chunks` was omitted due to character limits."

---

## 4. Step 3: Claim-to-Evidence Semantic Matching

### Raw Semantic Similarity Measurement
- Computes cosine similarity between each sentence-level claim vector and each retrieved chunk vector:
  $$\text{similarity} = \mathbf{c}_{\text{claim}} \cdot \mathbf{e}_{\text{chunk}}$$
- Reuses precomputed chunk embeddings from `cache/embeddings.npy` (mapped to `cache/chunks.json`) to avoid redundant re-computation.
- Safely falls back to on-the-fly chunk encoding if an un-cached chunk is encountered.
- Explicitly tracks `in_llm_context: true/false` for every evaluated chunk based on whether it was included in `context_chunks`.

### Strict Boundaries for Step 3
- ❌ **No Similarity Thresholds:** Scores are raw continuous values $[-1.0, 1.0]$.
- ❌ **No Classification / Verdicts:** Does not assign `SUPPORTED`, `CONTRADICTED`, `HALLUCINATED`, or `UNSUPPORTED` labels.
- ❌ **No Confidence / Faithfulness Metrics:** Scores represent solely dense semantic similarity.

---

## 5. Directory Structure

```
verification_1/
├── README.md
├── claim_extractor.py
├── test_claim_extractor.py
├── context_snapshot.py
├── test_context_snapshot.py
├── similarity_matcher.py
├── test_similarity_matcher.py
├── input/
│   └── test_answers.json
└── output/
    ├── claims.json
    ├── rag_snapshots/
    │   ├── INTEG_001.json
    │   └── INTEG_002.json
    └── similarity_scores/
        ├── INTEG_001.json
        └── INTEG_002.json
```

---

## 6. Data Formats & Schemas

### Claim Units (`output/claims.json`)
```json
[
  {
    "query_id": "TEST_001",
    "answer": "Thomas Szasz was a psychiatrist. He argued that mental illness was invented by society.",
    "claims": [
      {
        "claim_id": "TEST_001_C1",
        "sentence_index": 1,
        "text": "Thomas Szasz was a psychiatrist."
      },
      {
        "claim_id": "TEST_001_C2",
        "sentence_index": 2,
        "text": "He argued that mental illness was invented by society."
      }
    ]
  }
]
```

### RAG Snapshot Schema (`output/rag_snapshots/<query_id>.json`)
```json
{
  "query_id": "INTEG_001",
  "question": "What is classical conditioning?",
  "answer": "Classical conditioning is an associative learning process...",
  "retrieved_chunks": [
    {
      "chunk_id": "learning_classical_conditioning_chunk_5",
      "text": "...",
      "section": "6.2 Classical Conditioning",
      "section_path": "learning/classical_conditioning",
      "chapter": "Chapter 6 Learning",
      "page_start": 199,
      "page_end": 200,
      "pages": [199, 200],
      "rrf_score": 0.031818,
      "source": "both",
      "is_supplementary": "False"
    }
  ],
  "context_chunks": [
    {
      "chunk_id": "learning_classical_conditioning_chunk_5",
      "text": "...",
      "section": "6.2 Classical Conditioning",
      "section_path": "learning/classical_conditioning",
      "chapter": "Chapter 6 Learning",
      "page_start": 199,
      "page_end": 200,
      "pages": [199, 200],
      "rrf_score": 0.031818,
      "source": "both",
      "is_supplementary": "False"
    }
  ],
  "final_context": "[Source 1] Section: 6.2 Classical Conditioning | Pages: 199-200\n...",
  "metadata": {
    "max_context_chars": 20000,
    "retrieved_chunk_count": 5,
    "context_chunk_count": 5
  }
}
```

### Semantic Similarity Output Schema (`output/similarity_scores/<query_id>.json`)
```json
{
  "query_id": "INTEG_001",
  "question": "What is classical conditioning?",
  "answer": "Classical conditioning is an associative learning process...",
  "claims": [
    {
      "claim_id": "INTEG_001_C1",
      "sentence_index": 1,
      "text": "Classical conditioning is an associative learning process in which an organism learns to link two stimuli that occur together.",
      "evidence_matches": [
        {
          "chunk_id": "learning_what_is_learning_chunk_2",
          "similarity_score": 0.7499,
          "in_llm_context": true,
          "section": "6.1 What Is Learning?",
          "section_path": "learning/what_is_learning",
          "chapter": "Chapter 6 Learning",
          "page_start": 194,
          "page_end": 194,
          "pages": [194],
          "rrf_score": 0.029911
        },
        {
          "chunk_id": "learning_classical_conditioning_chunk_0",
          "similarity_score": 0.6526,
          "in_llm_context": true,
          "section": "6.2 Classical Conditioning",
          "section_path": "learning/classical_conditioning",
          "chapter": "Chapter 6 Learning",
          "page_start": 195,
          "page_end": 195,
          "pages": [195],
          "rrf_score": 0.031258
        }
      ]
    }
  ],
  "metadata": {
    "embedding_model": "all-MiniLM-L6-v2",
    "similarity_metric": "cosine_similarity",
    "claim_count": 3,
    "evaluated_chunks_count": 5
  }
}
```

---

## 7. How to Run

### Run Claim Extractor
```bash
python verification_1/claim_extractor.py
```

### Run Similarity Matcher on All Snapshots
```bash
python verification_1/similarity_matcher.py
```

### Run Complete Verification Test Suite
```bash
python -m unittest discover -s verification_1 -p "test_*.py"
```
