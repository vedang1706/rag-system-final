# RAG System — OpenStax Psychology 2e

WCE ACM Hackathon 2026 | Problem 3

## Overview

A Retrieval-Augmented Generation system that answers questions
using the OpenStax Psychology 2e textbook and returns verifiable
references (sections and page numbers).

## Architecture

```
PDF → Section-Aware Chunking → Embeddings → ChromaDB
                                           ↓
queries.json → Hybrid Retrieval (Vector + BM25) → NVIDIA LLM → submission.csv
```

## Chunking Strategy

- Extracts Table of Contents using PyMuPDF `get_toc()`
- Splits text at section boundaries (not fixed character counts)
- Sections under 400 words → single chunk
- Sections over 400 words → paragraph splits with 50-word overlap
- Every chunk stores: section path, chapter, page start, page end

## Retrieval Strategy

- Vector search via ChromaDB (cosine similarity, all-MiniLM-L6-v2)
- Keyword search via BM25 (rank_bm25)
- Results merged using Reciprocal Rank Fusion (RRF, k=60)
- Supplementary chunks (summaries, review questions) ranked below main content
- Page-overlap deduplication prevents redundant context

## Reference Tracking

References are built directly from chunk metadata — never generated
by the LLM. This guarantees citations match the actual evidence used.

## Setup

### Requirements

- Python 3.10+
- NVIDIA NIM API key (free tier at build.nvidia.com)

### Install

```bash
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac/Linux
pip install -r requirements.txt
```

### Configure

Create a `.env` file in the project root:

```
NVIDIA_API_KEY=nvapi-your-key-here
```

### Add Data Files

Place these in the `data/` folder:

- `book.pdf` — OpenStax Psychology 2e
- `queries.json` — queries with query_id and question fields

## Run

### Step 1 — Build index (run once)

```bash
python src/ingest.py
python src/embed_index.py
```

### Step 2 — Generate submission

```bash
python src/run_pipeline.py
```

Output: `outputs/submission.csv`

### Resume after interruption

The pipeline saves progress after every query.
Simply re-run `python src/run_pipeline.py` to continue.

## Verify Setup

```bash
python verify.py
```

All components should show ✓.

## Output Format

`submission.csv` columns:

- **ID** — query_id from queries.json
- **context** — retrieved textbook text used to answer
- **answer** — generated response grounded in context
- **references** — JSON string: `{"sections": [...], "pages": [...]}`

## Performance

- Answer rate: 100% (50/50 queries)
- Avg pages cited per query: 6.6
- Index build time: ~5 minutes (cached after first run)
- Query processing: ~2.5 seconds per query

## Tech Stack

| Component      | Library                                  |
| -------------- | ---------------------------------------- |
| PDF parsing    | PyMuPDF 1.27.2                           |
| Embeddings     | sentence-transformers (all-MiniLM-L6-v2) |
| Vector DB      | ChromaDB 0.5.3                           |
| Keyword search | rank-bm25                                |
| LLM            | NVIDIA NIM (meta/llama-3.1-8b-instruct)  |
