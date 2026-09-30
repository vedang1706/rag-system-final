# utils/batch.py

import json
import time
import pandas as pd
from typing import List, Dict, Callable
import streamlit as st

from utils.citation import (
    find_exact_citations,
    build_references_from_citations
)


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

API_DELAY = 1.5  # seconds between API calls


# ─────────────────────────────────────────────
# FUNCTION 1 — Normalize Query Input
# ─────────────────────────────────────────────

def normalize_queries(data) -> List[Dict]:
    """
    Normalizes queries from either JSON or CSV format
    into a standard list of {query_id, question} dicts.

    Handles multiple possible key names:
    - query_id / id / ID
    - question / query / text
    """
    normalized = []

    for item in data:
        if isinstance(item, dict):
            query_id = (item.get("query_id") or
                        item.get("id") or
                        item.get("ID") or
                        str(len(normalized) + 1))

            question = (item.get("question") or
                        item.get("query") or
                        item.get("text") or
                        item.get("Question") or "")
        else:
            query_id = str(len(normalized) + 1)
            question = str(item)

        if question.strip():
            normalized.append({
                "query_id": str(query_id),
                "question": question.strip()
            })

    return normalized


# ─────────────────────────────────────────────
# FUNCTION 2 — Process Single Query
# ─────────────────────────────────────────────

def process_single(query_id: str,
                   question: str,
                   components: dict) -> dict:
    """
    Runs the full RAG pipeline for one query.
    Returns one CSV row.
    """
    from retrieve import retrieve
    from generate import generate_answer

    # Retrieve
    retrieved = retrieve(
        query      = question,
        collection = components["collection"],
        model      = components["model"],
        bm25       = components["bm25"],
        chunks     = components["chunks"],
        top_k      = 7
    )

    # Generate
    result = generate_answer(
        client           = components["client"],
        query_id         = query_id,
        question         = question,
        retrieved_chunks = retrieved
    )

    # Find exact citations
    citations = find_exact_citations(
        answer           = result["answer"],
        retrieved_chunks = retrieved
    )

    # Build precise references from exact citations
    if citations:
        references = build_references_from_citations(citations)
    else:
        # Fallback to all retrieved chunks
        import json as _json
        references = _json.loads(result["references"])

    # Build context string
    context_str = " | ".join([
        f"[{c['section']} p.{c['page_start']}-{c['page_end']}]"
        f" {c['text'][:150]}"
        for c in retrieved[:3]
    ])

    return {
        "ID":         query_id,
        "context":    context_str,
        "answer":     result["answer"],
        "references": json.dumps(references)
    }


# ─────────────────────────────────────────────
# FUNCTION 3 — Process Batch With Progress
# ─────────────────────────────────────────────

def process_batch(queries: List[Dict],
                  components: dict,
                  progress_callback: Callable = None
                  ) -> pd.DataFrame:
    """
    Processes all queries with progress tracking.
    Shows progress bar in Streamlit UI.

    Returns DataFrame ready for CSV download.
    """
    results   = []
    total     = len(queries)
    bar       = st.progress(0, text="Starting batch processing...")
    status    = st.empty()
    error_log = []

    for i, query in enumerate(queries):
        query_id = query["query_id"]
        question = query["question"]

        status.info(
            f"Processing {i+1}/{total}: {question[:60]}..."
        )

        try:
            row = process_single(query_id, question, components)
            results.append(row)

        except Exception as e:
            error_log.append(f"Query {query_id}: {str(e)}")
            results.append({
                "ID":         query_id,
                "context":    "Error during processing",
                "answer":     "Not found in the provided textbook.",
                "references": json.dumps({"sections": [], "pages": []})
            })

        # Update progress
        progress = int((i + 1) / total * 100)
        bar.progress(progress, text=f"Processed {i+1}/{total} queries")

        # Rate limit delay
        if i < total - 1:
            time.sleep(API_DELAY)

    bar.progress(100, text="Batch complete!")
    status.empty()

    if error_log:
        st.warning(f"{len(error_log)} queries had errors:")
        for err in error_log:
            st.text(err)

    df = pd.DataFrame(results,
                      columns=["ID", "context", "answer", "references"])
    return df


# ─────────────────────────────────────────────
# FUNCTION 4 — Load JSON Queries
# ─────────────────────────────────────────────

def load_json_queries(uploaded_file) -> List[Dict]:
    """
    Loads and normalizes queries from uploaded JSON file.
    """
    try:
        data    = json.load(uploaded_file)
        queries = normalize_queries(data)
        return queries
    except json.JSONDecodeError as e:
        st.error(f"Invalid JSON file: {e}")
        return []


# ─────────────────────────────────────────────
# FUNCTION 5 — Load CSV Queries
# ─────────────────────────────────────────────

def load_csv_queries(uploaded_file) -> List[Dict]:
    """
    Loads and normalizes queries from uploaded CSV file.
    """
    try:
        import pandas as pd
        df      = pd.read_csv(uploaded_file)
        records = df.to_dict('records')
        queries = normalize_queries(records)
        return queries
    except Exception as e:
        st.error(f"Invalid CSV file: {e}")
        return []


# ─────────────────────────────────────────────
# FUNCTION 6 — Convert DataFrame to CSV Download
# ─────────────────────────────────────────────

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """
    Converts DataFrame to CSV bytes for Streamlit download.
    """
    return df.to_csv(index=False).encode('utf-8')