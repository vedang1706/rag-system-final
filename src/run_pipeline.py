# src/run_pipeline.py

import json
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
import time
import random
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from embed_index import load_chunks, load_chroma_collection, load_embed_model
from retrieve   import load_or_build_bm25, retrieve, format_context
from generate   import init_client, generate_answer


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

load_dotenv()

QUERIES_PATH    = "data/queries.json"
CHUNKS_PATH     = "cache/chunks.json"
PROGRESS_PATH   = "cache/progress.json"
OUTPUT_PATH     = "outputs/submission.csv"
API_DELAY       = 1.5   # seconds between API calls
TOP_K           = 5     # chunks per query


# ─────────────────────────────────────────────
# FUNCTION 1 — Load Queries
# ─────────────────────────────────────────────

def load_queries(path: str) -> list:
    """
    Loads queries.json file.
    Expected format: [{"query_id": "...", "question": "..."}, ...]

    Also handles alternative key names just in case.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"queries.json not found at {path}\n"
            f"Download it and place in data/ folder."
        )

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Normalize key names
    # Some versions use "query_id", some use "id"
    # Some use "question", some use "query"
    normalized = []
    for item in data:
        query_id = (item.get("query_id") or
                    item.get("id") or
                    str(item.get("ID", "")))
        question = (item.get("question") or
                    item.get("query") or
                    item.get("text", ""))

        if query_id and question:
            normalized.append({
                "query_id": str(query_id),
                "question": question
            })

    print(f"✓ Loaded {len(normalized)} queries from {path}")
    return normalized


# ─────────────────────────────────────────────
# FUNCTION 2 — Load Progress
# ─────────────────────────────────────────────

def load_progress(path: str) -> dict:
    """
    Loads previously completed results from progress file.
    Returns dict of {query_id: result_row}
    """
    if not os.path.exists(path):
        return {}

    with open(path, 'r', encoding='utf-8') as f:
        progress = json.load(f)

    print(f"✓ Resuming from progress: {len(progress)} queries done")
    return progress


# ─────────────────────────────────────────────
# FUNCTION 3 — Save Progress
# ─────────────────────────────────────────────

def save_progress(progress: dict, path: str):
    """
    Saves completed results to progress file.
    Called after every single query.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


# ─────────────────────────────────────────────
# FUNCTION 4 — Load All Components
# ─────────────────────────────────────────────

def load_all_components():
    """
    Loads every component once before the main loop.
    Returns all components needed for processing.
    """
    print("\nLoading pipeline components...")
    print("(This takes ~30 seconds, happens only once)")

    components = {}

    # Chunks
    components['chunks'] = load_chunks(CHUNKS_PATH)
    print(f"  ✓ Chunks loaded: {len(components['chunks'])}")

    # ChromaDB
    components['collection'] = load_chroma_collection()
    print(f"  ✓ ChromaDB loaded: {components['collection'].count()} indexed")

    # Embedding model
    components['model'] = load_embed_model()
    print(f"  ✓ Embedding model loaded")

    # BM25
    components['bm25'] = load_or_build_bm25(components['chunks'])
    print(f"  ✓ BM25 index loaded")

    # NVIDIA client
    components['client'] = init_client()
    print(f"  ✓ NVIDIA client initialized")

    print(f"\n✓ All components ready")
    return components


# ─────────────────────────────────────────────
# FUNCTION 5 — Process One Query
# ─────────────────────────────────────────────

def process_query(query_id: str, question: str,
                  components: dict) -> dict:
    """
    Full pipeline for one query.
    Returns one row ready for CSV.
    """

    # Retrieve top 5 chunks
    retrieved = retrieve(
        query      = question,
        collection = components['collection'],
        model      = components['model'],
        bm25       = components['bm25'],
        chunks     = components['chunks'],
        top_k      = TOP_K
    )
    
    # -- Save Retrieved Context to .txt --
    context_dir = "outputs/retrieved_contexts"
    os.makedirs(context_dir, exist_ok=True)
    
    txt_path = os.path.join(context_dir, f"{query_id}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"QUERY ID: {query_id}\n")
        f.write(f"QUESTION: {question}\n")
        f.write("="*60 + "\n\n")
        
        for i, chunk in enumerate(retrieved):
            f.write(f"--- Chunk {i+1} ---\n")
            f.write(f"Section:       {chunk.get('section', 'Unknown')}\n")
            f.write(f"Pages:         {chunk.get('page_start', '?')}-{chunk.get('page_end', '?')}\n")
            f.write(f"Hybrid Score:  {chunk.get('rrf_score', 0):.4f}\n")
            f.write(f"Raw Text:\n{chunk.get('text', '')}\n\n")

    # Generate answer
    result = generate_answer(
        client           = components['client'],
        query_id         = query_id,
        question         = question,
        retrieved_chunks = retrieved
    )

    # Save verification snapshot (safe, non-blocking)
    try:
        from verification_1.context_snapshot import create_snapshot, save_snapshot
        snapshot_obj = create_snapshot(
            query_id         = query_id,
            question         = question,
            answer           = result["answer"],
            retrieved_chunks = retrieved
        )
        save_snapshot(snapshot_obj)
    except Exception:
        pass

    return result


# ─────────────────────────────────────────────
# FUNCTION 6 — Run Full Pipeline
# ─────────────────────────────────────────────

def run_pipeline(queries: list, components: dict) -> list:
    """
    Main loop. Processes all queries with progress saving.
    Skips already-completed queries on resume.
    """

    # Load existing progress
    progress = load_progress(PROGRESS_PATH)
    results  = list(progress.values())

    # Filter out already-done queries
    remaining = [q for q in queries
                 if q['query_id'] not in progress]

    print(f"\n{'='*60}")
    print(f"PROCESSING QUERIES")
    print(f"{'='*60}")
    print(f"  Total queries    : {len(queries)}")
    print(f"  Already done     : {len(progress)}")
    print(f"  Remaining        : {len(remaining)}")
    print(f"  Estimated time   : ~{len(remaining) * API_DELAY / 60:.1f} minutes")
    print(f"{'='*60}\n")

    if not remaining:
        print("✓ All queries already processed. Generating CSV...")
        return results

    for i, query in enumerate(tqdm(remaining, desc="Processing queries")):
        query_id = query['query_id']
        question = query['question']

        try:
            # Process this query
            result = process_query(query_id, question, components)
            results.append(result)

            # Save progress immediately
            progress[query_id] = result
            save_progress(progress, PROGRESS_PATH)

            # Rate limiting delay
            time.sleep(API_DELAY)

        except Exception as e:
            print(f"\n  Error on query {query_id}: {e}")
            print(f"  Saving fallback row and continuing...")

            # Save fallback row so we don't lose this query_id
            fallback = {
                "ID":         query_id,
                "context":    "Error during retrieval",
                "answer":     "Not found in the provided textbook.",
                "references": json.dumps({"sections": [], "pages": []})
            }
            results.append(fallback)
            progress[query_id] = fallback
            save_progress(progress, PROGRESS_PATH)

            time.sleep(API_DELAY)

    return results


# ─────────────────────────────────────────────
# FUNCTION 7 — Save Submission CSV
# ─────────────────────────────────────────────

def save_submission(results: list, path: str):
    """
    Saves final submission.csv with required columns.
    Column order matches problem statement exactly.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    df = pd.DataFrame(results, columns=["ID", "context",
                                         "answer", "references"])

    # Verify all required columns exist
    required = ["ID", "context", "answer", "references"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df.to_csv(path, index=False, encoding='utf-8')

    print(f"\n✓ submission.csv saved: {path}")
    print(f"  Rows    : {len(df)}")
    print(f"  Columns : {list(df.columns)}")

    # Verify references column is valid JSON
    invalid_refs = 0
    for ref in df['references']:
        try:
            parsed = json.loads(ref)
            if 'sections' not in parsed or 'pages' not in parsed:
                invalid_refs += 1
        except:
            invalid_refs += 1

    if invalid_refs == 0:
        print(f"  ✓ All references are valid JSON")
    else:
        print(f"  ⚠ {invalid_refs} rows have invalid references")

    return df


# ─────────────────────────────────────────────
# FUNCTION 8 — Citation Audit
# ─────────────────────────────────────────────

def citation_audit(df: pd.DataFrame, num_samples: int = 5):
    """
    Prints random sample rows for manual verification.

    For each sample:
    - Open PDF to cited pages
    - Verify context text appears there
    - Verify answer is answerable from context

    This is your quality gate before submission.
    """
    print(f"\n{'='*60}")
    print(f"CITATION AUDIT — Verify these manually against PDF")
    print(f"{'='*60}")
    print(f"Instructions:")
    print(f"  1. Open book.pdf")
    print(f"  2. Go to cited page numbers")
    print(f"  3. Verify the answer matches the content there")
    print(f"  4. If wrong, check your chunking or retrieval\n")

    samples = df.sample(min(num_samples, len(df)))

    for i, (_, row) in enumerate(samples.iterrows()):
        refs = json.loads(row['references'])
        print(f"Sample {i+1}:")
        print(f"  ID        : {row['ID']}")
        print(f"  Answer    : {row['answer'][:200]}")
        print(f"  Sections  : {refs.get('sections', [])}")
        print(f"  Pages     : {refs.get('pages', [])}")
        print(f"  Context   : {row['context'][:150]}...")
        print()


# ─────────────────────────────────────────────
# FUNCTION 9 — Print Final Stats
# ─────────────────────────────────────────────

def print_final_stats(df: pd.DataFrame):
    """
    Prints summary statistics about the submission.
    """
    print(f"\n{'='*60}")
    print(f"SUBMISSION STATISTICS")
    print(f"{'='*60}")

    total       = len(df)
    not_found   = df['answer'].str.contains(
                    "Not found", case=False, na=False).sum()
    answered    = total - not_found
    avg_ans_len = df['answer'].str.len().mean()

    print(f"  Total queries     : {total}")
    print(f"  Answered          : {answered} "
          f"({100*answered/total:.1f}%)")
    print(f"  Not found         : {not_found} "
          f"({100*not_found/total:.1f}%)")
    print(f"  Avg answer length : {avg_ans_len:.0f} chars")

    # Check reference coverage
    total_pages = 0
    for ref_str in df['references']:
        try:
            ref = json.loads(ref_str)
            total_pages += len(ref.get('pages', []))
        except:
            pass

    avg_pages = total_pages / total if total > 0 else 0
    print(f"  Avg pages cited   : {avg_pages:.1f} per query")

    if not_found / total > 0.3:
        print(f"\n  ⚠ High not-found rate ({100*not_found/total:.0f}%)")
        print(f"    Consider checking retrieval quality")
    else:
        print(f"\n  ✓ Not-found rate is acceptable")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("RAG PIPELINE — FULL RUN")
    print("=" * 60)

    # Load queries
    queries = load_queries(QUERIES_PATH)

    # Load all components once
    components = load_all_components()

    # Run pipeline
    results = run_pipeline(queries, components)

    # Save CSV
    df = save_submission(results, OUTPUT_PATH)

    # Print stats
    print_final_stats(df)

    # Citation audit
    citation_audit(df)

    print(f"\n{'='*60}")
    print(f"✓ PIPELINE COMPLETE")
    print(f"  Submit: outputs/submission.csv")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()