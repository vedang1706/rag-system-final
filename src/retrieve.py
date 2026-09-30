# src/retrieve.py

import json
import os
import pickle
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from tqdm import tqdm

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

CHUNKS_PATH      = "cache/chunks.json"
EMBEDDINGS_CACHE = "cache/embeddings.npy"
BM25_CACHE       = "cache/bm25_index.pkl"
CHROMA_PATH      = "chroma_db"
COLLECTION_NAME  = "psychology_textbook"
EMBED_MODEL      = "all-MiniLM-L6-v2"

TOP_K_VECTOR     = 15   # fetch top 15 from vector search
TOP_K_BM25       = 15   # fetch top 15 from BM25
TOP_K_FINAL      = 7    # return top 7 after merging
RRF_K            = 60   # RRF constant (standard value)


# ─────────────────────────────────────────────
# FUNCTION 1 — Load Everything
# ─────────────────────────────────────────────

def load_chunks(path: str) -> list:
    if not os.path.exists(path):
        raise FileNotFoundError(f"chunks.json not found. Run ingest.py first.")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_chroma_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)
    return collection


def load_embed_model():
    return SentenceTransformer(EMBED_MODEL)


# ─────────────────────────────────────────────
# FUNCTION 2 — Build or Load BM25 Index
# ─────────────────────────────────────────────

def load_or_build_bm25(chunks: list) -> BM25Okapi:
    """
    Loads BM25 index from cache or builds from scratch.

    BM25 tokenization: lowercase, split on whitespace.
    Simple but effective for this use case.
    """

    if os.path.exists(BM25_CACHE):
        with open(BM25_CACHE, 'rb') as f:
            bm25 = pickle.load(f)
        print(f"✓ BM25 index loaded from cache: {BM25_CACHE}")
        return bm25

    print("Building BM25 index...")

    # Tokenize each chunk's text
    tokenized_corpus = []
    for chunk in tqdm(chunks, desc="Tokenizing"):
        tokens = chunk['text'].lower().split()
        tokenized_corpus.append(tokens)

    bm25 = BM25Okapi(tokenized_corpus)

    # Save to cache
    os.makedirs(os.path.dirname(BM25_CACHE), exist_ok=True)
    with open(BM25_CACHE, 'wb') as f:
        pickle.dump(bm25, f)

    print(f"✓ BM25 index built and cached: {BM25_CACHE}")
    return bm25


# ─────────────────────────────────────────────
# FUNCTION 3 — Vector Search
# ─────────────────────────────────────────────

def vector_search(query: str, collection,
                  model, top_k: int = TOP_K_VECTOR) -> list:
    """
    Embeds the query and searches ChromaDB.
    Returns list of (chunk_id, metadata, text, score) tuples.
    """
    query_vector = model.encode(query).tolist()

    results = collection.query(
        query_embeddings = [query_vector],
        n_results        = top_k,
        include          = ["documents", "metadatas", "distances"]
    )

    output = []
    for i in range(len(results['ids'][0])):
        output.append({
            "chunk_id": results['ids'][0][i],
            "text":     results['documents'][0][i],
            "metadata": results['metadatas'][0][i],
            "score":    1 - results['distances'][0][i]  # cosine similarity
        })

    return output


# ─────────────────────────────────────────────
# FUNCTION 4 — BM25 Search
# ─────────────────────────────────────────────

def bm25_search(query: str, bm25: BM25Okapi,
                chunks: list, top_k: int = TOP_K_BM25) -> list:
    """
    Tokenizes query and searches BM25 index.
    Returns list of (chunk_id, metadata, text, score) tuples.
    """
    query_tokens = query.lower().split()
    scores       = bm25.get_scores(query_tokens)

    # Get top_k indices
    top_indices = np.argsort(scores)[::-1][:top_k]

    output = []
    for idx in top_indices:
        if scores[idx] > 0:  # skip zero-score results
            chunk = chunks[idx]
            output.append({
                "chunk_id": chunk['chunk_id'],
                "text":     chunk['text'],
                "metadata": {
                    "section":      chunk['section'],
                    "section_path": chunk['section_path'],
                    "chapter":      chunk['chapter'],
                    "page_start":   chunk['page_start'],
                    "page_end":     chunk['page_end'],
                    "pages":        json.dumps(chunk['pages'])
                },
                "score": float(scores[idx])
            })

    return output


# ─────────────────────────────────────────────
# FUNCTION 5 — Reciprocal Rank Fusion
# ─────────────────────────────────────────────

def reciprocal_rank_fusion(vector_results: list,
                            bm25_results: list,
                            top_k: int = TOP_K_FINAL) -> list:
    """
    Merges vector and BM25 results using Reciprocal Rank Fusion.

    RRF score = 1/(rank + K) summed across both result lists.
    Higher RRF score = more relevant chunk.

    Chunks appearing in both lists get boosted.
    """
    rrf_scores = {}
    sources    = {}  # track which system found each chunk

    # Score from vector results
    for rank, result in enumerate(vector_results):
        cid = result['chunk_id']
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 1 / (rank + RRF_K)
        sources[cid]    = "vector"

    # Score from BM25 results
    for rank, result in enumerate(bm25_results):
        cid = result['chunk_id']
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 1 / (rank + RRF_K)
        if cid in sources:
            sources[cid] = "both"   # appeared in both
        else:
            sources[cid] = "bm25"

    # Build merged result list
    all_results = {}
    for result in vector_results + bm25_results:
        cid = result['chunk_id']
        if cid not in all_results:
            all_results[cid] = result

    # Sort by RRF score
    sorted_ids = sorted(rrf_scores.keys(),
                        key=lambda x: rrf_scores[x],
                        reverse=True)

    # Build final top_k list
    final = []
    for cid in sorted_ids[:top_k]:
        result = all_results[cid].copy()
        result['rrf_score'] = round(rrf_scores[cid], 6)
        result['source']    = sources[cid]
        final.append(result)

    return final


# ─────────────────────────────────────────────
# FUNCTION 6 — Main Retrieve Function
# ─────────────────────────────────────────────


def retrieve(query: str, collection, model,
             bm25: BM25Okapi, chunks: list,
             top_k: int = TOP_K_FINAL) -> list:
             
    """
    Main retrieval function.
    Given a query, returns top_k chunks using hybrid search.

    This is the function called by run_pipeline.py for every query.
    """

    # Vector search
    vector_results = vector_search(query, collection, model)

    # BM25 search
    bm25_results   = bm25_search(query, bm25, chunks)

    # Merge with RRF
    final_results  = reciprocal_rank_fusion(
        vector_results, bm25_results, top_k=top_k + 5
    )

    # Enrich metadata
    enriched = []
    for result in final_results:
        meta = result['metadata']
        enriched.append({
            "chunk_id":         result['chunk_id'],
            "text":             result['text'],
            "section":          meta['section'],
            "section_path":     meta['section_path'],
            "chapter":          meta['chapter'],
            "page_start":       int(meta['page_start']),
            "page_end":         int(meta['page_end']),
            "pages":            json.loads(meta['pages'])
                                if isinstance(meta['pages'], str)
                                else meta['pages'],
            "rrf_score":        result['rrf_score'],
            "source":           result['source'],
            "is_supplementary": meta.get('is_supplementary', 'False')
        })

    # ── Rerank: main content before supplementary ──
    # Also filter out reference/bibliography chunks
    main_chunks = [c for c in enriched
                if c['is_supplementary'] == 'False'
                and 'references' not in c['chapter'].lower()]

    supp_chunks = [c for c in enriched
                if c['is_supplementary'] == 'True'
                and 'references' not in c['chapter'].lower()]

    # If we have enough main chunks, use only those
    if len(main_chunks) >= top_k:
        return main_chunks[:top_k]

    # Otherwise fill remaining slots with supplementary
    combined = main_chunks + supp_chunks

    # ── Deduplicate by page overlap ──
    # If two chunks cover the same pages, keep only the higher-ranked one
   # ── Deduplicate by page overlap ──
    seen_pages = set()
    deduped    = []

    for chunk in enriched:
        chunk_pages = set(chunk['pages'])

        # Skip if this exact page already fully covered
        if chunk_pages and chunk_pages.issubset(seen_pages):
            continue

        overlap       = chunk_pages & seen_pages
        overlap_ratio = len(overlap) / len(chunk_pages) if chunk_pages else 0

        if overlap_ratio > 0.4:  # was 0.5, now stricter
            continue

        deduped.append(chunk)
        seen_pages.update(chunk_pages)

        if len(deduped) >= top_k:
            break

    return deduped



# ─────────────────────────────────────────────
# FUNCTION 7 — Format Retrieved Chunks
# ─────────────────────────────────────────────

def format_context(chunks: list) -> str:
    """
    Formats retrieved chunks into a clean context string
    for the LLM prompt.

    Each chunk is clearly labelled with its source.
    """
    parts = []

    for i, chunk in enumerate(chunks):
        part = (
            f"[Source {i+1}] "
            f"Section: {chunk['section']} | "
            f"Pages: {chunk['page_start']}-{chunk['page_end']}\n"
            f"{chunk['text']}"
        )
        parts.append(part)

    return "\n\n---\n\n".join(parts)


# ─────────────────────────────────────────────
# VERIFICATION — Test Retrieval
# ─────────────────────────────────────────────

def verify_retrieval(collection, model, bm25, chunks):
    """
    Tests hybrid retrieval on sample queries.
    Compares vector-only vs hybrid results.
    """
    test_cases = [
        {
            "query": "What is classical conditioning?",
            "expected_chapter": "Chapter 6 Learning"
        },
        {
            "query": "Explain Maslow's hierarchy of needs",
            "expected_chapter": "Chapter 10 Emotion and Motivation"
        },
        {
            "query": "What are the stages of sleep?",
            "expected_chapter": "Chapter 4 States of Consciousness"
        }
    ]

    print(f"\n{'='*60}")
    print("HYBRID RETRIEVAL VERIFICATION")
    print(f"{'='*60}")

    for tc in test_cases:
        query    = tc["query"]
        expected = tc["expected_chapter"]

        print(f"\nQuery: '{query}'")
        print(f"Expected chapter: {expected}")

        # Vector only
        v_results = vector_search(query, collection, model, top_k=5)
        v_chapters = [r['metadata']['chapter'] for r in v_results[:3]]

        # Hybrid
        h_results  = retrieve(query, collection, model, bm25, chunks)
        h_chapters = [r['chapter'] for r in h_results[:3]]

        print(f"\n  Vector-only top 3 chapters:")
        for i, ch in enumerate(v_chapters):
            print(f"    {i+1}. {ch}")

        print(f"\n  Hybrid top 3 chapters:")
        for i, r in enumerate(h_results[:3]):
            source_tag = f"[{r['source']}]"
            print(f"    {i+1}. {r['chapter']} "
                  f"| {r['section']} "
                  f"| p.{r['page_start']} "
                  f"| RRF:{r['rrf_score']:.4f} "
                  f"{source_tag}")

        # Check if expected chapter appears in top 3
        hit = any(expected in ch for ch in h_chapters)
        print(f"\n  Expected chapter in top 3: {'✓ YES' if hit else '✗ NO'}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("STEP 3: HYBRID RETRIEVAL TEST")
    print("=" * 60)

    # Load everything
    print("\nLoading components...")
    chunks     = load_chunks(CHUNKS_PATH)
    collection = load_chroma_collection()
    model      = load_embed_model()
    bm25       = load_or_build_bm25(chunks)

    print(f"✓ All components loaded")
    print(f"  Chunks     : {len(chunks)}")
    print(f"  ChromaDB   : {collection.count()} indexed")

    # Run verification
    verify_retrieval(collection, model, bm25, chunks)

    # Demo: show full formatted context for one query
    print(f"\n{'='*60}")
    print("SAMPLE CONTEXT PACK FOR LLM")
    print(f"{'='*60}")

    demo_query   = "What is the difference between classical and operant conditioning?"
    demo_results = retrieve(demo_query, collection, model, bm25, chunks)
    context_str  = format_context(demo_results)

    print(f"\nQuery: '{demo_query}'")
    print(f"\nContext pack ({len(demo_results)} chunks):")
    print(context_str[:1000] + "...\n")

    print(f"\n{'='*60}")
    print("✓ RETRIEVAL COMPLETE")
    print("  Run next: python src/generate.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()