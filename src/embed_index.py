# src/embed_index.py

import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import json
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ── SUPPRESS CHROMA TELEMETRY SPAM ──
try:
    from chromadb.telemetry.posthog import Posthog
    Posthog.capture = lambda *args, **kwargs: None
except:
    pass
# ────────────────────────────────────

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

CHUNKS_PATH      = "cache/chunks.json"
EMBEDDINGS_CACHE = "cache/embeddings.npy"
CHROMA_PATH      = "chroma_db"
COLLECTION_NAME  = "psychology_textbook"
EMBED_MODEL      = "all-MiniLM-L6-v2"
BATCH_SIZE       = 64  # embed 64 chunks at a time


# ─────────────────────────────────────────────
# FUNCTION 1 — Load Chunks
# ─────────────────────────────────────────────

def load_chunks(path: str) -> list:
    """
    Loads chunks from cache/chunks.json
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"chunks.json not found at {path}\n"
            f"Run src/ingest.py first."
        )

    with open(path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)

    print(f"✓ Loaded {len(chunks)} chunks from {path}")
    return chunks


# ─────────────────────────────────────────────
# FUNCTION 2 — Load or Generate Embeddings
# ─────────────────────────────────────────────

def load_or_generate_embeddings(chunks: list,
                                 cache_path: str) -> np.ndarray:
    """
    Loads embeddings from cache if available.
    Otherwise generates them and saves to cache.

    Returns numpy array of shape (num_chunks, 384)
    """

    # ── Try loading from cache first ──
    if os.path.exists(cache_path):
        embeddings = np.load(cache_path)

        # Verify cache matches current chunks
        if embeddings.shape[0] == len(chunks):
            print(f"✓ Embeddings loaded from cache: {cache_path}")
            print(f"  Shape: {embeddings.shape}")
            return embeddings
        else:
            print(f"  Cache mismatch: {embeddings.shape[0]} cached "
                  f"vs {len(chunks)} chunks. Regenerating...")

    # ── Generate embeddings ──
    print(f"\nLoading embedding model: {EMBED_MODEL}")
    print("  This downloads ~90MB on first run, then it's cached locally.")
    model = SentenceTransformer(EMBED_MODEL)
    print(f"✓ Model loaded")

    texts = [chunk['text'] for chunk in chunks]

    print(f"\nGenerating embeddings for {len(texts)} chunks...")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  This takes 3-8 minutes on CPU. Do not interrupt.")

    all_embeddings = []

    for i in tqdm(range(0, len(texts), BATCH_SIZE),
                  desc="Embedding batches"):
        batch = texts[i : i + BATCH_SIZE]
        batch_embeddings = model.encode(
            batch,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        all_embeddings.append(batch_embeddings)

    embeddings = np.vstack(all_embeddings)

    # Save to cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, embeddings)

    print(f"✓ Embeddings generated and cached: {cache_path}")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Each chunk → {embeddings.shape[1]}-dimensional vector")

    return embeddings


# ─────────────────────────────────────────────
# FUNCTION 3 — Initialize ChromaDB
# ─────────────────────────────────────────────

def init_chroma(path: str):
    """
    Initializes persistent ChromaDB client.
    Data survives restarts — stored on disk at chroma_db/
    """
    client = chromadb.PersistentClient(
        path=path,
        settings=chromadb.config.Settings(anonymized_telemetry=False)
    )
    print(f"✓ ChromaDB initialized at: {path}")
    return client


# ─────────────────────────────────────────────
# FUNCTION 4 — Build Index
# ─────────────────────────────────────────────

def build_index(client, chunks: list,
                embeddings: np.ndarray) -> object:
    """
    Stores all chunks with embeddings and metadata in ChromaDB.

    Skips if collection already has the correct number of items.
    Delete chroma_db/ folder to force a full rebuild.
    """

    # Get or create collection
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}  # cosine similarity
    )

    existing_count = collection.count()

    # ── Already indexed ──
    if existing_count == len(chunks):
        print(f"✓ Index already built: {existing_count} chunks in ChromaDB")
        print(f"  Delete chroma_db/ folder to force rebuild")
        return collection

    # ── Partial or empty — rebuild ──
    if existing_count > 0:
        print(f"  Partial index found ({existing_count} items). Rebuilding...")
        client.delete_collection(COLLECTION_NAME)
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

    print(f"\nIndexing {len(chunks)} chunks into ChromaDB...")

    # Insert in batches
    # ChromaDB handles large inserts better in batches of 100
    batch_size = 100

    for i in tqdm(range(0, len(chunks), batch_size),
                  desc="Indexing batches"):

        batch_chunks     = chunks[i : i + batch_size]
        batch_embeddings = embeddings[i : i + batch_size]

        ids         = []
        documents   = []
        metadatas   = []
        embed_list  = []

        for j, chunk in enumerate(batch_chunks):
            ids.append(chunk['chunk_id'])
            documents.append(chunk['text'])
            metadatas.append({
                "section":          chunk['section'],
                "section_path":     chunk['section_path'],
                "chapter":          chunk['chapter'],
                "page_start":       chunk['page_start'],
                "page_end":         chunk['page_end'],
                "pages":            json.dumps(chunk['pages']),
                "is_supplementary": str(chunk['is_supplementary'])
            })
            embed_list.append(batch_embeddings[j].tolist())

        collection.add(
            ids        = ids,
            documents  = documents,
            metadatas  = metadatas,
            embeddings = embed_list
        )

    final_count = collection.count()
    print(f"✓ Index built: {final_count} chunks stored in ChromaDB")

    return collection


# ─────────────────────────────────────────────
# FUNCTION 5 — Verify With Test Query
# ─────────────────────────────────────────────

def verify_retrieval(collection, model_name: str = EMBED_MODEL):
    """
    Runs a test query to prove retrieval works end-to-end.
    This is your proof before running the full pipeline.
    """
    print(f"\n{'='*60}")
    print("RETRIEVAL VERIFICATION TEST")
    print(f"{'='*60}")

    test_queries = [
        "What is classical conditioning?",
        "How does memory work in the brain?",
        "What are Freud's stages of development?"
    ]

    # Load model for query embedding
    print(f"Loading model for query embedding...")
    model = SentenceTransformer(model_name)

    for query in test_queries:
        print(f"\nQuery: '{query}'")

        # Embed query
        query_vector = model.encode(query).tolist()

        # Search ChromaDB
        results = collection.query(
            query_embeddings = [query_vector],
            n_results        = 3,
            include          = ["documents", "metadatas", "distances"]
        )

        print(f"Top 3 results:")
        for i in range(len(results['ids'][0])):
            chunk_id = results['ids'][0][i]
            metadata = results['metadatas'][0][i]
            distance = results['distances'][0][i]
            text_preview = results['documents'][0][i][:100]

            print(f"\n  Result {i+1}:")
            print(f"    section  : {metadata['section']}")
            print(f"    chapter  : {metadata['chapter']}")
            print(f"    pages    : {metadata['page_start']}"
                  f" → {metadata['page_end']}")
            print(f"    similarity: {1 - distance:.3f}")
            print(f"    preview  : {text_preview}...")


# ─────────────────────────────────────────────
# FUNCTION 6 — Print Index Stats
# ─────────────────────────────────────────────

def print_stats(collection, chunks: list):
    """
    Prints summary statistics about the built index.
    """
    print(f"\n{'='*60}")
    print("INDEX STATISTICS")
    print(f"{'='*60}")

    total    = collection.count()
    main     = len([c for c in chunks if not c['is_supplementary']])
    supp     = len([c for c in chunks if c['is_supplementary']])

    chapters = list(set(c['chapter'] for c in chunks
                       if not c['is_supplementary']))

    print(f"  Total chunks indexed : {total}")
    print(f"  Main content chunks  : {main}")
    print(f"  Supplementary chunks : {supp}")
    print(f"  Chapters covered     : {len(chapters)}")
    print(f"  Embedding dimensions : 384")
    print(f"  Similarity metric    : cosine")
    print(f"\n  Chapters in index:")
    for ch in sorted(chapters):
        print(f"    - {ch}")


# ─────────────────────────────────────────────
# LOADER FUNCTIONS — used by run_pipeline.py
# ─────────────────────────────────────────────

def load_chunks(path: str = CHUNKS_PATH) -> list:
    """
    Loads chunks from cache/chunks.json
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"chunks.json not found at {path}\n"
            f"Run src/ingest.py first."
        )
    with open(path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    print(f"✓ Loaded {len(chunks)} chunks from {path}")
    return chunks


def load_chroma_collection():
    """
    Loads existing ChromaDB collection.
    """
    client = chromadb.PersistentClient(
        path=CHROMA_PATH,
        settings=chromadb.config.Settings(anonymized_telemetry=False)
    )
    collection = client.get_collection(name=COLLECTION_NAME)
    print(f"✓ ChromaDB collection loaded: {collection.count()} chunks")
    return collection


def load_embed_model():
    """
    Loads the sentence transformer model.
    """
    model = SentenceTransformer(EMBED_MODEL)
    print(f"✓ Embedding model loaded: {EMBED_MODEL}")
    return model


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("STEP 2: EMBEDDING AND INDEXING")
    print("=" * 60)

    # Load chunks
    chunks = load_chunks(CHUNKS_PATH)

    # ── Save Full Extracted Text ──
    print("\nSaving raw extracted text to outputs/full_extracted_textbook.txt...")
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/full_extracted_textbook.txt", "w", encoding="utf-8") as f:
        f.write("FULL EXTRACTED TEXTBOOK (FROM CHUNKS)\n")
        f.write("======================================\n\n")
        for i, chunk in enumerate(chunks):
            f.write(f"--- Chunk {i+1} ---\n")
            f.write(f"Chapter: {chunk.get('chapter', 'Unknown')}\n")
            f.write(f"Section: {chunk.get('section', 'Unknown')}\n")
            f.write(f"Pages:   {chunk.get('page_start', '?')}-{chunk.get('page_end', '?')}\n\n")
            f.write(f"{chunk.get('text', '')}\n\n")
            f.write("*"*60 + "\n\n")
    print("✓ Raw textbook text saved successfully.")

    # Load or generate embeddings
    embeddings = load_or_generate_embeddings(chunks, EMBEDDINGS_CACHE)

    # Initialize ChromaDB
    client = init_chroma(CHROMA_PATH)

    # Build index
    collection = build_index(client, chunks, embeddings)

    # Print stats
    print_stats(collection, chunks)

    # Verify with test queries
    verify_retrieval(collection)

    print(f"\n{'='*60}")
    print("✓ EMBEDDING AND INDEXING COMPLETE")
    print("  Run next: python src/retrieve.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()