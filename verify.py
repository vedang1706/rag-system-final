print("Checking all imports...")

try:
    import fitz
    print(f"✓ PyMuPDF version: {fitz.__version__}")
except:
    print("✗ PyMuPDF FAILED")

try:
    from sentence_transformers import SentenceTransformer
    print("✓ sentence-transformers OK")
except:
    print("✗ sentence-transformers FAILED")

try:
    import chromadb
    print(f"✓ ChromaDB version: {chromadb.__version__}")
except:
    print("✗ ChromaDB FAILED")

try:
    from rank_bm25 import BM25Okapi
    print("✓ rank-bm25 OK")
except:
    print("✗ rank-bm25 FAILED")

try:
    import openai
    print(f"✓ openai version: {openai.__version__}")
except:
    print("✗ openai FAILED")

try:
    import pandas
    print(f"✓ pandas version: {pandas.__version__}")
except:
    print("✗ pandas FAILED")

try:
    import numpy
    print(f"✓ numpy version: {numpy.__version__}")
except:
    print("✗ numpy FAILED")

print("\nAll checks done.")
print("If all show ✓ you are ready to build.")