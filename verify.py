import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

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
    import torch
    print(f"✓ torch version: {torch.__version__}")
except:
    print("✗ torch FAILED")

try:
    import transformers
    print(f"✓ transformers version: {transformers.__version__}")
except:
    print("✗ transformers FAILED")

try:
    import sys, os
    sys.path.append("src")
    from verifier import load_nli_verifier
    v = load_nli_verifier()
    if v.get("available"):
        print(f"✓ ModernBERT NLI Verifier OK ({v.get('type')})")
    else:
        print(f"⚠ Verifier in fallback mode: {v.get('error')}")
except Exception as e:
    print(f"✗ Verifier FAILED: {e}")

print("\nAll checks done.")
print("If all show ✓ you are ready to build.")