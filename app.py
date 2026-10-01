# app.py

import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import streamlit as st
import sys
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from embed_index import load_chunks, load_chroma_collection, load_embed_model
from retrieve   import load_or_build_bm25
from generate   import init_client
from verifier   import load_nli_verifier

from pages.chat       import render_chat
from pages.batch_json import render_batch_json
from pages.batch_csv  import render_batch_csv

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title = "Psychology RAG System",
    page_icon  = "book",
    layout     = "wide"
)

# ─────────────────────────────────────────────
# LOAD COMPONENTS ONCE
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_all():
    chunks     = load_chunks("cache/chunks.json")
    collection = load_chroma_collection()
    model      = load_embed_model()
    bm25       = load_or_build_bm25(chunks)
    client     = init_client()
    
    # Load Fast Lightweight Grounding Verifier
    try:
        verifier = load_nli_verifier()
    except Exception as e:
        print(f"Warning: could not load verifier: {e}")
        verifier = {"available": False, "error": str(e)}

    return {
        "chunks":     chunks,
        "collection": collection,
        "model":      model,
        "bm25":       bm25,
        "client":     client,
        "verifier":   verifier
    }

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():

    # Load components
    with st.spinner("Loading system..."):
        try:
            components = load_all()
        except Exception as e:
            st.error(f"Failed to load system: {e}")
            st.info("Make sure ingest.py and embed_index.py have been run.")
            st.stop()

    # ── Mode Selector ──
    st.markdown(
        "<h1 style='margin-bottom:0;'>"
        "Psychology Textbook RAG System"
        "</h1>"
        "<p style='color:gray; margin-top:2px;'>"
        "OpenStax Psychology 2e — Grounded answers with verifiable citations"
        "</p>",
        unsafe_allow_html=True
    )

    mode = st.radio(
    label     = "Mode",
    options   = ["Chat", "Batch JSON", "Batch CSV", "Evaluation Dashboard"],
    horizontal = True,
    label_visibility = "collapsed"
    )

    st.divider()

    # ── Route to Mode ──
    if mode == "Chat":
        render_chat(components)

    elif mode == "Batch JSON":
        render_batch_json(components)

    elif mode == "Batch CSV":
        render_batch_csv(components)
        
    elif mode == "Evaluation Dashboard":
        from pages.evaluation import render_evaluation
        render_evaluation()


if __name__ == "__main__":
    main()