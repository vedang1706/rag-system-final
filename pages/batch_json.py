# pages/batch_json.py

import streamlit as st
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from components.sidebar import render_sidebar
from components.results import render_batch_results_table
from utils.batch import load_json_queries, process_batch, df_to_csv_bytes


# ─────────────────────────────────────────────
# INIT SESSION STATE
# ─────────────────────────────────────────────

def init_batch_json_session():
    defaults = {
        "batch_json_queries": [],
        "batch_json_results": None,
        "batch_json_done":    False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ─────────────────────────────────────────────
# MAIN RENDER
# ─────────────────────────────────────────────

def render_batch_json(components: dict):
    """
    Renders the batch JSON processing mode.
    Upload queries.json → process → download results.csv
    """
    init_batch_json_session()

    render_sidebar(components, active_mode="batch_json")

    # ── Header ──
    st.markdown(
        "<h2 style='margin-bottom:0;'>Batch Processing — JSON Input</h2>"
        "<p style='color:gray; margin-top:4px;'>"
        "Upload a queries.json file and download results as CSV"
        "</p>",
        unsafe_allow_html=True
    )
    st.divider()

    # ── Expected Format Info ──
    with st.expander("Expected JSON format", expanded=False):
        st.markdown("Your JSON file should be a list of objects with these fields:")
        st.code("""
[
  {"query_id": "1", "question": "What is classical conditioning?"},
  {"query_id": "2", "question": "Explain Maslow's hierarchy of needs"},
  ...
]
        """, language="json")
        st.markdown(
            "Accepted key names: `query_id` or `id` for ID, "
            "`question` or `query` or `text` for the question."
        )

    # ── File Upload ──
    uploaded = st.file_uploader(
        "Upload queries JSON file",
        type            = ["json"],
        help            = "Upload any .json file with a list of questions",
        key             = "json_uploader"
    )

    if uploaded:
        # Load and preview queries
        queries = load_json_queries(uploaded)

        if not queries:
            st.error("No valid queries found in file.")
            return

        st.success(f"Loaded {len(queries)} queries from {uploaded.name}")

        # Preview
        st.markdown("#### Query Preview")
        preview_data = [
            {"ID": q["query_id"], "Question": q["question"]}
            for q in queries[:10]
        ]
        st.table(preview_data)

        if len(queries) > 10:
            st.caption(f"Showing first 10 of {len(queries)} queries")

        st.divider()

        # Estimate time
        est_minutes = round(len(queries) * 1.5 / 60, 1)
        st.info(
            f"Estimated processing time: ~{est_minutes} minutes "
            f"({len(queries)} queries x 1.5s each)"
        )

        # Process button
        if st.button(
            f"Process {len(queries)} queries",
            type                = "primary",
            use_container_width = True
        ):
            st.markdown("#### Processing")
            df = process_batch(queries, components)
            st.session_state.batch_json_results = df
            st.session_state.batch_json_done    = True

    # ── Results ──
    if st.session_state.batch_json_done and \
       st.session_state.batch_json_results is not None:

        df = st.session_state.batch_json_results

        st.divider()
        st.markdown("#### Results")

        # Stats
        total     = len(df)
        answered  = df['answer'].str.contains(
                        "Not found", case=False, na=False
                    ).sum()
        answered  = total - answered

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Queries",  total)
        c2.metric("Answered",       answered)
        c3.metric("Answer Rate",    f"{100*answered//total}%")

        # Preview table
        st.markdown("#### Preview")
        render_batch_results_table(df)

        # Download button
        csv_bytes = df_to_csv_bytes(df)

        st.download_button(
            label            = "Download results.csv",
            data             = csv_bytes,
            file_name        = "results.csv",
            mime             = "text/csv",
            use_container_width = True
        )

        # Reset button
        if st.button("Process another file", use_container_width=True):
            st.session_state.batch_json_results = None
            st.session_state.batch_json_done    = False
            st.rerun()