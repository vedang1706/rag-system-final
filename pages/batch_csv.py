# pages/batch_csv.py

import streamlit as st
import pandas as pd
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from components.sidebar import render_sidebar
from components.results import render_batch_results_table
from utils.batch import load_csv_queries, process_batch, df_to_csv_bytes


# ─────────────────────────────────────────────
# INIT SESSION STATE
# ─────────────────────────────────────────────

def init_batch_csv_session():
    defaults = {
        "batch_csv_queries": [],
        "batch_csv_results": None,
        "batch_csv_done":    False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ─────────────────────────────────────────────
# MAIN RENDER
# ─────────────────────────────────────────────

def render_batch_csv(components: dict):
    """
    Renders the batch CSV processing mode.
    Upload queries.csv → process → download results.csv
    """
    init_batch_csv_session()

    render_sidebar(components, active_mode="batch_csv")

    # ── Header ──
    st.markdown(
        "<h2 style='margin-bottom:0;'>Batch Processing — CSV Input</h2>"
        "<p style='color:gray; margin-top:4px;'>"
        "Upload a queries CSV file and download results as CSV"
        "</p>",
        unsafe_allow_html=True
    )
    st.divider()

    # ── Expected Format ──
    with st.expander("Expected CSV format", expanded=False):
        st.markdown("Your CSV file should have these columns:")
        st.code("""
query_id,question
1,What is classical conditioning?
2,Explain Maslow's hierarchy of needs
        """)
        st.markdown(
            "Accepted column names: `query_id` or `id` for ID, "
            "`question` or `query` or `text` for the question."
        )

    # ── File Upload ──
    uploaded = st.file_uploader(
        "Upload queries CSV file",
        type  = ["csv"],
        help  = "Upload any .csv file with query_id and question columns",
        key   = "csv_uploader"
    )

    if uploaded:
        queries = load_csv_queries(uploaded)

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
            st.session_state.batch_csv_results = df
            st.session_state.batch_csv_done    = True

    # ── Results ──
    if st.session_state.batch_csv_done and \
       st.session_state.batch_csv_results is not None:

        df = st.session_state.batch_csv_results

        st.divider()
        st.markdown("#### Results")

        total    = len(df)
        answered = total - df['answer'].str.contains(
                       "Not found", case=False, na=False
                   ).sum()

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Queries", total)
        c2.metric("Answered",      answered)
        c3.metric("Answer Rate",   f"{100*answered//total}%")

        st.markdown("#### Preview")
        render_batch_results_table(df)

        csv_bytes = df_to_csv_bytes(df)

        st.download_button(
            label               = "Download results.csv",
            data                = csv_bytes,
            file_name           = "results.csv",
            mime                = "text/csv",
            use_container_width = True
        )

        if st.button("Process another file", use_container_width=True):
            st.session_state.batch_csv_results = None
            st.session_state.batch_csv_done    = False
            st.rerun()