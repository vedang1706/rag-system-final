# components/sidebar.py

import streamlit as st

SAMPLE_QUESTIONS = [
    "What is classical conditioning?",
    "Explain Maslow's hierarchy of needs",
    "What are the stages of sleep?",
    "What are the basic parts of a neuron?",
    "What is the difference between classical and operant conditioning?",
    "What are Freud's stages of psychosexual development?",
    "How does short-term memory work?",
    "What is the Big Five personality model?",
    "What is cognitive dissonance?",
    "What are the symptoms of major depressive disorder?",
]


def render_sidebar(components: dict, active_mode: str = "chat"):
    """
    Renders the sidebar with:
    - System status
    - Index statistics
    - Chapter list
    - Sample questions (only in chat mode)
    - Tech stack info
    """
    with st.sidebar:
        st.markdown(
            """
            <style>
            [data-testid="stSidebarNav"] {
                display: none !important;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        # ── System Status ──
        st.markdown("### System Status")
        st.success("All components loaded")

        st.divider()

        # ── Statistics ──
        chunks      = components["chunks"]
        main_chunks = len([c for c in chunks
                           if not c['is_supplementary']])
        chapters    = sorted(list(set(
            c['chapter'] for c in chunks
            if not c['is_supplementary']
        )))

        st.markdown("### Index Statistics")
        c1, c2 = st.columns(2)
        c1.metric("Total",    len(chunks))
        c2.metric("Main",     main_chunks)
        c1.metric("Chapters", len(chapters))
        c2.metric("Indexed",  components["collection"].count())

        st.divider()

        # ── Chapters ──
        with st.expander("Chapters Covered", expanded=False):
            for ch in chapters:
                short = ch.replace("Chapter ", "Ch.")
                st.markdown(f"- {short}")

        st.divider()

        # ── Sample Questions (chat mode only) ──
        if active_mode == "chat":
            st.markdown("### Sample Questions")
            st.caption("Click any question to use it")

            for sample in SAMPLE_QUESTIONS:
                if st.button(
                    sample,
                    key              = f"sidebar_btn_{hash(sample)}",
                    use_container_width = True
                ):
                    st.session_state.sidebar_question   = sample
                    st.session_state.trigger_from_sidebar = True
                    st.rerun()

            st.divider()

        # ── UI Theme ──
        st.markdown("### UI Settings")
        st.session_state.ui_theme = st.selectbox(
            "Color Theme",
            options=["Standard (System)", "Light Mode", "Dark Mode"],
            index=0
        )
        st.divider()

        # ── Tech Stack ──
        with st.expander("Tech Stack", expanded=False):
            st.markdown("""
| Component | Details |
|---|---|
| Embeddings | all-MiniLM-L6-v2 |
| Vector DB | ChromaDB 0.5.3 |
| Retrieval | Hybrid RRF |
| LLM | openai/gpt-oss-20b |
| Framework | Streamlit |
            """)

        # ── Footer ──
        st.markdown(
            "<p style='color:gray; font-size:11px; text-align:center;'>"
            "WCE ACM Hackathon 2026<br>"
            "OpenStax Psychology 2e"
            "</p>",
            unsafe_allow_html=True
        )