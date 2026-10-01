# components/chat_history.py

import streamlit as st
import json
from utils.citation import find_exact_citations, format_citations


def render_chat_message(role: str, content: dict, msg_index: int = 0):
    """
    Renders a single chat message.

    role    : "user" or "assistant"
    content : dict with message data
    """
    if role == "user":
        with st.chat_message("user"):
            st.markdown(content["text"])

    elif role == "assistant":
        with st.chat_message("assistant"):

            answer = content.get("answer", "")
            elapsed = content.get("elapsed", 0)

            # Answer
            if "Not found" in answer:
                st.warning(answer)
            else:
                st.markdown(answer)

            st.caption(f"Response time: {elapsed:.2f}s")

            # Verification Block
            verification = content.get("verification")
            if verification:
                from components.verification import render_verification_block
                render_verification_block(verification, msg_index=msg_index)

            # Exact Citations
            citations  = content.get("citations", [])
            formatted  = format_citations(citations)

            if formatted:
                st.markdown("**Source Citations**")
                for cite in formatted:
                    _render_citation(cite)
            else:
                refs = content.get("references", {})
                if refs.get("pages"):
                    st.caption(
                        f"Pages referenced: "
                        f"{', '.join(str(p) for p in refs['pages'])}"
                    )

            # Retrieved chunks expander
            retrieved = content.get("retrieved", [])
            if retrieved:
                with st.expander("Retrieved Context", expanded=False):
                    _render_chunks(retrieved, msg_index)


def _render_citation(cite: dict):
    """
    Renders one exact citation block.
    Clean, no emojis, information-dense.
    """
    confidence_color = {
        "High":   "green",
        "Medium": "orange",
        "Low":    "gray"
    }.get(cite['confidence'], "gray")

    st.markdown(
        f"<div style='"
        f"border-left: 3px solid {confidence_color};"
        f"padding: 8px 12px;"
        f"margin: 4px 0;"
        f"background: rgba(255,255,255,0.03);"
        f"border-radius: 0 4px 4px 0;"
        f"'>"
        f"<strong>{cite['section']}</strong>"
        f"<br>"
        f"<span style='color:gray; font-size:12px;'>"
        f"{cite['chapter']} &nbsp;|&nbsp; "
        f"Pages {cite['page_start']}–{cite['page_end']} &nbsp;|&nbsp; "
        f"Confidence: {cite['confidence']}"
        f"</span>"
        f"<br>"
        f"<span style='font-size:12px; font-style:italic;'>"
        f"\"{cite['evidence']}\""
        f"</span>"
        f"</div>",
        unsafe_allow_html=True
    )


def _render_chunks(retrieved: list, msg_index: int = 0):
    """
    Renders all retrieved chunks in expander.
    """
    # ── Compile Text for Download ──
    full_text = "RETRIEVED CONTEXT ALIVE\n=================================\n\n"
    for i, chunk in enumerate(retrieved):
        full_text += f"--- Chunk {i+1} ---\n"
        full_text += f"Section:       {chunk.get('section', 'Unknown')}\n"
        full_text += f"Pages:         {chunk.get('page_start', '?')}-{chunk.get('page_end', '?')}\n"
        full_text += f"Hybrid Score:  {chunk.get('rrf_score', 0):.4f}\n"
        full_text += f"Raw Text:\n{chunk.get('text', '')}\n\n"

    # ── Download Button ──
    st.download_button(
        label="📥 Download all retrieved context (.txt)",
        data=full_text,
        file_name=f"retrieved_context_msg_{msg_index}.txt",
        mime="text/plain",
        key=f"dl_txt_ctx_{msg_index}",
        use_container_width=True
    )
    
    st.markdown("---")

    source_label = {
        "both":   "[vector + bm25]",
        "vector": "[vector]",
        "bm25":   "[bm25]"
    }

    for i, chunk in enumerate(retrieved):
        label = source_label.get(chunk.get("source", ""), "")
        st.markdown(
            f"**Chunk {i+1}:** {chunk['section']} | "
            f"p.{chunk['page_start']}–{chunk['page_end']} | "
            f"RRF: {chunk.get('rrf_score', 0):.4f} {label}"
        )
        preview = chunk['text'][:400]
        if len(chunk['text']) > 400:
            preview += "..."
        st.text_area(
            label            = f"chunk_{i}",
            value            = preview,
            height           = 80,
            disabled         = True,
            label_visibility = "collapsed",
            key              = f"chunk_{msg_index}_{i}"
        )


def render_chat_history(history: list):
    """
    Renders full chat history from session state.
    Called on every rerun to show all messages.
    """
    for idx, message in enumerate(history):
        render_chat_message(message["role"], message["content"], msg_index=idx)