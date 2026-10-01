# components/results.py

import streamlit as st
import json
from utils.citation import format_citations


def render_answer_block(result: dict,
                        citations: list,
                        elapsed: float,
                        verification: dict = None):
    """
    Renders answer with exact citations and optional verification details.
    Used in batch mode result preview.
    """
    answer    = result.get("answer", "")
    formatted = format_citations(citations)

    # Answer
    st.markdown("#### Answer")
    if "Not found" in answer:
        st.warning(answer)
    else:
        st.success(answer)

    st.caption(f"Response time: {elapsed:.2f}s")

    # Verification block if available
    if verification:
        from components.verification import render_verification_block
        render_verification_block(verification)

    # Exact citations
    if formatted:
        st.markdown("#### Source Citations")
        for cite in formatted:
            _render_citation_card(cite)
    else:
        refs = json.loads(result.get("references", "{}"))
        pages = refs.get("pages", [])
        if pages:
            st.caption(
                f"Pages: {', '.join(str(p) for p in pages)}"
            )

    # Reference JSON
    with st.expander("Reference JSON", expanded=False):
        st.code(result.get("references", "{}"), language="json")


def _render_citation_card(cite: dict):
    """
    Renders one citation as a clean info card.
    """
    confidence_color = {
        "High":   "#2e7d32",
        "Medium": "#e65100",
        "Low":    "#546e7a"
    }.get(cite['confidence'], "#546e7a")

    st.markdown(
        f"<div style='"
        f"border: 1px solid {confidence_color}33;"
        f"border-left: 4px solid {confidence_color};"
        f"padding: 10px 14px;"
        f"margin: 6px 0;"
        f"border-radius: 0 6px 6px 0;"
        f"background: {confidence_color}08;"
        f"'>"
        f"<div style='font-weight:600; margin-bottom:4px;'>"
        f"Citation {cite['index']}: {cite['section']}"
        f"</div>"
        f"<div style='color:gray; font-size:12px; margin-bottom:6px;'>"
        f"{cite['chapter']}"
        f" &nbsp;|&nbsp; Pages {cite['page_start']}–{cite['page_end']}"
        f" &nbsp;|&nbsp; Confidence: "
        f"<span style='color:{confidence_color};'>"
        f"{cite['confidence']}</span>"
        f"</div>"
        f"<div style='"
        f"font-size:12px;"
        f"font-style:italic;"
        f"color:#ccc;"
        f"border-top:1px solid {confidence_color}22;"
        f"padding-top:6px;"
        f"margin-top:4px;"
        f"'>"
        f"Evidence: \"{cite['evidence']}\""
        f"</div>"
        f"</div>",
        unsafe_allow_html=True
    )


def render_batch_results_table(df):
    """
    Renders batch results as a preview table.
    Shows ID, truncated answer, page count.
    """
    import pandas as pd

    preview_df = pd.DataFrame({
        "ID":      df["ID"],
        "Answer":  df["answer"].str[:100] + "...",
        "Pages":   df["references"].apply(
            lambda r: len(json.loads(r).get("pages", []))
        )
    })

    st.dataframe(preview_df, use_container_width=True)