# components/verification.py

"""
Streamlit UI rendering components for ModernBERT NLI Hallucination Verification.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, List

STATUS_COLORS = {
    "VERIFIED": {
        "bg": "rgba(16, 185, 129, 0.12)",
        "border": "#10b981",
        "text": "#10b981",
        "badge_bg": "#10b981",
        "badge_text": "#ffffff",
        "icon": "✓"
    },
    "PARTIALLY VERIFIED": {
        "bg": "rgba(245, 158, 11, 0.12)",
        "border": "#f59e0b",
        "text": "#f59e0b",
        "badge_bg": "#f59e0b",
        "badge_text": "#ffffff",
        "icon": "◐"
    },
    "CONTRADICTED": {
        "bg": "rgba(239, 68, 68, 0.12)",
        "border": "#ef4444",
        "text": "#ef4444",
        "badge_bg": "#ef4444",
        "badge_text": "#ffffff",
        "icon": "✗"
    },
    "EVIDENCE CONFLICT": {
        "bg": "rgba(139, 92, 246, 0.12)",
        "border": "#8b5cf6",
        "text": "#8b5cf6",
        "badge_bg": "#8b5cf6",
        "badge_text": "#ffffff",
        "icon": "⚡"
    },
    "UNSUPPORTED": {
        "bg": "rgba(107, 114, 128, 0.12)",
        "border": "#6b7280",
        "text": "#6b7280",
        "badge_bg": "#6b7280",
        "badge_text": "#ffffff",
        "icon": "―"
    },
    "NOT APPLICABLE": {
        "bg": "rgba(107, 114, 128, 0.08)",
        "border": "#9ca3af",
        "text": "#9ca3af",
        "badge_bg": "#9ca3af",
        "badge_text": "#ffffff",
        "icon": "ℹ"
    },
    "UNAVAILABLE": {
        "bg": "rgba(107, 114, 128, 0.08)",
        "border": "#9ca3af",
        "text": "#9ca3af",
        "badge_bg": "#9ca3af",
        "badge_text": "#ffffff",
        "icon": "⚠"
    }
}


def render_verification_badge(status: str, confidence: float = None) -> str:
    """
    Returns HTML for a styled status badge.
    """
    style = STATUS_COLORS.get(status, STATUS_COLORS["UNSUPPORTED"])
    conf_str = f" ({confidence*100:.1f}%)" if confidence is not None else ""
    return (
        f"<span style='"
        f"display: inline-block;"
        f"background-color: {style['badge_bg']};"
        f"color: {style['badge_text']};"
        f"padding: 2px 9px;"
        f"border-radius: 9999px;"
        f"font-size: 11px;"
        f"font-weight: 700;"
        f"letter-spacing: 0.03em;"
        f"text-transform: uppercase;"
        f"margin-left: 6px;"
        f"vertical-align: middle;"
        f"'>"
        f"{style['icon']} {status}{conf_str}"
        f"</span>"
    )


def render_verification_block(verification: Dict[str, Any], msg_index: int = 0):
    """
    Renders complete claim-level verification results beneath an answer.
    """
    if not verification:
        return

    if not verification.get("is_available", True) or verification.get("overall_status") == "UNAVAILABLE":
        st.markdown(
            "<div style='color: gray; font-size: 13px; margin-top: 8px;'>"
            "<em>Verification unavailable</em>"
            "</div>",
            unsafe_allow_html=True
        )
        return

    overall_status = verification.get("overall_status", "UNSUPPORTED")
    if overall_status == "NOT APPLICABLE":
        return

    style = STATUS_COLORS.get(overall_status, STATUS_COLORS["UNSUPPORTED"])
    claims = verification.get("claims", [])
    total_claims = verification.get("total_claims", len(claims))
    verified_count = verification.get("verified_count", 0)
    contradicted_count = verification.get("contradicted_count", 0)
    unsupported_count = verification.get("unsupported_count", 0)
    conflict_count = verification.get("conflict_count", 0)
    elapsed = verification.get("elapsed_seconds", 0.0)

    # ── Header Card ──
    badge_html = render_verification_badge(overall_status)
    st.markdown(
        f"<div style='"
        f"border: 1px solid {style['border']}40;"
        f"border-left: 4px solid {style['border']};"
        f"background: {style['bg']};"
        f"padding: 12px 16px;"
        f"border-radius: 0 8px 8px 0;"
        f"margin: 12px 0 16px 0;"
        f"'>"
        f"<div style='display: flex; justify-content: space-between; align-items: center;'>"
        f"<div>"
        f"<span style='font-size: 14px; font-weight: 700; color: {style['text']};'>ModernBERT NLI Verification</span>"
        f"{badge_html}"
        f"</div>"
        f"<span style='font-size: 12px; color: gray;'>"
        f"Verified against 7 retrieved chunks in {elapsed:.2f}s"
        f"</span>"
        f"</div>"
        f"<div style='font-size: 12px; color: gray; margin-top: 4px;'>"
        f"Claims: {total_claims} &nbsp;|&nbsp; "
        f"<span style='color: #10b981;'>✓ Supported: {verified_count}</span> &nbsp;|&nbsp; "
        f"<span style='color: #6b7280;'>― Unsupported: {unsupported_count}</span>"
        + (f" &nbsp;|&nbsp; <span style='color: #ef4444;'>✗ Contradicted: {contradicted_count}</span>" if contradicted_count > 0 else "")
        + (f" &nbsp;|&nbsp; <span style='color: #8b5cf6;'>⚡ Conflict: {conflict_count}</span>" if conflict_count > 0 else "")
        + f"</div>"
        f"</div>",
        unsafe_allow_html=True
    )

    if not claims:
        return

    # ── Claim by Claim Breakdown ──
    st.markdown("##### Claim-Level Verification Breakdown")

    for i, claim_data in enumerate(claims):
        claim_text = claim_data.get("claim", "")
        status = claim_data.get("status", "UNSUPPORTED")
        conf = claim_data.get("confidence", 0.0)
        method = claim_data.get("method", "lightweight")
        best_chunk = claim_data.get("best_chunk")
        evidence_snippet = claim_data.get("best_evidence_snippet", "")
        c_style = STATUS_COLORS.get(status, STATUS_COLORS["UNSUPPORTED"])

        c_badge = render_verification_badge(status, conf)

        # Method badge
        method_label = "ModernBERT NLI" if "modernbert" in method else ("Lightweight Filter" if "lightweight" in method else method)
        method_color = "#8b5cf6" if "modernbert" in method else "#3b82f6"
        method_badge = (
            f"<span style='"
            f"display: inline-block;"
            f"border: 1px solid {method_color};"
            f"color: {method_color};"
            f"background: transparent;"
            f"padding: 1px 7px;"
            f"border-radius: 9999px;"
            f"font-size: 10px;"
            f"font-weight: 600;"
            f"margin-left: 6px;"
            f"vertical-align: middle;"
            f"'>"
            f"⚡ {method_label}"
            f"</span>"
        )

        # Build metadata text
        meta_text = ""
        if status != "UNSUPPORTED" and best_chunk:
            sec = best_chunk.get("section", "Unknown")
            ch = best_chunk.get("chapter", "Unknown")
            p_start = best_chunk.get("page_start", "?")
            p_end = best_chunk.get("page_end", "?")
            meta_text = f"{sec} &nbsp;|&nbsp; {ch} &nbsp;|&nbsp; Pages {p_start}–{p_end}"
        elif status == "UNSUPPORTED":
            meta_text = "<em>None &mdash; Claim is not established by retrieved chunks</em>"

        # Evidence text formatting
        evidence_display = evidence_snippet
        if not evidence_display or status == "UNSUPPORTED":
            evidence_display = "No supporting evidence found in the retrieved textbook context."

        st.markdown(
            f"<div style='"
            f"border: 1px solid rgba(128, 128, 128, 0.15);"
            f"border-left: 3px solid {c_style['border']};"
            f"border-radius: 0 6px 6px 0;"
            f"padding: 10px 14px;"
            f"margin-bottom: 10px;"
            f"background: rgba(255, 255, 255, 0.02);"
            f"'>"
            f"<div style='font-size: 13px; font-weight: 600; margin-bottom: 4px;'>"
            f"Claim {i+1}: \"{claim_text}\" {c_badge} {method_badge}"
            f"</div>"
            + (
                f"<div style='font-size: 11px; color: gray; margin-bottom: 6px;'>"
                f"Source: {meta_text}"
                f"</div>"
                if meta_text else ""
            )
            + f"<div style='font-size: 12px; font-style: italic; color: {'#9ca3af' if status == 'UNSUPPORTED' else '#d1d5db'}; border-top: 1px dashed rgba(128, 128, 128, 0.2); padding-top: 6px; margin-top: 4px;'>"
            + f"Evidence: \"{evidence_display}\""
            + f"</div>"
            + f"</div>",
            unsafe_allow_html=True
        )

    # ── Expandable Details for All 7 Chunks ──
    with st.expander("🔍 Detailed NLI Evaluations Across All 7 Chunks", expanded=False):
        for i, claim_data in enumerate(claims):
            st.markdown(f"**Claim {i+1}:** *\"{claim_data.get('claim', '')}\"*")
            evals = claim_data.get("evaluations", [])

            table_rows = []
            for j, ev in enumerate(evals):
                entail = ev.get("entailment", 0.0)
                contra = ev.get("contradiction", 0.0)
                neutral = ev.get("neutral", 0.0)

                if entail >= contra and entail >= neutral:
                    top_lbl = "Entailment"
                elif contra >= entail and contra >= neutral:
                    top_lbl = "Contradiction"
                else:
                    top_lbl = "Neutral"

                table_rows.append({
                    "Chunk": f"Chunk {j+1}",
                    "Section": ev.get("section", "Unknown"),
                    "Pages": f"{ev.get('page_start', '?')}–{ev.get('page_end', '?')}",
                    "Entailment": f"{entail*100:.1f}%",
                    "Neutral": f"{neutral*100:.1f}%",
                    "Contradiction": f"{contra*100:.1f}%",
                    "Top NLI Label": top_lbl
                })

            if table_rows:
                df_evals = pd.DataFrame(table_rows)
                st.dataframe(df_evals, use_container_width=True, hide_index=True)
            st.markdown("---")
