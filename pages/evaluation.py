import streamlit as st
import json
import os

def render_evaluation():
    st.markdown("## RAG System Evaluation metrics")
    
    # ── Bulletproof Path Resolution ──
    base_dir = os.path.join(os.path.dirname(__file__), '..')
    eval_path = os.path.abspath(os.path.join(base_dir, "outputs", "evaluation.json"))
    
    if not os.path.exists(eval_path):
        st.warning(f"Evaluation data not found at {eval_path}\nPlease run `python src/evaluate.py` first.")
        return
        
    with open(eval_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    metrics = data.get("metrics", {})
    results = data.get("results", [])
    weak_cases = data.get("weak_cases", [])
    
    # ── Render Metrics ──
    st.markdown("### 📊 Performance Overview")
    
    # Format accuracy colors
    acc = metrics.get('Accuracy', 0)
    
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.metric(
            label="Accuracy (Semantic Sim > 0.8)", 
            value=f"{acc * 100:.1f}%", 
            delta="Excellent" if acc > 0.70 else "Needs Work"
        )
        
    with c2:
        st.metric(
            label="Avg Semantic Similarity", 
            value=f"{metrics.get('Average Semantic Similarity', 0):.3f}"
        )
        
    with c3:
        st.metric(
            label="Avg Token F1 Score", 
            value=f"{metrics.get('Average Token F1', 0):.3f}"
        )
    
    st.caption(f"Evaluated on {metrics.get('Total Questions', 0)} automated textbook queries against the Llama-3.1 model.")
    
    st.divider()
    
    # ── Render Weak Cases ──
    st.markdown("### ⚠️ Weak Cases Analysis")
    st.caption("Answers flagged because their raw sentence embedding similarity fell below 0.60 threshold.")
    
    if not weak_cases:
        st.success("✓ No weak cases found! The RAG system performed exceptionally well.")
    else:
        for wc in weak_cases:
            with st.expander(f"Question: {wc['question']}", expanded=True):
                st.markdown(f"**Target Answer:**  \n{wc['true']}")
                st.markdown(f"**RAG Prediction:**  \n{wc['pred']}")
                st.info(f"Similarity Penalty: {wc['similarity']:.3f}   |   Token F1 Penalty: {wc['f1']:.3f}")
                
    st.divider()
    
    # ── Render Full Breakdown ──
    st.markdown("### 📄 Full Dataset Predictions Log")
    with st.expander("View all 30 generated results..."):
        for i, res in enumerate(results):
            st.markdown(f"**Q{i+1}: {res['question']}**")
            st.markdown(f"> *True*: {res['true']}")
            st.markdown(f"> *RAG*: {res['pred']}")
            st.markdown("---")
