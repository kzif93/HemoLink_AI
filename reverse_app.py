# reverse_app.py

import os
import sys
import streamlit as st
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from reverse_modeling import evaluate_mouse_models

st.set_page_config(page_title="Reverse Modeling AI", layout="wide")

# -------------------- HEADER --------------------
st.markdown("""
    <h1>🔁 Reverse Translational Model Discovery</h1>
    <h4 style='color: gray;'>Train on human data ➜ discover best-fitting animal models</h4>
""", unsafe_allow_html=True)

# -------------------- UPLOAD --------------------
st.markdown("### 🧬 Upload Human Expression Data")
human_file = st.file_uploader("CSV file (samples as rows, gene symbols as columns)", type=["csv"])
label_input = st.text_area("Paste binary labels (comma-separated, e.g., 1,0,0,1)")

# -------------------- LOAD ORTHOLOGS --------------------
ortholog_path = "data/mouse_to_human_orthologs.csv"
if not os.path.exists(ortholog_path):
    st.error("Ortholog map not found at data/mouse_to_human_orthologs.csv")
else:
    ortholog_df = pd.read_csv(ortholog_path)

# -------------------- RUN --------------------
if human_file and label_input:
    human_df = pd.read_csv(human_file, index_col=0)
    y = pd.Series([int(x) for x in label_input.strip().split(",")])

    st.success(f"✅ Loaded {human_df.shape[0]} samples and {human_df.shape[1]} genes.")

    with st.spinner("🔁 Evaluating animal models..."):
        summary_df = evaluate_mouse_models(human_df, y, ortholog_df)
        st.markdown("### 📊 Model Ranking")
        st.dataframe(summary_df, use_container_width=True)

        if st.button("💾 Export Results"):
            summary_df.to_csv("reverse_model_summary.csv", index=False)
            st.success("Saved to reverse_model_summary.csv")

else:
    st.info("Please upload your human dataset and provide class labels to begin.")
