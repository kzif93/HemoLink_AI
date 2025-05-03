# app.py

import os
import sys
import streamlit as st
import pandas as pd

# 🔧 Ensure src/ folder is in Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# ✅ Import modules from src/
from preprocessing import clean_and_scale
from ortholog_mapper import map_orthologs
from model_training import train_model
from prediction import predict_on_human
from explainability import generate_shap_plots

# ⬇️ Streamlit App UI Starts Here
st.set_page_config(page_title="HemoLink_AI", layout="wide")
st.title("🧠 HemoLink_AI: Cross-Species Thrombosis Predictor")

# 1. Upload CSVs
st.header("📂 Upload Data")
mouse_file = st.file_uploader("Upload preprocessed **mouse** expression matrix (.csv)", type="csv")
human_file = st.file_uploader("Upload preprocessed **human** expression matrix (.csv)", type="csv")
ortholog_file = st.file_uploader("Upload mouse-to-human ortholog mapping (.csv)", type="csv")

if mouse_file and human_file and ortholog_file:
    # 2. Load CSVs
    mouse_df = pd.read_csv(mouse_file, index_col=0)
    human_df = pd.read_csv(human_file, index_col=0)
    ortholog_df = pd.read_csv(ortholog_file)

    st.success("✅ All files loaded successfully.")

    # 3. Map orthologs
    mouse_aligned, human_aligned = map_orthologs(mouse_df, human_df, ortholog_df)

    # 4. Preprocess (clean + scale)
    mouse_scaled = clean_and_scale(mouse_aligned)
    human_scaled = clean_and_scale(human_aligned)

    # 5. Train model on mouse
    st.header("🧪 Training on Mouse Data")
    model, metrics = train_model(mouse_scaled)
    st.write("📊 Training Metrics:", metrics)

    # 6. Predict on human
    st.header("🔍 Predicting on Human Data")
    predictions = predict_on_human(model, human_scaled)
    st.dataframe(predictions)

    # 7. SHAP Explainability
    st.header("🧬 SHAP Explainability")
    if st.checkbox("Show SHAP explanations"):
        shap_fig = generate_shap_plots(model, human_scaled)
        st.pyplot(shap_fig)

else:
    st.info("Please upload all three CSV files to begin: mouse, human, and ortholog mapping.")
