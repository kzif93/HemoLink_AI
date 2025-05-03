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

# 1. Upload CSV or TXT files
st.header("📂 Upload Data")
mouse_file = st.file_uploader("Upload preprocessed **mouse** expression matrix (.csv or .txt)", type=["csv", "txt"])
human_file = st.file_uploader("Upload preprocessed **human** expression matrix (.csv or .txt)", type=["csv", "txt"])
ortholog_file = st.file_uploader("Upload mouse-to-human ortholog mapping (.csv or .txt)", type=["csv", "txt"])

if mouse_file and human_file and ortholog_file:
    # 2. Load files with auto delimiter detection
    mouse_df = pd.read_csv(mouse_file, sep=None, engine="python", index_col=0)
    human_df = pd.read_csv(human_file, sep=None, engine="python", index_col=0)
    ortholog_df = pd.read_csv(ortholog_file, sep=None, engine="python")

    st.success("✅ All files loaded successfully.")

    # 3. Map orthologs
    mouse_aligned, human_aligned = map_orthologs(mouse_df, human_df, ortholog_df)

    # 4. Preprocess
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
    st.info("Please upload all three files (.csv or .txt) to begin.")
