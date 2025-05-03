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

# Streamlit App Config
st.set_page_config(page_title="HemoLink_AI", layout="wide")
st.title("🧠 HemoLink_AI: Cross-Species Thrombosis Predictor")

# 🗂 Load all files directly from local repo
try:
    st.info("📂 Loading datasets from local repository...")

    mouse_df = pd.read_csv("GSE125965_annotated_cleaned.csv", index_col=0)
    human_df = pd.read_csv("data/compressed_data.csv.gz", compression="gzip", index_col=0)
    ortholog_df = pd.read_csv("data/mouse_to_human_orthologs.csv")

    st.success("✅ Files loaded successfully!")

    # 1. Map orthologs
    mouse_aligned, human_aligned = map_orthologs(mouse_df, human_df, ortholog_df)

    # 2. Preprocess
    mouse_scaled = clean_and_scale(mouse_aligned)
    human_scaled = clean_and_scale(human_aligned)

    # 3. Train model on mouse
    st.header("🧪 Training on Mouse Data")
    model, metrics = train_model(mouse_scaled)
    st.write("📊 Training Metrics:", metrics)

    # 4. Predict on human
    st.header("🔍 Predicting on Human Data")
    predictions = predict_on_human(model, human_scaled)
    st.dataframe(predictions)

    # 5. SHAP Explainability
    st.header("🧬 SHAP Explainability")
    if st.checkbox("Show SHAP explanations"):
        shap_fig = generate_shap_plots(model, human_scaled)
        st.pyplot(shap_fig)

except Exception as e:
    st.error("❌ Failed to load or process data.")
    st.exception(e)
