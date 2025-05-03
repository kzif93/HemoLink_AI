# app.py

import os
import sys
import streamlit as st
import pandas as pd

# Extend sys path to access src/
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from preprocessing import clean_and_scale
from ortholog_mapper import map_orthologs
from model_training import train_model
from prediction import predict_on_human
from explainability import generate_shap_plots

st.set_page_config(page_title="HemoLink_AI", layout="wide")
st.title("🧠 HemoLink_AI: Cross-Species Thrombosis Predictor")

try:
    st.info("📂 Loading datasets from local repository...")

    # ✅ Load and transpose mouse data
    mouse_df = pd.read_csv("GSE125965_annotated_cleaned.csv", index_col=0).T

    # ✅ Automatically extract DVT labels from sample names
    y_mouse = mouse_df.index.to_series().apply(
        lambda x: 1 if "DVT" in x.upper() else 0
    )

    # ✅ Load human and ortholog files
    human_df = pd.read_csv("data/compressed_data.csv.gz", compression="gzip", index_col=0)
    ortholog_df = pd.read_csv("data/mouse_to_human_orthologs.csv")

    # Normalize gene names to uppercase
    mouse_df.columns = mouse_df.columns.str.upper()
    human_df.columns = human_df.columns.str.upper()
    ortholog_df["mouse_symbol"] = ortholog_df["mouse_symbol"].str.upper()
    ortholog_df["human_symbol"] = ortholog_df["human_symbol"].str.upper()

    # Debug display
    st.write("🧬 Sample mouse genes:", list(mouse_df.columns[:10]))
    st.write("🧬 Sample human genes:", list(human_df.columns[:10]))
    st.write("🧬 Sample orthologs:", ortholog_df.head())
    st.success("✅ Files loaded and normalized.")

    # 1. Align by shared orthologs
    mouse_aligned, human_aligned = map_orthologs(mouse_df, human_df, ortholog_df)

    # 2. Preprocess features
    mouse_scaled = clean_and_scale(mouse_aligned)
    human_scaled = clean_and_scale(human_aligned)

    # 3. Train model on mouse data
    st.header("🧪 Training on Mouse Data")
    model, metrics = train_model(mouse_scaled, y_mouse)
    st.write("📊 Training Metrics:", metrics)

    # 4. Predict on human data
    st.header("🔍 Predicting on Human Data")
    predictions = predict_on_human(model, human_scaled)
    st.dataframe(predictions)

    # 5. SHAP explanations
    st.header("🧬 SHAP Explainability")
    if st.checkbox("Show SHAP explanations"):
        shap_fig = generate_shap_plots(model, human_scaled)
        st.pyplot(shap_fig)

except Exception as e:
    st.error("❌ Failed to load or process data.")
    st.exception(e)
