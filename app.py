# app.py

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np

# Extend sys path to access src/
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from preprocessing import clean_and_scale
from ortholog_mapper import map_orthologs
from model_training import train_model
from prediction import predict_on_human
from explainability import generate_shap_plots
from enrichment import enrich_genes

st.set_page_config(page_title="HemoLink_AI", layout="wide")
st.title("🧠 HemoLink_AI: Cross-Species Thrombosis Predictor")

try:
    st.info("📂 Loading datasets from local repository...")

    # Load and transpose mouse data
    mouse_df = pd.read_csv("GSE125965_annotated_cleaned.csv", index_col=0).T

    # Manual label assignment
    y_mouse = pd.Series({
        "GSM3586432": 0,
        "GSM3586433": 1,
        "GSM3586434": 0,
        "GSM3586435": 1
    })

    # Load human and ortholog files
    human_df = pd.read_csv("data/compressed_data.csv.gz", compression="gzip", index_col=0)
    ortholog_df = pd.read_csv("data/mouse_to_human_orthologs.csv")

    # Normalize gene names to uppercase
    mouse_df.columns = mouse_df.columns.str.upper()
    human_df.columns = human_df.columns.str.upper()
    ortholog_df["mouse_symbol"] = ortholog_df["mouse_symbol"].str.upper()
    ortholog_df["human_symbol"] = ortholog_df["human_symbol"].str.upper()

    st.success("✅ Files loaded and normalized.")

    # 1. Align by shared orthologs
    mouse_aligned, human_aligned = map_orthologs(mouse_df, human_df, ortholog_df)

    # 2. Preprocess features
    mouse_scaled = clean_and_scale(mouse_aligned)
    human_scaled = clean_and_scale(human_aligned)

    st.write("🧪 Is human_scaled a DataFrame?", isinstance(human_scaled, pd.DataFrame))
    st.write("🧪 Feature columns:", human_scaled.columns[:5])

    # 3. Train model on mouse data
    st.header("🧪 Training on Mouse Data")
    model, metrics = train_model(mouse_scaled, y_mouse)
    st.write("📊 Training Metrics:", metrics)

    # 4. Predict on human data
    st.header("🔍 Predicting on Human Data")
    try:
        predictions = predict_on_human(model, human_scaled)
        st.dataframe(predictions)
    except IndexError:
        st.warning("⚠️ Model was trained on a single class — skipping probability predictions.")

    # 5. SHAP Explainability
    st.header("🧬 SHAP Explainability")
    if st.checkbox("Show SHAP explanations"):
        shap_fig, shap_matrix, gene_names = generate_shap_plots(model, human_scaled, return_values=True)
        st.pyplot(shap_fig)

        # ➕ GO/Pathway Enrichment (Override version)
        st.subheader("🧬 GO/Pathway Enrichment for Top SHAP Genes")

        # 🔁 Manual override for top SHAP genes (testing only)
        top_genes = [
            "TRIM27", "ZMIZ1", "BTC", "HOOK2", "KRT32",
            "PTPN21", "CCL5", "ALDH1L1", "YWHAE", "SLC
