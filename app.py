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

# Set wide layout and page title
st.set_page_config(page_title="HemoLink_AI", layout="wide")

# -------------------- HEADER --------------------
st.markdown("""
    <div style="display: flex; align-items: center; gap: 15px;">
        <img src="https://raw.githubusercontent.com/kzif93/HemoLink_AI/main/assets/logo.png" width="60">
        <div>
            <h1 style="margin-bottom: 0;">HemoLink_AI</h1>
            <h3 style="margin-top: 0; color: #ccc;">🧬 From genes to models: AI for cross-species disease insight</h3>
        </div>
    </div>
""", unsafe_allow_html=True)

# -------------------- DATA LOADING --------------------
st.info("📂 Loading and aligning data...")

try:
    mouse_df = pd.read_csv("GSE125965_annotated_cleaned.csv", index_col=0).T
    y_mouse = pd.Series({
        "GSM3586432": 0,
        "GSM3586433": 1,
        "GSM3586434": 0,
        "GSM3586435": 1
    })

    human_df = pd.read_csv("data/compressed_data.csv.gz", compression="gzip", index_col=0)
    ortholog_df = pd.read_csv("data/mouse_to_human_orthologs.csv")

    mouse_df.columns = mouse_df.columns.str.upper()
    human_df.columns = human_df.columns.str.upper()
    ortholog_df["mouse_symbol"] = ortholog_df["mouse_symbol"].str.upper()
    ortholog_df["human_symbol"] = ortholog_df["human_symbol"].str.upper()

    st.success("✅ Data loaded and normalized.")

    # -------------------- ALIGNMENT --------------------
    mouse_aligned, human_aligned = map_orthologs(mouse_df, human_df, ortholog_df)
    mouse_scaled = clean_and_scale(mouse_aligned)
    human_scaled = clean_and_scale(human_aligned)

    # -------------------- MODEL TRAINING --------------------
    st.markdown("## 🧪 Train on Mouse Data")
    model, metrics = train_model(mouse_scaled, y_mouse)
    st.json(metrics)

    # -------------------- PREDICTION --------------------
    st.markdown("## 🔍 Predict on Human Samples")
    try:
        predictions = predict_on_human(model, human_scaled)
        st.dataframe(predictions)
    except:
        st.warning("⚠️ Model may lack class diversity — skipping predictions.")

    # -------------------- SHAP --------------------
    st.markdown("## 🧬 SHAP Explainability")
    if st.checkbox("Show SHAP Plot"):
        shap_fig, shap_matrix, gene_names = generate_shap_plots(model, human_scaled, return_values=True)
        st.pyplot(shap_fig)

        # -------------------- ENRICHMENT --------------------
        st.markdown("## 🧠 Pathway Enrichment (Top SHAP Genes)")
        top_genes = [
            "TRIM27", "ZMIZ1", "BTC", "HOOK2", "KRT32",
            "PTPN21", "CCL5", "ALDH1L1", "YWHAE", "SLC25A3"
        ]
        enrich_df = enrich_genes(top_genes, library="GO_Biological_Process_2021", top_n=10)
        st.dataframe(enrich_df)

except Exception as e:
    st.error("❌ Something went wrong.")
    st.exception(e)
