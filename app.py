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
from data_fetcher import dataset_search_ui
from probe_mapper import download_platform_annotation, map_probes_to_genes

st.set_page_config(page_title="HemoLink_AI", layout="wide")

# -------------------- HEADER --------------------
st.markdown("""
    <div style="display: flex; align-items: center; gap: 20px;">
        <img src="https://raw.githubusercontent.com/kzif93/HemoLink_AI/main/assets/logo.png" width="200">
        <div>
            <h1 style="margin-bottom: 0;">HemoLink_AI</h1>
            <h3 style="margin-top: 0; color: #ccc;">🧬 From genes to models: AI for cross-species disease insight</h3>
        </div>
    </div>
""", unsafe_allow_html=True)

# -------------------- GEO UI --------------------
dataset_search_ui()

# -------------------- DATA LOADING --------------------
st.info("📂 Loading and aligning data...")

try:
    # Load mouse dataset
    mouse_df = pd.read_csv("GSE125965_annotated_cleaned.csv", index_col=0).T
    y_mouse = pd.Series({
        "GSM3586432": 0,
        "GSM3586433": 1,
        "GSM3586434": 0,
        "GSM3586435": 1
    })

    # Automatically select latest downloaded GEO expression file
    human_files = [f for f in os.listdir("data") if f.endswith("_expression.csv")]
    human_files.sort(reverse=True)
    if not human_files:
        raise FileNotFoundError("No GEO expression CSV found in data/ directory.")

    latest_human_path = os.path.join("data", human_files[0])
    st.success(f"✅ Using {os.path.basename(latest_human_path)} and mouse dataset.")
    human_df = pd.read_csv(latest_human_path, index_col=0)

    # Detect probe-style IDs (heuristic: 90%+ look like probes)
    def looks_like_probe(val):
        return str(val).endswith("_at") or str(val).isdigit()

    probe_like = human_df.index.to_series().apply(looks_like_probe).mean()
    if probe_like > 0.9:
        st.warning("🔍 Detected probe-style IDs. Mapping to gene symbols...")
        gse_id = os.path.basename(latest_human_path).split("_")[0]  # e.g. GSE16561
        gpl_path = download_platform_annotation(gse_id)
        human_df = map_probes_to_genes(latest_human_path, gpl_path)
        human_df = human_df.T  # 🔄 transpose to make genes columns

    # Uppercase for ortholog matching
    mouse_df.columns = mouse_df.columns.str.upper()
    human_df.columns = human_df.columns.str.upper()
    ortholog_df = pd.read_csv("data/mouse_to_human_orthologs.csv")
    ortholog_df["mouse_symbol"] = ortholog_df["mouse_symbol"].str.upper()
    ortholog_df["human_symbol"] = ortholog_df["human_symbol"].str.upper()

    # ---------------- DEBUG --------------------
    st.write("🧬 Sample human gene columns:", list(human_df.columns[:10]))
    st.write("🧬 Sample ortholog human symbols:", list(ortholog_df["human_symbol"].unique()[:10]))
    shared = set(human_df.columns).intersection(set(ortholog_df["human_symbol"]))
    st.write(f"🧬 Shared gene symbols: {len(shared)}")

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
        top_genes = gene_names[:10] if gene_names else []
        if top_genes:
            enrich_df = enrich_genes(top_genes, library="GO_Biological_Process_2021", top_n=10)
            st.dataframe(enrich_df)
        else:
            st.warning("⚠️ No genes available for enrichment analysis.")

except Exception as e:
    st.error("❌ Something went wrong.")
    st.exception(e)
