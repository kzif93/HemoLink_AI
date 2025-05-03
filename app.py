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

    # ✅ Automatically extract labels from sample names
    y_mouse = mouse_df.index.to_series().apply(
        lambda x: 1 if "DVT" in x.upper() and "SHAM" not in x.upper() else 0
    )
    st.write("📊 DVT label counts:", y_mouse.value_counts())

    # Check for valid binary classification
    if y_mouse.nunique() < 2:
        st.error("❌ Only one class found in y_mouse. You need at least one sample for each class.")
        st.stop()

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
    st.write("🧬 Sample
