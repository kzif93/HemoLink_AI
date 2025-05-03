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

# 1. Upload CSV, TXT, or GZ files
st.header("📂 Upload Data")
mouse_file = st.file_uploader("Upload preprocessed **mouse** expression matrix (.csv, .txt, or .gz)", type=["csv", "txt", "gz"])
human_file = st.file_uploader("Upload preprocessed **human** expression matrix (.csv, .txt, or .gz)", type=["csv", "txt", "gz"])
ortholog_file = st.file_uploader("Upload mouse-to-human ortholog mapping (.csv, .txt, or .gz)", type=["csv", "txt", "gz"])

# 2. Load ortholog mapping (use default if none uploaded)
if ortholog_file is None:
    default_path = os.path.join("data", "mouse_to_human_orthologs.csv")
    if os.path.exists(default_path):
        st.info("ℹ️ No ortholog file uploaded — using default from /data folder.")
        ortholog_df = pd.read_csv(default_path)
    else:
        st.error("❌ No ortholog file uploaded and default file not found.")
        st.stop()
else:
    ortholog_df = pd.read_csv(ortholog_file, sep=None, engine="python", compression="infer")

# 3. Load mouse and human files if both are uploaded
if mouse_file and human_file:
    mouse_df = pd.read_csv(mouse_file, sep=None, engine="python", index_col=0, compression="infer")
    human_df = pd.read_csv(human_file, sep=None, engine="python", index_col=0, compression="infer")

    st.success("✅ All files loaded successfully.")

    # 4. Map orthologs
    mouse_aligned, human_aligned = map_orthologs(mouse_df, human_df, ortholog_df)

    # 5. Preprocess
    mouse_scaled = clean_and_scale(mouse_aligned)
    human_scaled = clean_and_scale(human_aligned)

    # 6. Train model on mouse
    st.header("🧪 Training on Mouse Data")
    model, metrics = train_model(mouse_scaled)
    st.write("📊 Training Metrics:", metrics)

    # 7. Predict on human
    st.header("🔍 Predicting on Human Data")
    predictions = predict_on_human(model, human_scaled)
    st.dataframe(predictions)

    # 8. SHAP Explainability
    st.header("🧬 SHAP Explainability")
    if st.checkbox("Show SHAP explanations"):
        shap_fig = generate_shap_plots(model, human_scaled)
        st.pyplot(shap_fig)

else:
    st.info("Please upload the **mouse** and **human** expression files (.csv, .txt, or .gz) to begin.")
