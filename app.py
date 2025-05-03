# app.py

import os
import sys
import streamlit as st
import pandas as pd

# 🔧 Ensure src/ folder is in Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# ✅ Now safe to import modules from src/
from preprocessing import clean_and_scale
from ortholog_mapper import map_orthologs
from model_training import train_model
from prediction import predict_on_human
from explainability import generate_shap_plots

# ⬇️ Streamlit UI starts here
st.title("🧠 HemoLink_AI: Cross-Species Thrombosis Predictor")

# 1. Upload CSVs
mouse_file = st.file_uploader("Upload preprocessed **mouse** expression matrix (.csv)", type="csv")
human_file = st.file_uploader("Upload preprocessed **human** expression matrix (.csv)", type="csv")
ortholog_file = st.file_uploader("Upload mouse-to-human ortholog mapping (.csv)", type="csv")

if mouse_file and human_file and ortholog_file:
    mouse_df = pd.read_csv(mouse_file, index_col=0)
    human_df = pd.read_csv(human_file, index_col=0)
    ortholog_df = pd.read_csv(ortholog_file)

    st.success("✅ All files loaded successfully.")

    # 2. Map orthologs
    mouse_aligned, human_aligned = map_orthologs(mouse_df, human_df, ortholog_df)

    # 3. Preprocess data
    mouse_scaled = clean_and_scale(mouse_aligned)
    human_scaled = clean_and_scale(human_aligned)

    # 4. Train model
    model, metrics = train_model(mouse_scaled)
    st.write("📊 Training Metrics:", metrics)

    # 5. Predict on human
    predictions = predict_on_human(model, human_scaled)
    st.write("🧬 Human Predictions", predictions)

    # 6. SHAP Explainability
    if st.checkbox("Show SHAP explanations"):
        shap_fig = generate_shap_plots(model, human_scaled)
        st.pyplot(shap_fig)

else:
    st.warning("📂 Please upload all required files to continue.")
import streamlit as st
import pandas as pd
import numpy as np
from src.preprocessing import clean_and_scale
from src.feature_engineering import reduce_features
from src.model_training import train_model
from src.prediction import predict
from src.explainability import shap_summary_plot
from src.ortholog_mapper import load_ortholog_map, align_gene_symbols

st.set_page_config(page_title="HemoLink AI – Cross-Species", layout="wide")

st.title("🧬 HemoLink AI – Cross-Species Analysis (Mouse ➜ Human)")

# Upload pre-cleaned, annotated CSVs
st.header("📂 Upload Preprocessed CSVs")
mouse_file = st.file_uploader("🐭 Upload Mouse CSV", type=["csv"])
human_file = st.file_uploader("👤 Upload Human CSV", type=["csv"])

if mouse_file and human_file:
    try:
        mouse_df = pd.read_csv(mouse_file, index_col=0)
        human_df = pd.read_csv(human_file, index_col=0)

        st.success("✅ Both files loaded successfully.")
        st.write(f"🐭 Mouse shape: {mouse_df.shape}")
        st.write(f"👤 Human shape: {human_df.shape}")

        # Load ortholog map and align
        ortholog_map = load_ortholog_map()
        mouse_aligned, human_aligned, shared_genes = align_gene_symbols(mouse_df, human_df, ortholog_map)

        st.write(f"✅ Shared genes: {len(shared_genes)}")
        st.write(f"🧬 Sample shared genes:\n{shared_genes[:10]}")

        if len(shared_genes) < 2:
            st.error("❌ Too few shared genes for modeling.")
            st.stop()

        # Prepare X and y
        X_mouse = clean_and_scale(mouse_aligned)
        X_human = clean_and_scale(human_aligned)
        y_mouse = np.array([-1 if "noDVT" in s else 1 for s in mouse_aligned.index])  # Example label logic

        # Train on mouse, predict on human
        st.subheader("🧠 Train on Mouse ➜ Predict on Human")
        model = train_model(X_mouse, y_mouse)
        preds = predict(model, X_human)

        st.write("🎯 Prediction outputs:")
        st.dataframe(pd.DataFrame({"Sample": X_human.index, "Prediction": preds}))

        # Optional SHAP
        st.subheader("🔍 SHAP Feature Importance (Mouse Model)")
        try:
            shap_summary_plot(model, X_mouse)
        except Exception as e:
            st.warning(f"⚠️ SHAP plot failed: {e}")

    except Exception as e:
        st.error(f"❌ Failed to load or process CSVs: {e}")
