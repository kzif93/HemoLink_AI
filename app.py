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
