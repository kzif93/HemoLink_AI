import streamlit as st
import pandas as pd
from src.data_loader import load_geo_series_matrix
from src.preprocessing import preprocess_features
from src.feature_engineering import reduce_features
from src.model_training import train_model
from src.prediction import predict
from src.explainability import shap_summary_plot
from src.gene_mapper import align_cross_species_data

st.set_page_config(page_title="HemoLink_AI", layout="wide")
st.title("🧠 HemoLink_AI: Predict Preclinical to Clinical Translation")

st.markdown("Upload a gene expression dataset to begin:")

uploaded_file = st.file_uploader("📂 Upload GEO Series Matrix (.txt) or CSV file", type=["txt", "csv"])

if uploaded_file:
    try:
        X, labels, metadata_df = load_geo_series_matrix(uploaded_file)
        st.success("✅ File loaded")
        st.write(f"📊 Data shape (rows = samples, cols = genes): {X.shape}")
        st.write(f"🔢 Number of labels: {len(labels)}")
        st.write(f"🧬 Unique label classes: {set(labels)}")

        if len(set(labels)) < 2:
            st.warning("⚠️ Only one class detected in labels. Classifier may fail.")
        else:
            X = preprocess_features(X)
            X = reduce_features(X)

            model, acc = train_model(X, labels)
            st.success(f"✅ Model trained (accuracy: {acc:.2f})")

            preds = predict(model, X)
            st.dataframe(preds.head())

            shap_summary_plot(model, X)

    except Exception as e:
        st.error(f"❌ Error: {e}")

# 🧠 Cross-species modeling
st.markdown("---")
st.subheader("🧠 Train on Mouse ➜ Predict on Human")

mouse_file = st.file_uploader("🐭 Upload mouse GEO matrix (.txt)", type=["txt"], key="mouse")
human_file = st.file_uploader("👤 Upload human GEO matrix (.txt)", type=["txt"], key="human")

if mouse_file and human_file:
    try:
        mouse_data, _ = load_geo_series_matrix(mouse_file)
        human_data, _ = load_geo_series_matrix(human_file)

        mouse_aligned, human_aligned, shared = align_cross_species_data(mouse_data, human_data)

        st.success(f"✅ Shared genes: {len(shared)}")
        st.write(f"🐭 Mouse shape: {mouse_aligned.shape}")
        st.write(f"👤 Human shape: {human_aligned.shape}")
        st.write("🧬 Sample shared genes:")
        st.code(shared[:10])

        if len(shared) == 0:
            st.error("❌ No shared genes found. Likely due to unmapped probe IDs.")
        else:
            dummy_labels = [0, 1] * (len(mouse_aligned) // 2) + [0] * (len(mouse_aligned) % 2)
            model, acc = train_model(mouse_aligned, dummy_labels)

            human_preds = predict(model, human_aligned)
            st.success("✅ Cross-species prediction completed.")
            st.dataframe(human_preds.head())

    except Exception as e:
        st.error(f"❌ Cross-species error: {e}")
