import streamlit as st
import pandas as pd
from src.data_loader import load_geo_series_matrix
from src.preprocessing import preprocess_features
from src.feature_engineering import reduce_features
from src.model_training import train_model
from src.prediction import predict
from src.explainability import shap_summary_plot
from src.gene_mapper import align_cross_species_data
from src.annotator import load_annotation_file, annotate_expression_matrix

st.set_page_config(page_title="HemoLink_AI", layout="wide")
st.title("🧠 HemoLink_AI: Predict Preclinical to Clinical Translation")

# ────────────────────────────────────────────────
# Upload and train standard matrix
# ────────────────────────────────────────────────
st.markdown("### 📂 Upload GEO Series Matrix")

uploaded_file = st.file_uploader("Upload GEO matrix (.txt or .csv)", type=["txt", "csv"])

if uploaded_file:
    try:
        X, labels, metadata_df = load_geo_series_matrix(uploaded_file)
        st.success("✅ File loaded")
        st.write(f"📊 Data shape: {X.shape}")
        st.write(f"🔢 Labels: {len(labels)} | Classes: {set(labels)}")

        if len(set(labels)) < 2:
            st.warning("⚠️ Only one class detected. Model training may not work.")
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

# ────────────────────────────────────────────────
# Annotate matrix using platform file
# ────────────────────────────────────────────────
st.markdown("---")
st.subheader("🧬 Annotate GEO Matrix with Gene Symbols")

expr_file = st.file_uploader("📂 Upload GEO matrix (.txt)", type=["txt"], key="expr")
annot_file = st.file_uploader("🧾 Upload platform annotation (.annot.gz or .txt)", type=["gz", "txt"], key="annot")

if expr_file and annot_file:
    try:
        expr_df, _ = load_geo_series_matrix(expr_file)
        annot_map = load_annotation_file(annot_file)
        annotated = annotate_expression_matrix(expr_df, annot_map)

        st.success(f"✅ Annotation complete. Matrix shape: {annotated.shape}")
        st.dataframe(annotated.iloc[:, :10])

    except Exception as e:
        st.error(f"❌ Annotation failed: {e}")

# ────────────────────────────────────────────────
# Cross-species modeling
# ────────────────────────────────────────────────
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
