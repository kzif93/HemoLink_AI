import streamlit as st
import pandas as pd
import os

from src.annotator import annotate_expression_matrix, load_annotation_file
from src.preprocessing import clean_and_scale
from src.feature_engineering import reduce_features
from src.model_training import train_model
from src.prediction import predict
from src.explainability import shap_summary_plot
from src.ortholog_mapper import map_orthologs_cross_species

st.set_page_config(layout="wide")
st.title("🧬 HemoLink AI – Cross-Species Gene Expression Analyzer")

# Load matrix
st.sidebar.header("📁 Upload Expression Matrix")
expr_file = st.sidebar.file_uploader("Upload .csv or .txt matrix", type=["csv", "txt"])
matrix_df = None

if expr_file is not None:
    try:
        st.sidebar.success("✅ Matrix uploaded")
        if expr_file.name.endswith(".csv"):
            matrix_df = pd.read_csv(expr_file, index_col=0)
        elif expr_file.name.endswith(".txt"):
            lines = expr_file.read().decode("utf-8").splitlines()
            data_lines = [line for line in lines if not line.startswith("!") and not line.startswith("#")]
            from io import StringIO
            matrix_df = pd.read_csv(StringIO("\n".join(data_lines)), sep="\t", index_col=0)
        st.write("📊 Matrix preview:", matrix_df.iloc[:5, :5])
    except Exception as e:
        st.sidebar.error(f"❌ Failed to load matrix: {e}")

# Annotation step
st.sidebar.header("🧾 Upload Annotation File")
annot_file = st.sidebar.file_uploader("Upload .txt or .gz annotation", type=["txt", "gz"])
annotated_df = None

if st.sidebar.button("🔄 Annotate") and matrix_df is not None and annot_file is not None:
    try:
        st.info("🔄 Annotating...")
        annotation_map = load_annotation_file(annot_file)
        annotated_df = annotate_expression_matrix(matrix_df, annotation_map)
        st.success(f"✅ Annotation complete. Genes: {annotated_df.shape[1]}")
        st.write("🧬 Annotated preview:", annotated_df.iloc[:5, :5])
    except Exception as e:
        st.error(f"❌ Annotation failed: {e}")

# Model training + SHAP if annotation worked
if annotated_df is not None:
    try:
        st.header("🧪 Model Training")
        X = clean_and_scale(annotated_df)
        y = [1 if "dvt" in idx.lower() else 0 for idx in X.index]  # simple binary label inference
        X_reduced = reduce_features(X)
        model = train_model(X_reduced, y)
        preds = predict(model, X_reduced)
        st.success("✅ Model trained")
        st.write("📈 Predictions:", preds)
        st.header("📉 Feature Importance (SHAP)")
        shap_summary_plot(model, X_reduced)
    except Exception as e:
        st.error(f"❌ Model training or SHAP failed: {e}")

# Cross-species analysis
st.sidebar.header("🧬 Cross-Species Evaluation")
human_file = st.sidebar.file_uploader("Upload annotated human .csv", type=["csv"], key="human")
mouse_file = st.sidebar.file_uploader("Upload annotated mouse .csv", type=["csv"], key="mouse")
ortholog_file = st.sidebar.file_uploader("Upload ortholog CSV", type=["csv"], key="ortholog")

if st.sidebar.button("🚀 Run Cross-Species") and human_file and mouse_file and ortholog_file:
    try:
        st.header("🔁 Cross-Species Analysis")
        human_df = pd.read_csv(human_file, index_col=0)
        mouse_df = pd.read_csv(mouse_file, index_col=0)
        ortholog_df = pd.read_csv(ortholog_file)

        results = map_orthologs_cross_species(mouse_df, human_df, ortholog_df)
        st.success("✅ Cross-species mapping complete")
        st.write("🔬 Prediction preview:", results.head())
    except Exception as e:
        st.error(f"❌ Cross-species error: {e}")
