import streamlit as st
from src.data_loader import load_geo_series_matrix
from src.model_training import train_random_forest
from src.prediction import predict_and_display
from src.explainability import show_shap_summary

st.set_page_config(page_title="HemoLink_AI", layout="wide")
st.title("🧠 HemoLink_AI: Predict Clinical Translatability")

uploaded_file = st.file_uploader(
    "📂 Upload your GEO series_matrix.txt or biomarker .csv file",
    type=["txt", "csv"]
)

if uploaded_file is not None:
    try:
        with st.spinner("Loading file..."):
            data, labels, metadata = load_geo_series_matrix(uploaded_file)
            st.success("✅ File loaded")
            st.write("Preview:", data.head())

        with st.spinner("Training model..."):
            model, acc, X_test, y_test = train_random_forest(data, labels)
            st.success(f"✅ Model trained (accuracy: {acc:.2f})")

        with st.spinner("Predicting..."):
            predict_and_display(model, X_test, y_test)

        with st.spinner("Explaining predictions..."):
            show_shap_summary(model, X_test)

        st.write("📋 Metadata (if available):")
        st.dataframe(metadata)

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("👈 Upload a `.txt` or `.csv` biomarker matrix to start.")
