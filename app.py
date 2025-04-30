import streamlit as st
from src.data_loader import load_geo_series_matrix
from src.model_training import train_random_forest
from src.prediction import predict_and_display
from src.explainability import show_shap_summary
from src.preprocessing import clean_and_scale
from src.feature_engineering import reduce_low_variance_features

st.set_page_config(page_title="HemoLink_AI", layout="wide")
st.title("🧠 HemoLink_AI: Predict Clinical Translatability")

uploaded_file = st.file_uploader(
    "📂 Upload your GEO series_matrix.txt or biomarker .csv file",
    type=["txt", "csv"]
)

if uploaded_file is not None:
    try:
        file_content = uploaded_file.read()
        if isinstance(file_content, bytes):
            decoded_content = file_content.decode("utf-8")
        else:
            decoded_content = file_content

        meta_lines = [line for line in decoded_content.splitlines() if "characteristics_ch1" in line.lower()]
        st.write("🧾 Raw metadata lines:")
        st.text("\n".join(meta_lines))

        from io import BytesIO
        uploaded_file = BytesIO(file_content)

        with st.spinner("Loading file..."):
            data, labels, metadata = load_geo_series_matrix(uploaded_file)
            st.success("✅ File loaded")

            st.write("📊 Data shape (rows = samples, cols = genes):", data.shape)
            st.write("🔢 Number of labels:", len(labels))
            st.write("🧬 Unique label classes:", set(labels))

            if len(set(labels)) < 2:
                st.warning("⚠️ Only one class detected in labels. Classifier may fail.")

        # ✅ Add preprocessing + feature reduction
        data = clean_and_scale(data)
        data = reduce_low_variance_features(data, threshold=0.01)
        st.write("🔍 Features after preprocessing:", data.shape[1])

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
        st.error(f"❌ Error: {e}")
else:
    st.info("👈 Upload a `.txt` or `.csv` biomarker matrix to begin.")
