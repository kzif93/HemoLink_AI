import streamlit as st
from src.data_loader import load_geo_series_matrix
from src.model_training import train_random_forest
from src.prediction import predict_and_display
from src.explainability import show_shap_summary, show_shap_comparison
from src.preprocessing import clean_and_scale
from src.feature_engineering import reduce_low_variance_features

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

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
            decoded_content = file_content.decode("utf-8", errors="ignore")
        else:
            decoded_content = file_content

        if uploaded_file.name.endswith(".txt"):
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

        # Clean and reduce
        data = clean_and_scale(data)
        data = reduce_low_variance_features(data, threshold=0.01)
        st.write("🔍 Features after preprocessing:", data.shape[1])

        # Split
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

        metadata.index = data.index
        metadata_test = metadata.loc[X_test.index]

        # Subgroup SHAP
        if not metadata.empty and len(metadata.columns) > 0:
            selected_column = st.selectbox("📊 Select metadata column for subgroup SHAP:", metadata.columns)

            if selected_column:
                options = metadata_test[selected_column].dropna().unique().tolist()
                selected_values = st.multiselect(
                    f"🎯 Choose TWO values from '{selected_column}':", options
                )

                if len(selected_values) == 2:
                    subgroup_a = selected_values[0]
                    subgroup_b = selected_values[1]

                    mask_a = metadata_test[selected_column] == subgroup_a
                    mask_b = metadata_test[selected_column] == subgroup_b

                    X_a = X_test[mask_a]
                    X_b = X_test[mask_b]

                    st.success(f"🎉 Comparing SHAP for '{subgroup_a}' vs '{subgroup_b}'")

        with st.spinner("Training model..."):
            model, acc, _, _ = train_random_forest(data, labels)
            st.success(f"✅ Model trained (accuracy: {acc:.2f})")

        with st.spinner("Predicting..."):
            predict_and_display(model, X_test, y_test)

        with st.spinner("Explaining predictions..."):
            if 'X_a' in locals() and 'X_b' in locals() and len(X_a) > 1 and len(X_b) > 1:
                show_shap_comparison(model, {
                    subgroup_a: X_a,
                    subgroup_b: X_b
                })
            elif len(set(y_test)) > 1:
                show_shap_summary(model, X_test)
            else:
                st.warning("⚠️ Cannot show SHAP — only one class present in test set.")

    except Exception as e:
        st.error(f"❌ Error: {e}")
else:
    st.info("👈 Upload a `.txt` or `.csv` biomarker matrix to begin.")
