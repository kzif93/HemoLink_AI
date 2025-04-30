import streamlit as st
from src.data_loader import load_geo_series_matrix
from src.model_training import train_random_forest
from src.prediction import predict_and_display
from src.explainability import show_shap_summary
from src.preprocessing import clean_and_scale
from src.feature_engineering import reduce_low_variance_features

import numpy as np
import pandas as pd

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

        # Clean + reduce
        data = clean_and_scale(data)
        data = reduce_low_variance_features(data, threshold=0.01)
        st.write("🔍 Features after preprocessing:", data.shape[1])

        # Split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

        # 🔍 Metadata filtering UI
        if not metadata.empty:
            metadata.index = data.index
            metadata_test = metadata.loc[X_test.index]

            # Debug print
            st.write("📋 Metadata preview:")
            st.dataframe(metadata.head())
            st.write("🧪 Metadata columns:", metadata.columns.tolist())

            if len(metadata.columns) > 0:
                selected_column = st.selectbox("📊 Select metadata column to filter SHAP:", metadata.columns)
                if selected_column:
                    options = metadata_test[selected_column].dropna().unique().tolist()
                    selected_value = st.selectbox(f"🎯 Choose value from '{selected_column}':", options)

                    mask = metadata_test[selected_column] == selected_value
                    X_test = X_test[mask]
                    y_test = [label for i, label in enumerate(y_test) if mask.iloc[i]]
                    st.success(f"🎉 Showing SHAP for {selected_column} = '{selected_value}' ({len(X_test)} samples)")
        else:
            st.warning("⚠️ No metadata available for filtering.")

        # Train and show
        with st.spinner("Training model..."):
            model, acc, _, _ = train_random_forest(data, labels)
            st.success(f"✅ Model trained (accuracy: {acc:.2f})")

        with st.spinner("Predicting..."):
            predict_and_display(model, X_test, y_test)

        with st.spinner("Explaining predictions..."):
            st.write("🛠 SHAP input feature count:", len(X_test.columns))
            show_shap_summary(model, X_test)

    except Exception as e:
        st.error(f"❌ Error: {e}")
else:
    st.info("👈 Upload a `.txt` or `.csv` biomarker matrix to begin.")
