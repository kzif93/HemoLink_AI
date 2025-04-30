import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from src.data_loader import load_geo_series_matrix
from src.model_training import train_random_forest
from src.prediction import predict_and_display
from src.explainability import show_shap_summary, show_shap_comparison
from src.preprocessing import clean_and_scale
from src.feature_engineering import reduce_low_variance_features
from src.feature_selection import select_top_shap_features
from src.gene_mapper import align_cross_species_data

# Page setup
st.set_page_config(page_title="HemoLink_AI", layout="wide")
st.image("images/hemolink_logo.png", width=300)
st.title("🧠 HemoLink_AI: Predict Clinical Translatability")

# ============================
# 📂 Main Upload Section
# ============================
st.markdown("### 📂 Upload Human GEO `.txt` or biomarker `.csv` file")

uploaded_file = st.file_uploader("Upload your file:", type=["txt", "csv"], key="main")

if uploaded_file is not None:
    try:
        file_content = uploaded_file.read()
        decoded_content = file_content.decode("utf-8", errors="ignore") if isinstance(file_content, bytes) else file_content

        if uploaded_file.name.endswith(".txt"):
            meta_lines = [line for line in decoded_content.splitlines() if "characteristics_ch1" in line.lower()]
            st.markdown("#### 🧾 Raw Metadata Lines")
            st.text("\n".join(meta_lines))
            from io import BytesIO
            uploaded_file = BytesIO(file_content)

        # Load
        with st.spinner("Loading..."):
            data, labels, metadata = load_geo_series_matrix(uploaded_file)
            st.success("✅ File loaded")

        st.metric("Samples", data.shape[0])
        st.metric("Genes (Raw)", data.shape[1])
        st.metric("Detected Classes", len(set(labels)))

        if len(set(labels)) < 2:
            st.warning("⚠️ Only one class detected. Model may fail.")

        # Preprocessing
        data = clean_and_scale(data)
        data = reduce_low_variance_features(data, threshold=0.01)

        # Feature selection
        st.markdown("### 🎯 Feature Selection")
        top_n = st.slider("Select number of top SHAP features:", 10, 500, 100, step=10)
        data = select_top_shap_features(data, labels, top_n=top_n)
        st.write(f"✅ Features after SHAP filtering: {data.shape[1]}")
        st.dataframe(data.head())

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

        # Show top SHAP features
        st.markdown("### 🧬 Top SHAP Features")
        model_temp = train_random_forest(X_train, y_train)[0]
        explainer_temp = shap.TreeExplainer(model_temp)
        shap_values_temp = explainer_temp.shap_values(X_train)
        if isinstance(shap_values_temp, list):
            shap_values_temp = shap_values_temp[1]
        shap_df = pd.DataFrame({
            "Feature": X_train.columns,
            "SHAP Importance": np.abs(shap_values_temp).mean(axis=0)
        }).sort_values(by="SHAP Importance", ascending=False).reset_index(drop=True)
        st.dataframe(shap_df.head(15))

        # Metadata filtering (optional)
        metadata.index = data.index
        metadata_test = metadata.loc[X_test.index]
        st.markdown("### 🧪 Subgroup Comparison (Optional)")
        if not metadata.empty and len(metadata.columns) > 0:
            selected_column = st.selectbox("Select metadata column:", metadata.columns)
            options = metadata_test[selected_column].dropna().unique().tolist()
            selected_values = st.multiselect(f"Choose TWO values from '{selected_column}':", options)
            if len(selected_values) == 2:
                subgroup_a, subgroup_b = selected_values
                mask_a = metadata_test[selected_column] == subgroup_a
                mask_b = metadata_test[selected_column] == subgroup_b
                X_a = X_test[mask_a]
                X_b = X_test[mask_b]
                st.success(f"🎉 SHAP comparison: '{subgroup_a}' vs '{subgroup_b}'")

        # Train model
        st.markdown("### 🧠 Train Model")
        with st.spinner("Training..."):
            model, acc, _, _ = train_random_forest(data, labels)
            st.metric("Model Accuracy", f"{acc:.2f}")
            st.success("✅ Trained")

        # Predict
        st.markdown("### 🔬 Predict")
        with st.spinner("Predicting..."):
            predict_and_display(model, X_test, y_test)

        # SHAP explanation
        st.markdown("### 🔍 SHAP Interpretation")
        with st.spinner("Explaining with SHAP..."):
            if 'X_a' in locals() and 'X_b' in locals() and len(X_a) > 1 and len(X_b) > 1:
                show_shap_comparison(model, {subgroup_a: X_a, subgroup_b: X_b})
            elif len(set(y_test)) > 1:
                show_shap_summary(model, X_test)
            else:
                st.warning("⚠️ Cannot show SHAP — only one class in test set.")

    except Exception as e:
        st.error(f"❌ Error: {e}")

# ============================
# 🧬 Cross-Species Section
# ============================
st.markdown("## 🔁 Cross-Species Modeling (Mouse ➜ Human)")

mouse_file = st.file_uploader("🐭 Upload Mouse GEO .txt", type=["txt"], key="mouse")
human_file = st.file_uploader("👤 Upload Human GEO .txt", type=["txt"], key="human")

if mouse_file and human_file:
    try:
        with st.spinner("📥 Loading both species..."):
            mouse_data, mouse_labels, _ = load_geo_series_matrix(mouse_file)
            human_data, human_labels, _ = load_geo_series_matrix(human_file)
            mouse_data, human_data, shared_genes = align_cross_species_data(mouse_data, human_data)
            st.success(f"✅ Shared genes: {len(shared_genes)}")

        with st.spinner("🧠 Training on mouse data..."):
            model, acc, _, _ = train_random_forest(mouse_data, mouse_labels)
            st.metric("Mouse-trained Model Accuracy", f"{acc:.2f}")

        with st.spinner("🔬 Predicting on human data..."):
            preds = model.predict(human_data)
            st.dataframe(pd.DataFrame({
                "Prediction": preds,
                "Label": human_labels
            }, index=human_data.index))

        with st.spinner("🔍 SHAP Explanation on Human"):
            show_shap_summary(model, human_data)

    except Exception as e:
        st.error(f"❌ Cross-species error: {e}")
