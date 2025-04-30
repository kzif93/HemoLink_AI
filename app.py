import streamlit as st
from src.data_loader import load_geo_series_matrix
from src.model_training import train_random_forest
from src.prediction import predict_and_display
from src.explainability import show_shap_summary, show_shap_comparison
from src.preprocessing import clean_and_scale
from src.feature_engineering import reduce_low_variance_features
from src.feature_selection import select_top_shap_features
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Page setup
st.set_page_config(page_title="HemoLink_AI", layout="wide")
st.image("images/hemolink_logo.png", width=300)
st.title("🧠 HemoLink_AI: Predict Clinical Translatability")

# Upload
st.markdown("### 📂 Upload Data")
uploaded_file = st.file_uploader("Upload a GEO .txt or biomarker .csv file", type=["txt", "csv"])

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

        with st.spinner("Loading..."):
            data, labels, metadata = load_geo_series_matrix(uploaded_file)
            st.success("✅ File loaded")

        st.metric("Samples", data.shape[0])
        st.metric("Genes (Raw)", data.shape[1])
        st.metric("Detected Classes", len(set(labels)))

        if len(set(labels)) < 2:
            st.warning("⚠️ Only one class detected. Model may not work.")

        # Preprocessing
        data = clean_and_scale(data)
        data = reduce_low_variance_features(data, threshold=0.01)

        # Feature selection
        st.markdown("### 🎯 Feature Selection with SHAP")
        top_n = st.slider("Select number of top SHAP features to keep:", 10, 500, 100, step=10)
        data = select_top_shap_features(data, labels, top_n=top_n)
        st.write(f"✅ Features after SHAP filtering: {data.shape[1]}")
        st.dataframe(data.head())

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

        # Show top SHAP genes
        st.markdown("### 🧬 Top SHAP Features")
        from sklearn.ensemble import RandomForestClassifier
        import shap
        model_temp = RandomForestClassifier(n_estimators=50, random_state=42).fit(X_train, y_train)
        explainer_temp = shap.TreeExplainer(model_temp)
        shap_values_temp = explainer_temp.shap_values(X_train)
        if isinstance(shap_values_temp, list):
            shap_values_temp = shap_values_temp[1]
        shap_df = pd.DataFrame({
            "Feature": X_train.columns,
            "SHAP Importance": np.abs(shap_values_temp).mean(axis=0)
        }).sort_values(by="SHAP Importance", ascending=False).reset_index(drop=True)
        st.dataframe(shap_df.head(15))

        # Subgroup filter
        metadata.index = data.index
        metadata_test = metadata.loc[X_test.index]

        st.markdown("### 🧪 SHAP Subgroup Comparison (Optional)")
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
                st.success(f"🧠 Comparing SHAP for '{subgroup_a}' vs '{subgroup_b}'")

        # Train model
        st.markdown("### 🧠 Train Model")
        with st.spinner("Training..."):
            model, acc, _, _ = train_random_forest(data, labels)
            st.metric("Final Model Accuracy", f"{acc:.2f}")
            st.success("✅ Model trained")

        # Predict
        st.markdown("### 🔬 Predict")
        with st.spinner("Predicting..."):
            predict_and_display(model, X_test, y_test)

        # SHAP Explainability
        st.markdown("### 🔍 SHAP Feature Importance")
        with st.spinner("Explaining with SHAP..."):
            if 'X_a' in locals() and 'X_b' in locals() and len(X_a) > 1 and len(X_b) > 1:
                show_shap_comparison(model, {subgroup_a: X_a, subgroup_b: X_b})
            elif len(set(y_test)) > 1:
                show_shap_summary(model, X_test)
            else:
                st.warning("⚠️ Cannot show SHAP — only one class in test set.")

    except Exception as e:
        st.error(f"❌ Error: {e}")
else:
    st.info("👈 Please upload your file to begin.")
