import shap
import streamlit as st
import matplotlib.pyplot as plt

# Original SHAP summary

def show_shap_summary(model, X_test):
    st.subheader("🔍 Feature Importance (SHAP)")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Handle binary classification
    if isinstance(shap_values, list) and len(shap_values) > 1:
        shap_values = shap_values[1]

    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    st.pyplot(fig)


# New side-by-side comparison

def show_shap_comparison(model, subgroup_dict):
    st.subheader("🧬 Subgroup SHAP Comparison")
    explainer = shap.TreeExplainer(model)

    col1, col2 = st.columns(2)
    for idx, (group_name, X_group) in enumerate(subgroup_dict.items()):
        if X_group.shape[0] < 2:
            st.warning(f"⚠️ Not enough samples in '{group_name}' for SHAP analysis.")
            continue

        shap_values = explainer.shap_values(X_group)
        if isinstance(shap_values, list) and len(shap_values) > 1:
            shap_values = shap_values[1]

        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, X_group, plot_type="bar", show=False)

        with (col1 if idx == 0 else col2):
            st.markdown(f"#### 🔎 {group_name}")
            st.pyplot(fig)
