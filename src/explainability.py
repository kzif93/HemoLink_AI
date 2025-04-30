import shap
import streamlit as st
import matplotlib.pyplot as plt

def show_shap_summary(model, X_test):
    st.subheader("🔍 Feature Importance (SHAP)")

    # Use TreeExplainer for decision trees (RandomForest)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Only take shap_values[1] if binary classification
    if isinstance(shap_values, list) and len(shap_values) > 1:
        shap_values = shap_values[1]

    # Create figure safely
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    st.pyplot(fig)
