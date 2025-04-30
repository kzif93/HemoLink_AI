import shap
import shap
import streamlit as st
import matplotlib.pyplot as plt

def show_shap_summary(model, X_test):
    st.subheader("🔍 Feature Importance (SHAP)")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Handle binary case
    if isinstance(shap_values, list) and len(shap_values) > 1:
        shap_values = shap_values[1]

    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    st.pyplot(fig)
