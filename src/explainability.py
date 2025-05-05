# src/explainability.py

import shap
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np

def generate_shap_plots(model, X, return_values=False):
    # Create SHAP explainer
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)

    # Handle multiclass vs single output
    if len(shap_values.shape) == 3:
        shap_matrix = shap_values[..., 1]
    else:
        shap_matrix = shap_values.values

    st.write("🧬 SHAP matrix shape:", shap_matrix.shape)
    st.write("📊 SHAP mean(abs):", np.abs(shap_matrix).mean())
    st.write("🧬 Feature matrix shape:", X.shape)

    if np.abs(shap_matrix).mean() < 1e-5:
        st.warning("⚠️ SHAP values are nearly zero. The plot may appear empty.")

    fig = plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_matrix, X, plot_type="dot", show=False, max_display=10)

    # Safely return gene names if DataFrame
    if return_values:
        if hasattr(X, "columns"):
            return fig, shap_matrix, X.columns.tolist()
        else:
            return fig, shap_matrix, []
    return fig
