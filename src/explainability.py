import shap
import matplotlib.pyplot as plt
import streamlit as st

def shap_summary_plot(model, X, max_display=20):
    """
    Displays a SHAP summary bar plot for feature importance.
    """
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        # For binary classification models
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # focus on class 1

        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, X, plot_type="bar", max_display=max_display, show=False)
        st.pyplot(fig)

    except Exception as e:
        st.warning(f"⚠️ Could not generate SHAP plot: {e}")
