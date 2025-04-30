import shap
import streamlit as st
import matplotlib.pyplot as plt

def show_shap_summary(model, X_test):
    st.subheader("🔍 SHAP Summary")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    if isinstance(shap_values, list) and len(shap_values) > 1:
        shap_values = shap_values[1]
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, X_test, plot_type="dot", max_display=15, show=False)
    st.pyplot(fig)

def show_shap_comparison(model, subgroup_dict):
    st.subheader("🧬 Side-by-Side SHAP Comparison")
    explainer = shap.TreeExplainer(model)
    cols = st.columns(2)
    for idx, (label, X_subgroup) in enumerate(subgroup_dict.items()):
        shap_values = explainer.shap_values(X_subgroup)
        if isinstance(shap_values, list) and len(shap_values) > 1:
            shap_values = shap_values[1]
        with cols[idx]:
            st.markdown(f"**{label}**")
            fig, ax = plt.subplots()
            shap.summary_plot(shap_values, X_subgroup, plot_type="dot", max_display=15, show=False)
            st.pyplot(fig)
