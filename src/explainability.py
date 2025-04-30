import shap
import streamlit as st
import matplotlib.pyplot as plt

def show_shap_summary(model, X_test):
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    st.pyplot(bbox_inches='tight')
