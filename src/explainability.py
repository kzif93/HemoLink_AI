# src/explainability.py

import shap
import matplotlib.pyplot as plt

def generate_shap_plots(model, X):
    # Create SHAP explainer
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)

    # Determine SHAP value format
    if isinstance(shap_values, list):
        # Multi-class or binary
        if len(shap_values) > 1:
            shap_matrix = shap_values[1].values
        else:
            shap_matrix = shap_values[0].values
    else:
        # Regression or single-output
        shap_matrix = shap_values.values

    # Debug logs
    print("SHAP matrix shape:", shap_matrix.shape)
    print("SHAP mean abs:", abs(shap_matrix).mean())

    # Optional warning if all SHAP values are 0
    if abs(shap_matrix).mean() < 1e-5:
        print("⚠️ SHAP values are near zero. Plot may appear empty.")

    # Create plot
    fig = plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_matrix, X, plot_type="dot", show=False, max_display=10)
    return fig
