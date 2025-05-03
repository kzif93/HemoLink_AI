# src/explainability.py

import shap
import matplotlib.pyplot as plt

def generate_shap_plots(model, X):
    # Create SHAP explainer
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)

    # Determine SHAP value format (multiclass vs single-output)
    if isinstance(shap_values, list):
        # Binary/multiclass classifier: use class 1 if available
        if len(shap_values) > 1:
            shap_matrix = shap_values[1].values
        else:
            shap_matrix = shap_values[0].values
    else:
        # Regression or single-output model
        shap_matrix = shap_values.values

    # Plot summary
    fig = plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_matrix, X, plot_type="dot", show=False)
    return fig
