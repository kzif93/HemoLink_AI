# src/explainability.py

import shap
import matplotlib.pyplot as plt

def generate_shap_plots(model, X):
    """
    Generate a SHAP summary (beeswarm) plot to explain model predictions.

    Parameters:
        model: Trained tree-based model (e.g., RandomForestClassifier)
        X (pd.DataFrame): Input data used for prediction

    Returns:
        matplotlib.figure.Figure: SHAP beeswarm plot figure
    """
    # Use TreeExplainer for RandomForest or similar models
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Generate beeswarm plot for class 1 (if binary classification)
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values[1], X, plot_type="dot", show=False)
    return fig
