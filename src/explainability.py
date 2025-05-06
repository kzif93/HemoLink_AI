import shap
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def extract_shap_values(model, X):
    """
    Extracts mean absolute SHAP values per feature using the appropriate SHAP explainer
    based on the model type (RandomForest, XGBoost, or LogisticRegression).
    
    Args:
        model: Trained model object.
        X: Input feature matrix (DataFrame).
    
    Returns:
        1D numpy array of mean absolute SHAP values per feature.
    """
    # Select appropriate SHAP explainer
    if isinstance(model, RandomForestClassifier):
        explainer = shap.TreeExplainer(model)
    elif isinstance(model, xgb.XGBClassifier):
        explainer = shap.Explainer(model, X, feature_perturbation="tree_path_dependent")
    elif isinstance(model, LogisticRegression):
        explainer = shap.LinearExplainer(model, X, feature_perturbation="interventional")
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")

    # Compute SHAP values
    shap_values = explainer.shap_values(X)

    # Handle binary classification SHAP format
    if isinstance(shap_values, list):
        # Use class 1 SHAP values (positive class)
        shap_array = np.abs(shap_values[1]).mean(axis=0)
    else:
        shap_array = np.abs(shap_values).mean(axis=0)

    return shap_array


def compare_shap_vectors(shap1, shap2):
    """
    Computes cosine similarity between two SHAP vectors.
    
    Args:
        shap1, shap2: 1D arrays of SHAP values.
    
    Returns:
        Cosine similarity float between 0 (dissimilar) and 1 (identical).
    """
    if len(shap1) != len(shap2):
        raise ValueError("SHAP vectors must have the same length for comparison.")

    dot = np.dot(shap1, shap2)
    norm1 = np.linalg.norm(shap1)
    norm2 = np.linalg.norm(shap2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(dot / (norm1 * norm2))
