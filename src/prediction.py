# src/prediction.py

import pandas as pd

def predict_on_human(model, X_human):
    """
    Predict outcomes on human expression data using a trained model.

    Parameters:
        model: Trained sklearn classifier
        X_human (pd.DataFrame): Preprocessed and aligned human expression data

    Returns:
        pd.DataFrame: Predictions as a DataFrame with binary output and probabilities
    """
    # Predict class labels and probabilities
    predicted_labels = model.predict(X_human)
    predicted_probs = model.predict_proba(X_human)[:, 1]

    return pd.DataFrame({
        "Prediction": predicted_labels,
        "Probability": predicted_probs
    }, index=X_human.index)
