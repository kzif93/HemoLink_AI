import pandas as pd

def predict(model, X):
    """
    Returns a DataFrame with prediction probabilities and predicted classes.
    """
    proba = model.predict_proba(X)
    preds = model.predict(X)

    df = pd.DataFrame({
        "Prediction": preds,
        "Probability_0": proba[:, 0],
        "Probability_1": proba[:, 1]
    }, index=X.index)

    return df
