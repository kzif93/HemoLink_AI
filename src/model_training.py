import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
import streamlit as st


def train_model(X, y):
    """
    Train a basic RandomForest model and return model and evaluation metrics.
    """
    try:
        # Show preview of features and labels
        st.write("🧬 Training feature matrix (X):", X.shape)
        st.write("🏷️ Labels (y):", y.shape)
        st.write("🔍 y type:", type(y))
        st.write("🔍 y unique values:", y.unique())

        # Ensure y is a flat vector
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]
        y = y.astype(int)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        preds = model.predict_proba(X)[:, 1]

        auc = roc_auc_score(y, preds)
        report = classification_report(y, (preds > 0.5).astype(int), output_dict=True)

        metrics = {
            "roc_auc": round(auc, 4),
            "classification_report": report,
        }
        return model, metrics

    except Exception as e:
        raise RuntimeError(f"Training failed: {e}")
