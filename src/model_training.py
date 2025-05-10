import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.utils.multiclass import unique_labels
import streamlit as st

def train_model(X, y):
    try:
        st.write("🧬 Training feature matrix (X):", X.shape)
        st.write("🏷️ Labels (y):", y.shape)
        st.write("🔍 y type:", type(y))
        st.write("🔍 y unique values:", pd.Series(y).unique())

        # Flatten label to 1D NumPy array
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]
        y = y.values.ravel() if hasattr(y, 'values') else y
        y = y.astype(int)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)

        preds = model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, preds)

        # Ensure y and preds are 1D arrays
        y_true = y
        y_pred = (preds > 0.5).astype(int)
        labels = unique_labels(y_true, y_pred)

        report = classification_report(y_true, y_pred, labels=labels, output_dict=True)

        metrics = {
            "roc_auc": round(auc, 4),
            "classification_report": report,
        }
        return model, metrics

    except Exception as e:
        raise RuntimeError(f"Training failed: {e}")
