# src/model_training.py

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

def train_model(X, y):
    # If the dataset is too small to split reliably, train on all data
    if len(X) < 6:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        metrics = {
            "accuracy": accuracy_score(y, model.predict(X)),
            "note": "⚠️ Trained on full dataset due to small sample size (no train/test split)."
        }
        return model, metrics

    # Standard split if enough samples
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "report": classification_report(y_test, y_pred, output_dict=True)
    }

    return model, metrics
