import pandas as pd
import shap
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier

def apply_variance_threshold(X, threshold=0.01):
    selector = VarianceThreshold(threshold)
    X_reduced = selector.fit_transform(X)
    kept_columns = X.columns[selector.get_support()]
    return pd.DataFrame(X_reduced, columns=kept_columns, index=X.index)

def select_top_shap_features(X, y, top_n=100):
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap_importance = pd.DataFrame(
        abs(shap_values).mean(axis=0),
        index=X.columns,
        columns=["shap_importance"]
    )
    top_features = shap_importance.sort_values(by="shap_importance", ascending=False).head(top_n).index
    return X[top_features]
