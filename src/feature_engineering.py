from sklearn.feature_selection import VarianceThreshold
import pandas as pd

def reduce_low_variance_features(X, threshold=0.01):
    selector = VarianceThreshold(threshold)
    X_reduced = selector.fit_transform(X)
    selected_cols = X.columns[selector.get_support()]
    return pd.DataFrame(X_reduced, columns=selected_cols)
