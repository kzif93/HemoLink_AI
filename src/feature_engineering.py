from sklearn.feature_selection import VarianceThreshold
import pandas as pd

def reduce_features(X, threshold=0.0):
    """
    Removes features with low variance (default threshold = 0.0 keeps only features with variance > 0).
    """
    selector = VarianceThreshold(threshold=threshold)
    X_reduced = selector.fit_transform(X)

    selected_columns = X.columns[selector.get_support()]
    return pd.DataFrame(X_reduced, columns=selected_columns, index=X.index)
