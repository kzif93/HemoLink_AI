import pandas as pd
from sklearn.feature_selection import VarianceThreshold

def reduce_low_variance_features(X, threshold=0.01):
    selector = VarianceThreshold(threshold)
    X_reduced = selector.fit_transform(X)
    selected_columns = X.columns[selector.get_support()]
    return pd.DataFrame(X_reduced, columns=selected_columns, index=X.index)
