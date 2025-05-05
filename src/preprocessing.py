# src/preprocessing.py

import pandas as pd
from sklearn.preprocessing import StandardScaler

def clean_and_scale(X):
    """
    Fill missing values with 0 and apply standard scaling.
    Ensures the output retains original column and index structure.
    """
    X_clean = X.fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)
    return pd.DataFrame(X_scaled, columns=X_clean.columns, index=X_clean.index)
