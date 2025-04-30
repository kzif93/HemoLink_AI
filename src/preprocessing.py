import pandas as pd
from sklearn.preprocessing import StandardScaler

def clean_and_scale(X):
    # Fill missing values if any
    X_clean = X.fillna(0)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    return pd.DataFrame(X_scaled, columns=X.columns)
