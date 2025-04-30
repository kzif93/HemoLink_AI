import pandas as pd
from sklearn.preprocessing import StandardScaler

def clean_and_scale(X):
    X_clean = X.fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)
    return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
