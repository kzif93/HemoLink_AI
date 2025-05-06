import pandas as pd

def preprocess_dataset(df, label_column):
    """
    Preprocesses a transcriptomic dataset by:
    - Extracting numeric gene expression features
    - Dropping NA values
    - Separating label column

    Args:
        df (pd.DataFrame): Input human transcriptomic dataset
        label_column (str): Name of the binary target column

    Returns:
        X (pd.DataFrame): Preprocessed features
        y (pd.Series): Binary labels
    """
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found in dataset.")

    df = df.dropna(axis=1, how='any')  # drop any columns with missing values

    # Separate labels
    y = df[label_column]
    X = df.drop(columns=[label_column])

    # Keep only numeric gene expression features
    X = X.select_dtypes(include=["float64", "int64"])

    if X.empty:
        raise ValueError("No numeric features found after preprocessing.")

    return X, y
