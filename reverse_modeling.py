import os
import pandas as pd

def list_animal_datasets(folder_path="animal_models"):
    """List available pre-downloaded animal expression datasets."""
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Animal model folder not found: {folder_path}")
    files = [f for f in os.listdir(folder_path) if f.endswith("_expression.csv")]
    return [f.replace("_expression.csv", "") for f in files]

def load_multiple_datasets(gse_list):
    """
    Load and merge multiple GEO datasets with label alignment.

    Returns:
        X: Feature matrix
        y: Label series
    """
    all_dfs = []

    for gse in gse_list:
        expr_path = f"data/{gse}_expression.csv"
        label_path = f"data/{gse}_labels.csv"

        if not os.path.exists(expr_path) or not os.path.exists(label_path):
            print(f"[WARN] Missing files for {gse}, skipping.")
            continue

        expr_df = pd.read_csv(expr_path, index_col=0)
        labels_df = pd.read_csv(label_path, index_col=0)

        if expr_df.shape[0] > expr_df.shape[1]:
            expr_df = expr_df.T

        if "label" not in labels_df.columns:
            labels_df.columns = ["label"]

        labels = labels_df["label"]

        # Align samples
        common_samples = expr_df.index.intersection(labels.index)
        if len(common_samples) < 2:
            print(f"[WARN] Not enough overlapping samples in {gse}, skipping.")
            continue

        expr_df = expr_df.loc[common_samples]
        labels = labels.loc[common_samples]

        expr_df["label"] = labels
        all_dfs.append(expr_df)

    if not all_dfs:
        return None, None

    merged_df = pd.concat(all_dfs, axis=0)
    if "label" not in merged_df.columns:
        raise ValueError("Label column not found after merging datasets.")

    X = merged_df.drop(columns=["label"])
    y = merged_df["label"]

    return X, y
