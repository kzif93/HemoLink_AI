import os
import pandas as pd

def load_multiple_datasets(gse_list):
    """
    Load expression and label data from multiple GSEs and concatenate.
    Returns:
        - merged_df: gene expression dataframe
        - merged_labels: corresponding binary labels
    """
    all_dfs = []
    all_labels = []

    for gse in gse_list:
        expr_path = os.path.join("data", f"{gse}_expression.csv")
        label_path = os.path.join("data", f"{gse}_labels.csv")

        if not os.path.exists(expr_path):
            print(f"❌ Missing expression file for {gse}")
            continue
        if not os.path.exists(label_path):
            print(f"❌ Missing label file for {gse}")
            continue

        try:
            df = pd.read_csv(expr_path, index_col=0)
            labels = pd.read_csv(label_path, index_col=0).squeeze("columns")

            if df.empty or labels.empty:
                print(f"⚠️ Empty data in {gse}")
                continue

            shared_samples = df.index.intersection(labels.index)
            df = df.loc[shared_samples]
            labels = labels.loc[shared_samples]

            all_dfs.append(df)
            all_labels.append(labels)
        except Exception as e:
            print(f"❌ Error loading {gse}: {e}")

    if all_dfs and all_labels:
        merged_df = pd.concat(all_dfs)
        merged_labels = pd.concat(all_labels)
        return merged_df, merged_labels
    else:
        return None
