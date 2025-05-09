import os
import pandas as pd


def list_animal_datasets(folder_path):
    """
    List available animal expression datasets in a folder.
    """
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Animal model folder not found: {folder_path}")

    files = [f for f in os.listdir(folder_path) if f.endswith("_expression.csv")]
    return [os.path.splitext(f)[0].replace("_expression", "") for f in files]


def load_multiple_datasets(gse_list, data_dir="data"):
    """
    Load and combine expression data and labels from multiple datasets.
    """
    all_data = []
    all_labels = []

    for gse in gse_list:
        exp_path = os.path.join(data_dir, f"{gse}_expression.csv")
        label_path = os.path.join(data_dir, f"{gse}_labels.csv")

        if not os.path.exists(exp_path) or not os.path.exists(label_path):
            raise FileNotFoundError(f"Missing files for {gse}: {exp_path} or {label_path}")

        df = pd.read_csv(exp_path, index_col=0)
        labels_df = pd.read_csv(label_path, index_col=0)
        label_col = labels_df.columns[0]  # Assume the first column is the label
        labels = labels_df[label_col]

        df = df.loc[:, df.columns.intersection(labels.index)]
        labels = labels.loc[df.columns]

        all_data.append(df)
        all_labels.append(labels)

    X = pd.concat(all_data, axis=1).T  # Combine samples
    y = pd.concat(all_labels, axis=0)

    return X, y
