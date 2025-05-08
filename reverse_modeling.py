import os
import pandas as pd

def list_animal_datasets(folder_path):
    """
    Lists all .csv files in the given folder that represent animal model expression datasets.
    """
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Animal model folder not found: {folder_path}")
    return [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith("_expression.csv")]

def load_multiple_datasets(file_paths):
    """
    Loads multiple expression datasets from file paths into a dictionary keyed by dataset name.
    """
    datasets = {}
    for path in file_paths:
        try:
            df = pd.read_csv(path, index_col=0)
            name = os.path.basename(path).replace("_expression.csv", "")
            datasets[name] = df
        except Exception as e:
            print(f"Failed to load {path}: {e}")
    return datasets
