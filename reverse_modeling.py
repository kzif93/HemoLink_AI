import os
import sys

# Allow importing modules from the 'src/' folder
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def list_animal_datasets(folder_path="animal_models"):
    """
    Returns a list of .csv files in the given animal model folder.

    Args:
        folder_path (str): Path to folder containing animal model datasets.

    Returns:
        List[str]: Filenames ending in .csv
    """
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Animal model folder not found: {folder_path}")

    return [
        f for f in os.listdir(folder_path)
        if f.endswith(".csv") and os.path.isfile(os.path.join(folder_path, f))
    ]
