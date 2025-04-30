import pandas as pd
import re
from io import StringIO

def load_geo_series_matrix(file):
    """
    Parses a GEO series_matrix.txt file or CSV to extract expression data,
    multiclass labels (e.g., Control, VTE, APS), and metadata (e.g., age, sex).
    """
    # Read file content
    content = file.read()
    if isinstance(content, bytes):
        content = content.decode("utf-8")
    lines = content.splitlines()

    # Identify start and end of the expression matrix
    try:
        start_idx = lines.index("!series_matrix_table_begin") + 1
        end_idx = lines.index("!series_matrix_table_end")
        matrix_lines = lines[start_idx:end_idx]
        df = pd.read_csv(StringIO("\n".join(matrix_lines)), sep="\t")
    except ValueError:
        # If markers not found, assume CSV format
        file.seek(0)
        df = pd.read_csv(file)
        labels = [0] * df.shape[0]
        metadata = pd.DataFrame()
        return df, labels, metadata

    # Extract sample metadata lines
    metadata_lines = [line for line in lines if line.startswith("!Sample_characteristics_ch1")]
    metadata_dict = {}
    for line in metadata_lines:
        parts = line.strip().split("\t")
        key = parts[0].replace("!Sample_characteristics_ch1", "").strip()
        values = parts[1:]
        metadata_dict[key] = values

    # Convert metadata_dict to DataFrame
    metadata = pd.DataFrame(metadata_dict)

    # Transpose expression data so samples are rows
    if "ID_REF" in df.columns:
        df = df.drop(columns=["ID_REF"])
    df = df.transpose()
    df.columns = df.iloc[0]
    df = df.drop(df.index[0])
    df.reset_index(drop=True, inplace=True)

    # Generate labels based on condition
    conditions = metadata.iloc[:, 0].str.lower()
    label_map = {"control": 0, "vte": 1, "aps": 2}
    labels = conditions.map(lambda x: next((label_map[key] for key in label_map if key in x), -1))
    labels = labels.fillna(-1).astype(int).tolist()

    return df, labels, metadata
