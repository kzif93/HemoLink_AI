import pandas as pd
from io import StringIO

def load_geo_series_matrix(file):
    """
    Parses a GEO series_matrix.txt file or CSV to extract:
    - Expression data (samples as rows, genes as columns)
    - Multiclass labels (e.g. 0=Control, 1=VTE, 2=APS)
    - Metadata for stratification (age, sex, etc.)
    """

    # Read file content
    content = file.read()
    if isinstance(content, bytes):
        content = content.decode("utf-8")
    lines = content.splitlines()

    # Try to detect GEO matrix structure
    try:
        start_idx = lines.index("!series_matrix_table_begin") + 1
        end_idx = lines.index("!series_matrix_table_end")
        matrix_lines = lines[start_idx:end_idx]

        # Load without assuming header row
        df = pd.read_csv(StringIO("\n".join(matrix_lines)), sep="\t", header=None)

        # Set gene names as first column
        df.columns = ["ID_REF"] + [f"Sample_{i}" for i in range(1, df.shape[1])]

    except ValueError:
        # If no matrix markers found, treat it as regular CSV
        file.seek(0)
        df = pd.read_csv(file)
        return df, [0] * df.shape[0], pd.DataFrame()

    # Drop gene ID column and transpose so samples = rows
    data = df.drop(columns=["ID_REF"], errors="ignore").T
    data.columns = df["ID_REF"].values

    # Initialize default labels
    labels = [-1] * data.shape[0]

    # Parse metadata from characteristics_ch1
    metadata_lines = [line for line in lines if "characteristics_ch1" in line.lower()]
    metadata_dict = {}

    for line in metadata_lines:
        parts = line.strip().split("\t")
        key = parts[0].replace("!Sample_characteristics_ch1", "").strip().lower()
        values = parts[1:]

        if not key:
            key = "condition"

        metadata_dict[key] = values

        # If the line contains disease conditions, assign multiclass labels
        if "condition" in key:
            labels = []
            for val in values:
                val = val.lower()
                if "control" in val:
                    labels.append(0)
                elif "vte" in val:
                    labels.append(1)
                elif "aps" in val:
                    labels.append(2)
                else:
                    labels.append(-1)

    # Convert metadata dict to DataFrame
    metadata = pd.DataFrame(metadata_dict)

    return data, labels, metadata
