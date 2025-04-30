import pandas as pd
from io import StringIO

def load_geo_series_matrix(file):
    """
    Parses a GEO series_matrix.txt file or CSV to extract:
    - Expression data (samples as rows, genes as columns)
    - Multiclass labels (0=control, 1=VTE, 2=APS)
    - Metadata (e.g. age, sex, etc.)
    """

    # Read content from uploaded file
    content = file.read()
    if isinstance(content, bytes):
        content = content.decode("utf-8")
    lines = content.splitlines()

    # Try to detect GEO matrix
    try:
        start_idx = lines.index("!series_matrix_table_begin") + 1
        end_idx = lines.index("!series_matrix_table_end")
        matrix_lines = lines[start_idx:end_idx]

        # Read as no-header, and manually define header
        df = pd.read_csv(StringIO("\n".join(matrix_lines)), sep="\t", header=None)
        df.columns = ["ID_REF"] + [f"Sample_{i}" for i in range(1, df.shape[1])]

        # Transpose: rows = samples, columns = genes
        data = df.set_index("ID_REF").T
        data = data.apply(pd.to_numeric, errors="coerce")
        data = data.fillna(0)

    except ValueError:
        # Fallback: assume user uploaded CSV
        file.seek(0)
        df = pd.read_csv(file)
        return df, [0] * df.shape[0], pd.DataFrame()

    # Extract metadata lines
    meta_lines = [l for l in lines if "characteristics_ch1" in l.lower()]
    metadata_dict = {}

    labels = [-1] * data.shape[0]  # default unknown labels

    for line in meta_lines:
        parts = line.strip().split("\t")
        key = parts[0].replace("!Sample_characteristics_ch1", "").strip().lower()
        values = parts[1:]

        if not key:
            key = "condition"

        metadata_dict[key] = values

        # Assign multiclass labels based on 'condition' key
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

    metadata = pd.DataFrame(metadata_dict)
    return data, labels, metadata
