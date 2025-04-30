import pandas as pd
from io import StringIO

def load_geo_series_matrix(file):
    content = file.read()
    if isinstance(content, bytes):
        content = content.decode("utf-8")
    lines = content.splitlines()

    # Load matrix
    try:
        start_idx = lines.index("!series_matrix_table_begin") + 1
        end_idx = lines.index("!series_matrix_table_end")
        matrix_lines = lines[start_idx:end_idx]
        df = pd.read_csv(StringIO("\n".join(matrix_lines)), sep="\t", header=None)
        df.columns = ["ID_REF"] + [f"Sample_{i}" for i in range(1, df.shape[1])]
        data = df.set_index("ID_REF").T
        data = data.apply(pd.to_numeric, errors="coerce").fillna(0)
    except ValueError:
        file.seek(0)
        df = pd.read_csv(file)
        return df, [0] * df.shape[0], pd.DataFrame()

    # Parse full sample metadata row-wise
    n_samples = data.shape[0]
    sample_ids = data.index.tolist()
    metadata_rows = []

    for line in lines:
        if "characteristics_ch1" in line.lower():
            parts = line.strip().split("\t")[1:]
            metadata_rows.append(parts)

    # Transpose to [n_samples x fields]
    metadata_df = pd.DataFrame(metadata_rows).T
    metadata_df.columns = [f"field_{i}" for i in range(metadata_df.shape[1])]
    metadata_df.index = sample_ids

    # Extract labels from any metadata field
    labels = [-1] * n_samples
    for col in metadata_df.columns:
        for i, val in enumerate(metadata_df[col]):
            val = val.lower()
            if "control" in val:
                labels[i] = 0
            elif "vte" in val or "dvt" in val:
                labels[i] = 1
            elif "aps" in val:
                labels[i] = 2

    # Optional: parse fields by name
    parsed = {}
    for col in metadata_df.columns:
        values = metadata_df[col].str.extract(r"(\w+):\s*(.*)")
        if values.shape[1] == 2:
            keys = values[0].str.lower()
            vals = values[1]
            for key in keys.unique():
                parsed[key] = vals[keys == key].reset_index(drop=True)

    metadata_parsed = pd.DataFrame(parsed)
    metadata_parsed.index = sample_ids

    return data, labels, metadata_parsed
