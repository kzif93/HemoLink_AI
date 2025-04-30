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

    sample_ids = data.index.tolist()
    metadata_rows = []

    for line in lines:
        if "characteristics_ch1" in line.lower():
            parts = [val.strip().strip('"') for val in line.strip().split("\t")[1:]]
            metadata_rows.append(parts)

    metadata_df = pd.DataFrame(metadata_rows).T
    metadata_df.columns = [f"field_{i}" for i in range(metadata_df.shape[1])]
    metadata_df.index = sample_ids

    # Parse key-value pairs from metadata
    parsed = {}
    for col in metadata_df.columns:
        split_values = metadata_df[col].str.extract(r'(?i)([\w\s\-]+):\s*(.*)')
        keys = split_values[0].str.strip().str.lower()
        vals = split_values[1].str.strip()
        for key in keys.unique():
            if pd.notna(key):
                parsed[key] = vals[keys == key].reset_index(drop=True)

    metadata_parsed = pd.DataFrame(parsed)
    metadata_parsed.index = sample_ids

    # Assign labels
    labels = []
    for i in range(len(metadata_parsed)):
        row = metadata_parsed.iloc[i].astype(str).str.lower().str.strip()
        label = -1
        for val in row:
            if "control" in val:
                label = 0
            elif "vte" in val or "dvt" in val:
                label = 1
            elif "aps" in val:
                label = 2
        labels.append(label)

    return data, labels, metadata_parsed
