import pandas as pd
from io import StringIO

def load_geo_series_matrix(file):
    content = file.read()
    if isinstance(content, bytes):
        content = content.decode("utf-8")
    lines = content.splitlines()

    # Detect matrix section
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

    # Look for sample annotations
    annotations = {
        "characteristics_ch1": [],
        "description": [],
        "title": [],
        "source_name_ch1": []
    }

    for line in lines:
        for key in annotations:
            if line.lower().startswith(f"!sample_{key}"):
                annotations[key].append(line.strip().split("\t")[1:])

    # Assign default -1 labels
    n_samples = data.shape[0]
    labels = [-1] * n_samples

    # Try to find condition in any metadata field
    for source in annotations.values():
        if not source:
            continue
        for i, val in enumerate(source[0]):
            val_lower = val.lower()
            if "control" in val_lower:
                labels[i] = 0
            elif "vte" in val_lower or "dvt" in val_lower or "thrombo" in val_lower:
                labels[i] = 1
            elif "aps" in val_lower:
                labels[i] = 2

    # Extract metadata for stratification
    metadata_dict = {}
    for line in lines:
        if "characteristics_ch1" not in line.lower():
            continue
        parts = line.strip().split("\t")
        key = parts[0].replace("!Sample_characteristics_ch1", "").strip().lower()
        if not key:
            continue
        metadata_dict[key] = parts[1:]

    metadata = pd.DataFrame(metadata_dict)
    return data, labels, metadata
