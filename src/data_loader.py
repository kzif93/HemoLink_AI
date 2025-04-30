import pandas as pd
from io import StringIO

def extract_expression_and_symbols(file_lines):
    start_idx = file_lines.index("!series_matrix_table_begin") + 1
    end_idx = file_lines.index("!series_matrix_table_end")
    matrix_lines = file_lines[start_idx:end_idx]

    # Read matrix
    df = pd.read_csv(StringIO("\n".join(matrix_lines)), sep="\t")

    # Try to map probe IDs to gene symbols if available
    if "ID_REF" in df.columns:
        df = df.set_index("ID_REF")
    else:
        df = df.set_index(df.columns[0])

    # Attempt gene symbol extraction from annotation if available
    gene_map = {}
    for line in file_lines:
        if line.startswith("!annotation_table_start"):
            break
    else:
        line = None

    # If annotation table exists, try to map probes to symbols
    if "!annotation_table_start" in file_lines:
        try:
            annot_start = file_lines.index("!annotation_table_start") + 1
            annot_end = file_lines.index("!annotation_table_end")
            annotation = pd.read_csv(StringIO("\n".join(file_lines[annot_start:annot_end])), sep="\t")
            if "ID" in annotation.columns and "Gene Symbol" in annotation.columns:
                symbol_map = dict(zip(annotation["ID"], annotation["Gene Symbol"]))
                df = df.rename(columns=lambda x: x.strip())
                df.index = df.index.map(lambda x: symbol_map.get(x, x))  # Map if exists, keep original if not
        except Exception as e:
            pass  # If anything goes wrong, fall back to default probe IDs

    # Transpose: samples as rows, genes as columns
    df = df.T
    return df

def extract_labels_and_metadata(file_lines):
    meta_lines = [l for l in file_lines if "characteristics_ch1" in l.lower()]
    labels = []
    parsed = {}

    for i, line in enumerate(meta_lines):
        parts = [val.strip().strip('"') for val in line.strip().split("\t")[1:]]
        split_values = pd.Series(parts).str.extract(r'(?i)([\w\s\-]+):\s*(.*)')
        keys = split_values[0].str.strip().str.lower()
        vals = split_values[1].str.strip()
        for key in keys.unique():
            if pd.notna(key):
                parsed.setdefault(key, []).append(vals[keys == key].values[0])

    metadata_df = pd.DataFrame(parsed)
    metadata_df.index.name = "Sample"

    # Assign label
    for i in range(metadata_df.shape[0]):
        row = metadata_df.iloc[i].astype(str).str.lower().str.strip()
        label = -1
        for val in row:
            if "control" in val:
                label = 0
            elif "vte" in val or "dvt" in val:
                label = 1
            elif "aps" in val:
                label = 2
        labels.append(label)

    return labels, metadata_df

def load_geo_series_matrix(file):
    content = file.read()
    if isinstance(content, bytes):
        content = content.decode("utf-8")
    lines = content.splitlines()

    expression_df = extract_expression_and_symbols(lines)
    labels, metadata = extract_labels_and_metadata(lines)

    return expression_df, labels, metadata
