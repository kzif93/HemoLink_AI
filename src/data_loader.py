import pandas as pd
from io import StringIO

def extract_expression_and_symbols(file_lines):
    start_idx = file_lines.index("!series_matrix_table_begin") + 1
    end_idx = file_lines.index("!series_matrix_table_end")
    matrix_lines = file_lines[start_idx:end_idx]

    # Read matrix
    df = pd.read_csv(StringIO("\n".join(matrix_lines)), sep="\t")

    # Try to use probe ID as index
    if "ID_REF" in df.columns:
        df = df.set_index("ID_REF")
    else:
        df = df.set_index(df.columns[0])

    # Attempt gene symbol mapping (if annotation block exists)
    if "!annotation_table_start" in file_lines:
        try:
            annot_start = file_lines.index("!annotation_table_start") + 1
            annot_end = file_lines.index("!annotation_table_end")
            annotation = pd.read_csv(StringIO("\n".join(file_lines[annot_start:annot_end])), sep="\t")
            if "ID" in annotation.columns and "Gene Symbol" in annotation.columns:
                symbol_map = dict(zip(annotation["ID"], annotation["Gene Symbol"]))
                df.index = df.index.map(lambda x: symbol_map.get(x, x))
        except Exception:
            pass  # fallback if mapping fails

    # Transpose to samples as rows, genes as columns
    df = df.T
    return df


def extract_labels_and_metadata(file_lines):
    meta_lines = [l for l in file_lines if "characteristics_ch1" in l.lower()]
    sample_count = len(meta_lines[0].split("\t")) - 1 if meta_lines else 0
    parsed = {}

    for line in meta_lines:
        values = [val.strip().strip('"') for val in line.strip().split("\t")[1:]]
        pairs = pd.Series(values).str.extract(r'(?i)([\w\s\-]+):\s*(.*)')
        keys = pairs[0].str.strip().str.lower()
        vals = pairs[1].str.strip()
        for key in keys.unique():
            if pd.notna(key):
                parsed.setdefault(key, []).extend(vals[keys == key].tolist())

    metadata_df = pd.DataFrame(parsed)
    metadata_df.index.name = "Sample"

    # Assign labels based on presence of known keywords
    labels = []

    if "stress" in metadata_df.columns:
        # Special case: mouse file like GSE125965
        for val in metadata_df["stress"].str.lower():
            if "nodvt" in val:
                labels.append(0)
            elif "dvt" in val:
                labels.append(1)
            else:
                labels.append(-1)
    else:
        # General case: human dataset like GSE19151
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
