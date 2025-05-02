import pandas as pd
from io import StringIO
import gzip

def parse_annotation_file(annot_file):
    try:
        with gzip.open(annot_file, "rt", encoding="utf-8") as f:
            lines = f.readlines()
        df = pd.read_csv(StringIO("".join(lines)), sep="\t", low_memory=False)

        if "ID" in df.columns and "Gene Symbol" in df.columns:
            mapping = dict(zip(df["ID"], df["Gene Symbol"]))
            print(f"[Annotation] Mapped {len(mapping)} probe IDs to gene symbols.")
            return mapping
        else:
            print("[Annotation] File found but 'ID' or 'Gene Symbol' column is missing.")
    except Exception as e:
        print(f"[Annotation Error] Failed to parse .annot.gz: {e}")
    return {}

def extract_expression_and_symbols(file_lines, symbol_map=None):
    start_idx = file_lines.index("!series_matrix_table_begin") + 1
    end_idx = file_lines.index("!series_matrix_table_end")
    matrix_lines = file_lines[start_idx:end_idx]

    df = pd.read_csv(StringIO("\n".join(matrix_lines)), sep="\t")
    df = df.set_index("ID_REF" if "ID_REF" in df.columns else df.columns[0])

    if symbol_map:
        df.index = df.index.map(lambda x: symbol_map.get(x, x))
        df = df[~df.index.isna()]
        df = df.loc[~df.index.duplicated(keep='first')]

    return df.T  # samples = rows, genes = columns

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

    # Assign binary/multi-class labels
    labels = []
    if "stress" in metadata_df.columns:
        for val in metadata_df["stress"].str.lower():
            if "nodvt" in val:
                labels.append(0)
            elif "dvt" in val:
                labels.append(1)
            else:
                labels.append(-1)
    else:
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

def load_geo_series_matrix(file, annot_file=None):
    content = file.read()
    if isinstance(content, bytes):
        content = content.decode("utf-8")
    lines = content.splitlines()

    symbol_map = {}
    if annot_file is not None:
        symbol_map = parse_annotation_file(annot_file)
        if not symbol_map:
            print("⚠️ Annotation file loaded but mapping failed.")
    else:
        print("📎 No annotation file provided. Using raw probe IDs.")

    expression_df = extract_expression_and_symbols(lines, symbol_map)
    labels, metadata = extract_labels_and_metadata(lines)

    return expression_df, labels, metadata
