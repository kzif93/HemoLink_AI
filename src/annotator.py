import pandas as pd
import gzip

def load_annotation_file(annot_file):
    try:
        if annot_file.name.endswith(".gz"):
            with gzip.open(annot_file, "rt", encoding="utf-8", errors="ignore") as f:
                df = pd.read_csv(f, sep="\t", comment="#", low_memory=False)
        else:
            df = pd.read_csv(annot_file, sep="\t", comment="#", low_memory=False)

        candidates = [col for col in df.columns if "ID" in col.upper()]
        if not candidates:
            raise ValueError("No ID_REF or probe ID column found in annotation file.")

        id_col = [col for col in df.columns if "ID" in col.upper()][0]
        symbol_col = next((c for c in df.columns if "GENE SYMBOL" in c.upper() or "GENE" in c.upper()), None)

        if symbol_col is None:
            raise ValueError("No 'Gene Symbol' column found in annotation file.")

        annot_map = df[[id_col, symbol_col]].dropna().drop_duplicates()
        annot_map.columns = ["probe_id", "gene_symbol"]
        annot_map["probe_id"] = annot_map["probe_id"].astype(str).str.upper()
        annot_map["gene_symbol"] = annot_map["gene_symbol"].astype(str).str.upper()
        return dict(zip(annot_map["probe_id"], annot_map["gene_symbol"]))

    except Exception as e:
        raise RuntimeError(f"Failed to parse annotation file: {e}")

def annotate_expression_matrix(expr_df, annotation_map):
    expr_df.columns = expr_df.columns.astype(str).str.upper()
    expr_df = expr_df.rename(columns=annotation_map)
    expr_df = expr_df.loc[:, expr_df.columns.notna()]
    expr_df = expr_df.loc[:, ~expr_df.columns.duplicated()]
    return expr_df
