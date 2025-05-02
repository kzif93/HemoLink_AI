import pandas as pd

def load_ortholog_map(csv_path="data/mouse_to_human_orthologs.csv"):
    try:
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=["mouse_symbol", "human_symbol"])
        df["mouse_symbol"] = df["mouse_symbol"].astype(str).str.upper()
        df["human_symbol"] = df["human_symbol"].astype(str).str.upper()
        return dict(zip(df["mouse_symbol"], df["human_symbol"]))
    except Exception as e:
        print(f"[Ortholog Error] Failed to load ortholog map: {e}")
        return {}

def convert_mouse_genes_to_human(df, ortholog_map):
    df.columns = df.columns.astype(str).str.upper()
    df.columns = df.columns.map(lambda g: ortholog_map.get(g, None))
    df = df.loc[:, df.columns.notna()]
    df = df.loc[:, ~df.columns.duplicated()]
    return df
