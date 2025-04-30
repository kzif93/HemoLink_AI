import pandas as pd
from gprofiler import GProfiler

def map_mouse_to_human_genes(mouse_genes):
    gp = GProfiler(return_dataframe=True)
    res = gp.convert(organism="mmusculus", query=mouse_genes, target="hsapiens")
    mapped = res[['incoming', 'name']].dropna().drop_duplicates()
    return mapped.set_index("incoming")["name"].to_dict()

def align_cross_species_data(mouse_df, human_df):
    mapping = map_mouse_to_human_genes(mouse_df.columns.tolist())
    renamed_mouse = mouse_df.rename(columns=mapping)
    shared_genes = list(set(renamed_mouse.columns) & set(human_df.columns))

    mouse_filtered = renamed_mouse[shared_genes]
    human_filtered = human_df[shared_genes]

    return mouse_filtered, human_filtered, shared_genes
