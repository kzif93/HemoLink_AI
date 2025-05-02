import pandas as pd
from src.ortholog_mapper import load_ortholog_map, convert_mouse_genes_to_human

def align_cross_species_data(mouse_df, human_df):
    ortholog_map = load_ortholog_map()
    mouse_df = convert_mouse_genes_to_human(mouse_df, ortholog_map)
    human_df.columns = human_df.columns.astype(str).str.upper()

    shared_genes = list(set(mouse_df.columns) & set(human_df.columns))
    mouse_df = mouse_df[shared_genes]
    human_df = human_df[shared_genes]

    return mouse_df, human_df, shared_genes
