# src/ortholog_mapper.py

import pandas as pd

def map_orthologs(mouse_df, human_df, ortholog_df):
    """
    Aligns mouse and human expression data using an ortholog mapping.

    Parameters:
        mouse_df (pd.DataFrame): Mouse expression matrix with gene symbols as columns
        human_df (pd.DataFrame): Human expression matrix with gene symbols as columns
        ortholog_df (pd.DataFrame): DataFrame with columns ['mouse_symbol', 'human_symbol']

    Returns:
        (pd.DataFrame, pd.DataFrame): Aligned mouse and human DataFrames
    """
    # Keep only orthologs present in both datasets
    valid_orthologs = ortholog_df[
        (ortholog_df["mouse_symbol"].isin(mouse_df.columns)) &
        (ortholog_df["human_symbol"].isin(human_df.columns))
    ]

    # Subset and reorder both datasets
    mouse_genes = valid_orthologs["mouse_symbol"].values
    human_genes = valid_orthologs["human_symbol"].values

    mouse_aligned = mouse_df[mouse_genes]
    human_aligned = human_df[human_genes]

    # Rename to shared names for downstream model compatibility
    mouse_aligned.columns = valid_orthologs["human_symbol"].values
    human_aligned.columns = valid_orthologs["human_symbol"].values

    return mouse_aligned, human_aligned
