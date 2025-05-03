# src/ortholog_mapper.py

import pandas as pd

def map_orthologs(mouse_df, human_df, ortholog_df):
    """
    Aligns mouse and human expression data using an ortholog mapping.
    """
    mouse_genes = set(mouse_df.columns)
    human_genes = set(human_df.columns)

    valid_orthologs = ortholog_df[
        (ortholog_df["mouse_symbol"].isin(mouse_genes)) &
        (ortholog_df["human_symbol"].isin(human_genes))
    ]

    # 🧪 Debug output
    print("Total orthologs:", len(ortholog_df))
    print("Matched orthologs:", len(valid_orthologs))
    print("Matched mouse genes:", len(mouse_genes & set(ortholog_df['mouse_symbol'])))
    print("Matched human genes:", len(human_genes & set(ortholog_df['human_symbol'])))

    if valid_orthologs.empty:
        raise ValueError("❌ No shared orthologs found between input data and mapping file.")

    mouse_selected = valid_orthologs["mouse_symbol"].values
    human_selected = valid_orthologs["human_symbol"].values

    mouse_aligned = mouse_df[mouse_selected]
    human_aligned = human_df[human_selected]

    # Rename mouse columns to human symbols
    mouse_aligned.columns = valid_orthologs["human_symbol"].values
    human_aligned.columns = valid_orthologs["human_symbol"].values

    return mouse_aligned, human_aligned
