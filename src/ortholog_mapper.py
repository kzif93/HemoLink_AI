# src/ortholog_mapper.py

import pandas as pd

def map_orthologs(mouse_df, human_df, ortholog_df):
    shared_mouse = list(set(mouse_df.columns).intersection(set(ortholog_df["mouse_symbol"])))
    shared_human = list(set(human_df.columns).intersection(set(ortholog_df["human_symbol"])))

    if not shared_mouse or not shared_human:
        raise ValueError("❌ No shared orthologs found between input data and mapping file.")

    # Filter orthologs to shared ones only
    valid_orthologs = ortholog_df[
        ortholog_df["mouse_symbol"].isin(shared_mouse) &
        ortholog_df["human_symbol"].isin(shared_human)
    ].drop_duplicates(subset="human_symbol")

    # Ensure the genes exist in both datasets
    mouse_genes = valid_orthologs["mouse_symbol"].unique()
    human_genes = valid_orthologs["human_symbol"].unique()

    mouse_aligned = mouse_df[mouse_genes]
    human_aligned = human_df[human_genes]

    # Reindex to match order
    valid_orthologs = valid_orthologs.set_index("human_symbol")
    valid_orthologs = valid_orthologs.loc[human_aligned.columns]
    mouse_aligned = mouse_aligned[valid_orthologs["mouse_symbol"].values]

    human_aligned.columns = valid_orthologs.index
    mouse_aligned.columns = valid_orthologs.index

    return mouse_aligned, human_aligned
