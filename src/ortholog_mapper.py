import pandas as pd
import os
import re

def map_human_to_model_genes(
    human_genes,
    animal_df,
    ortholog_path='data/mouse_to_human_orthologs.csv',
    filename_hint=None
):
    """
    Maps human gene symbols to orthologs in an animal model using a wide-format ortholog table.
    Automatically detects species from filename if provided.

    Args:
        human_genes (list): Gene symbols from human dataset.
        animal_df (pd.DataFrame): Animal model dataset.
        ortholog_path (str): Path to the ortholog table CSV.
        filename_hint (str): Optional filename or path from which to auto-detect species (e.g. "GSE233813_Mouse.csv").

    Returns:
        shared_genes (list): Common ortholog genes.
        X_animal (pd.DataFrame): Subset of animal_df aligned to shared genes.
    """
    # Try to auto-detect species from filename
    model_species = "Mouse"  # default fallback
    if filename_hint:
        match = re.search(r'_(Mouse|Rat|Zebrafish|Pig|Dog|Rabbit|Monkey)', filename_hint, re.IGNORECASE)
        if match:
            model_species = match.group(1).capitalize()
        else:
            raise ValueError("Could not detect model species from filename. Use underscore format like _Mouse, _Rat, etc.")

    species_col = f"{model_species}_Symbol"

    try:
        ortho_df = pd.read_csv(ortholog_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Ortholog file not found at: {ortholog_path}")

    if 'Human_Symbol' not in ortho_df.columns or species_col not in ortho_df.columns:
        raise ValueError(f"Ortholog file must contain 'Human_Symbol' and '{species_col}' columns.")

    ortho_df = ortho_df.dropna(subset=['Human_Symbol', species_col])
    human_to_model = dict(zip(ortho_df['Human_Symbol'], ortho_df[species_col]))

    mapped_genes = [human_to_model.get(g) for g in human_genes if g in human_to_model]
    shared_genes = [g for g in mapped_genes if g in animal_df.columns]

    if not shared_genes:
        raise ValueError(f"No shared ortholog genes found for detected species '{model_species}'.")

    X_animal = animal_df[shared_genes].copy()

    return shared_genes, X_animal
