import pandas as pd

def map_human_to_model_genes(
    human_genes,
    animal_df,
    ortholog_path='data/mouse_to_human_orthologs.csv',
    model_species='Mouse'
):
    """
    Maps human gene symbols to orthologs in a given model species using a wide-format ortholog table.

    Args:
        human_genes (list): Gene symbols from human dataset (columns of human X).
        animal_df (pd.DataFrame): Animal model dataset with gene symbol columns.
        ortholog_path (str): Path to the ortholog table CSV file.
        model_species (str): Species to map to (e.g., 'Mouse', 'Rat', 'Zebrafish').

    Returns:
        shared_genes (list): List of shared ortholog symbols found in animal dataset.
        X_animal_aligned (pd.DataFrame): Aligned animal dataframe subset with shared genes.
    """
    species_col = f"{model_species}_Symbol"

    try:
        ortho_df = pd.read_csv(ortholog_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Ortholog file not found at: {ortholog_path}")
    
    if 'Human_Symbol' not in ortho_df.columns or species_col not in ortho_df.columns:
        raise ValueError(f"Expected columns 'Human_Symbol' and '{species_col}' in ortholog file.")

    # Drop rows missing data
    ortho_df = ortho_df.dropna(subset=['Human_Symbol', species_col])

    # Build mapping dictionary
    human_to_model = dict(zip(ortho_df['Human_Symbol'], ortho_df[species_col]))

    # Map human genes to model genes
    mapped_genes = [human_to_model.get(g) for g in human_genes if g in human_to_model]
    shared_genes = [g for g in mapped_genes if g in animal_df.columns]

    if not shared_genes:
        raise ValueError(f"No shared ortholog genes found for species '{model_species}'.")

    # Subset and align animal data
    X_animal = animal_df[shared_genes].copy()

    return shared_genes, X_animal
