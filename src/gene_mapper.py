import pandas as pd
import mygene

def map_mouse_to_human_genes(mouse_genes):
    mg = mygene.MyGeneInfo()
    query = mg.querymany(mouse_genes, scopes='symbol', fields='homologene', species='mouse')

    mapping = {}
    for entry in query:
        if 'homologene' in entry and isinstance(entry['homologene'], dict):
            for species_id, gene_symbol in entry['homologene'].get('genes', []):
                if species_id == 9606:  # Human NCBI Taxonomy ID
                    mapping[entry['query']] = gene_symbol
                    break
    return mapping

def align_cross_species_data(mouse_df, human_df):
    # Map mouse gene names to human
    mapping = map_mouse_to_human_genes(mouse_df.columns.tolist())

    # Rename mouse genes using human homologs
    renamed_mouse = mouse_df.rename(columns=mapping)

    # Drop unmapped (NaN) columns
    renamed_mouse = renamed_mouse.loc[:, renamed_mouse.columns.notna()]
    human_df = human_df.loc[:, human_df.columns.notna()]

    # Keep only genes shared between mouse and human after mapping
    shared_genes = list(set(renamed_mouse.columns) & set(human_df.columns))
    shared_genes = [gene for gene in shared_genes if isinstance(gene, str) and gene.strip() != ""]

    # Final aligned dataframes
    mouse_filtered = renamed_mouse[shared_genes]
    human_filtered = human_df[shared_genes]

    return mouse_filtered, human_filtered, shared_genes
