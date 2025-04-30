import pandas as pd
import mygene

def map_mouse_to_human_genes(mouse_genes):
    mg = mygene.MyGeneInfo()
    query = mg.querymany(mouse_genes, scopes='symbol', fields='homologene', species='mouse')

    mapping = {}
    for entry in query:
        if 'homologene' in entry and isinstance(entry['homologene'], dict):
            human_homologs = entry['homologene'].get('genes', [])
            for sp_id, gene_symbol in human_homologs:
                if sp_id == 9606:  # human NCBI tax ID
                    mapping[entry['query']] = gene_symbol
                    break
    return mapping

def align_cross_species_data(mouse_df, human_df):
    mapping = map_mouse_to_human_genes(mouse_df.columns.tolist())
    renamed_mouse = mouse_df.rename(columns=mapping)
    shared_genes = list(set(renamed_mouse.columns) & set(human_df.columns))

    mouse_filtered = renamed_mouse[shared_genes]
    human_filtered = human_df[shared_genes]

    return mouse_filtered, human_filtered, shared_genes
