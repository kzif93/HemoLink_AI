from Bio import Entrez
import GEOparse
import pandas as pd
import os

# Set your email to comply with NCBI Entrez policy
Entrez.email = "your.email@example.com"  # <-- replace with your real email

def search_geo_datasets(query, max_results=10):
    """
    Searches GEO DataSets (GDS) for matching studies based on a keyword.

    Args:
        query (str): Disease or keyword to search for.
        max_results (int): Max number of datasets to return.

    Returns:
        List of GSE IDs (e.g., ['GSE16561', 'GSE22255'])
    """
    handle = Entrez.esearch(db="gds", term=query, retmax=max_results)
    record = Entrez.read(handle)
    id_list = record["IdList"]

    gse_ids = []
    for gid in id_list:
        summary = Entrez.read(Entrez.esummary(db="gds", id=gid))
        title = summary[0]["title"]
        if "GSE" in title:
            gse_id = title.split(" ")[0]
            gse_ids.append(gse_id)

    return gse_ids


def fetch_geo_platform_matrix(gse_id, destdir="data"):
    """
    Downloads and returns the expression matrix from a GEO GSE dataset.

    Args:
        gse_id (str): GEO Series ID (e.g., GSE16561)
        destdir (str): Directory to store downloaded files.

    Returns:
        pd.DataFrame: Expression matrix (samples as columns, genes as rows)
    """
    gse = GEOparse.get_GEO(geo=gse_id, destdir=destdir, annotate_gpl=True)

    # Extract and pivot sample data into a matrix
    df = gse.pivot_samples("VALUE")
    df.index.name = "Gene"
    df = df.reset_index().dropna()

    # Return as a tidy DataFrame
    return df
