from Bio import Entrez
import GEOparse
import pandas as pd
import os

# Required by NCBI for usage tracking — replace with your actual email
Entrez.email = "kzif93@gmail.com"


def search_geo_datasets(query, max_results=10):
    """
    Searches GEO Series (GSE) database for relevant expression datasets by keyword.

    Args:
        query (str): Keyword such as 'stroke', 'thrombosis', etc.
        max_results (int): Max number of GSE datasets to return.

    Returns:
        List of GSE IDs like ['GSE16561', 'GSE22255']
    """
    term = f"{query} AND Homo sapiens[Organism] AND expression profiling by array[DataSet Type]"
    handle = Entrez.esearch(db="gse", term=term, retmax=max_results)
    record = Entrez.read(handle)
    gse_ids = ["GSE" + geo_id for geo_id in record["IdList"]]
    return gse_ids


def fetch_geo_platform_matrix(gse_id, destdir="data"):
    """
    Downloads and returns the expression matrix from a GEO GSE dataset.

    Args:
        gse_id (str): GEO Series ID (e.g., GSE16561)
        destdir (str): Directory to store downloaded files.

    Returns:
        pd.DataFrame: Expression matrix with genes as rows, samples as columns.
    """
    gse = GEOparse.get_GEO(geo=gse_id, destdir=destdir, annotate_gpl=True)

    # Extract and pivot sample data into a gene x sample matrix
    df = gse.pivot_samples("VALUE")
    df.index.name = "Gene"
    df = df.reset_index().dropna()

    return df
