# src/enrichment.py

import requests
import pandas as pd
import streamlit as st

ENRICHR_URL = "https://maayanlab.cloud/Enrichr"

def enrich_genes(gene_list, library="GO_Biological_Process_2021", top_n=10):
    if not gene_list:
        st.warning("⚠️ No genes provided for enrichment.")
        return pd.DataFrame()

    # Submit gene list
    genes_str = "\n".join(gene_list)
    response = requests.post(f"{ENRICHR_URL}/addList", files={"list": (None, genes_str)})
    if not response.ok:
        st.error("❌ Failed to submit gene list to Enrichr.")
        return pd.DataFrame()

    user_list_id = response.json()["userListId"]

    # Query enrichment results
    enrich_url = f"{ENRICHR_URL}/enrich?userListId={user_list_id}&backgroundType={library}"
    enrich_response = requests.get(enrich_url)
    if not enrich_response.ok:
        st.error("❌ Failed to fetch enrichment results.")
        return pd.DataFrame()

    results = enrich_response.json()[library]
    df = pd.DataFrame(results, columns=[
        "Rank", "Term", "P-value", "Z-score", "Combined score", "Genes", "Adjusted P-value", "Old P-value", "Old Adjusted P-value"
    ])

    return df[["Term", "Adjusted P-value", "Combined score", "Genes"]].head(top_n)
