# data_fetcher.py

import os
import pandas as pd
import requests
import GEOparse
import streamlit as st

def fetch_geo_series(geo_id, out_dir="data"):
    """
    Download and extract expression matrix from GEO Series ID (e.g. GSE16561).
    """
    os.makedirs(out_dir, exist_ok=True)
    gse = GEOparse.get_GEO(geo=geo_id, destdir=out_dir)
    df = gse.pivot_samples('VALUE')
    clean_path = os.path.join(out_dir, f"{geo_id}_expression.csv")
    df.to_csv(clean_path)
    st.success(f"✅ Saved GEO data to {clean_path}")
    return df

def search_refinebio(query, limit=5):
    """
    Search refine.bio API for datasets matching a query string.
    """
    url = f"https://api.refine.bio/v1/dataset/?search={query}&limit={limit}"
    r = requests.get(url)
    if r.status_code != 200:
        st.error("Refine.bio API request failed")
        return []
    results = r.json()
    return results.get("results", [])

def list_refinebio_titles(results):
    """
    Utility to print dataset accession and title from search results.
    """
    titles = []
    for ds in results:
        titles.append(f"{ds['accession_code']}: {ds['title']}")
    return titles

# Streamlit Sidebar Integration
def dataset_search_ui():
    st.sidebar.markdown("### 🔍 Search Public Datasets")
    query = st.sidebar.text_input("Search keyword (e.g. stroke, thrombosis)", "stroke")

    if st.sidebar.button("Search GEO"):
        try:
            fetch_geo_series(query)
        except Exception as e:
            st.sidebar.error(f"Error: {e}")

    if st.sidebar.button("Search Refine.bio"):
        results = search_refinebio(query)
        titles = list_refinebio_titles(results)
        st.sidebar.markdown("**Top Matches:**")
        for title in titles:
            st.sidebar.write(title)

# Example usage for CLI
if __name__ == "__main__":
    fetch_geo_series("GSE16561")
    res = search_refinebio("stroke")
    for line in list_refinebio_titles(res):
        print(line)
