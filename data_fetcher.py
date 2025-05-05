# data_fetcher.py

import os
import pandas as pd
import requests
import GEOparse
import streamlit as st
from xml.etree import ElementTree as ET

# GEO keyword-based search using Entrez API
def search_geo_by_keyword(query, retmax=20):
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "gds",
        "term": query,
        "retmode": "xml",
        "retmax": retmax
    }
    response = requests.get(base_url, params=params)
    tree = ET.fromstring(response.content)
    ids = [id_elem.text for id_elem in tree.findall(".//Id")]

    summaries = []
    for gds_id in ids:
        summary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        summary_params = {
            "db": "gds",
            "id": gds_id,
            "retmode": "xml"
        }
        sum_resp = requests.get(summary_url, params=summary_params)

        try:
            sum_tree = ET.fromstring(sum_resp.content)
        except ET.ParseError:
            continue

        docsum = sum_tree.find(".//DocSum")
        gse = title = None
        for item in docsum.findall("Item"):
            if item.attrib.get("Name") == "GSE":
                gse = item.text
            if item.attrib.get("Name") == "title":
                title = item.text
        if gse and title:
            summaries.append((gse, title))

    return summaries

def fetch_geo_series(geo_id, out_dir="data"):
    os.makedirs(out_dir, exist_ok=True)
    gse = GEOparse.get_GEO(geo=geo_id, destdir=out_dir)
    df = gse.pivot_samples('VALUE')
    clean_path = os.path.join(out_dir, f"{geo_id}_expression.csv")
    df.to_csv(clean_path)
    st.success(f"✅ Saved GEO data to {clean_path}")
    return df

def search_refinebio(query, limit=5):
    try:
        url = f"https://api.refine.bio/v1/search/?search={query}&limit={limit}"
        r = requests.get(url)
        r.raise_for_status()
        data = r.json()
        return data.get("results", [])
    except Exception as e:
        st.error(f"Refine.bio search failed: {e}")
        return []

def list_refinebio_titles(results):
    titles = []
    for ds in results:
        titles.append(f"{ds['accession_code']}: {ds['title']}")
    return titles

def dataset_search_ui():
    st.sidebar.markdown("### 🔍 Search Public Datasets")
    query = st.sidebar.text_input("Search keyword (e.g. stroke, thrombosis)", "stroke")

    if st.sidebar.button("Search GEO (Entrez)"):
        geo_hits = search_geo_by_keyword(query)
        st.session_state["geo_hits"] = geo_hits

    geo_hits = st.session_state.get("geo_hits", [])
    if geo_hits:
        st.sidebar.markdown("**Top GEO Results:**")
        for i, (gse_id, title) in enumerate(geo_hits):
            if st.sidebar.button(f"⬇️ {gse_id}", key=f"geo_download_{i}"):
                st.session_state["gse_to_download"] = gse_id

    if "gse_to_download" in st.session_state:
        gse_id = st.session_state.pop("gse_to_download")
        st.sidebar.write(f"⏳ Downloading {gse_id} ...")
        fetch_geo_series(gse_id)

    if st.sidebar.button("Search Refine.bio"):
        results = search_refinebio(query)
        titles = list_refinebio_titles(results)
        st.sidebar.markdown("**Top Refine.bio Results:**")
        for title in titles:
            st.sidebar.write(title)
