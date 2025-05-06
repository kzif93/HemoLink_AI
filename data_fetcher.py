# data_fetcher.py

import os
import re
import pandas as pd
import requests
import GEOparse
import streamlit as st
from xml.etree import ElementTree as ET

# Trusted stroke-related human GEO datasets
TRUSTED_GEO = {
    "GSE16561": "Peripheral blood transcriptome in early ischemic stroke",
    "GSE22255": "PBMCs in ischemic stroke vs control",
    "GSE58294": "Temporal profile from 3h–24h post-stroke",
    "GSE37587": "Cardioembolic vs large vessel stroke",
    "GSE162955": "scRNA-seq of blood leukocytes post-stroke"
}

def search_geo_by_keyword(query, retmax=20):
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "gds",
        "term": f"{query} AND gse[Entry Type] AND txid9606[Organism]",
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
        title = gse_id = None
        for item in docsum.findall("Item"):
            if item.attrib.get("Name") == "Accession":
                gse_id = item.text
            if item.attrib.get("Name") == "title":
                title = item.text

        if gse_id and title and gse_id.startswith("GSE"):
            st.sidebar.write(f"✅ Valid GSE: {gse_id} — {title}")
            summaries.append((gse_id, title))
        else:
            st.sidebar.write(f"❌ Skipped entry (ID={gds_id}) — No valid GSE")

    st.sidebar.write(f"🔍 Found {len(summaries)} valid GSEs.")
    return summaries

def fetch_geo_series(geo_id, out_dir="data"):
    os.makedirs(out_dir, exist_ok=True)
    try:
        gse = GEOparse.get_GEO(geo=geo_id, destdir=out_dir)
        df = None

        try:
            df = gse.pivot_samples("VALUE")
        except Exception as e:
            st.warning(f"⚠️ Pivot failed: {e}")
            if hasattr(gse, "table") and isinstance(gse.table, pd.DataFrame):
                df = gse.table
            else:
                raise ValueError("No usable expression table available.")

        clean_path = os.path.join(out_dir, f"{geo_id}_expression.csv")
        df.to_csv(clean_path)
        st.success(f"✅ Saved GEO data to {clean_path}")
        return df

    except Exception as e:
        st.error(f"❌ Failed to download or process {geo_id}: {e}")
        return None

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
        st.sidebar.markdown("**Top GEO Results (Series Only):**")
        for i, (gse_id, title) in enumerate(geo_hits):
            if st.sidebar.button(f"⬇️ {gse_id}", key=f"geo_download_{i}"):
                st.session_state["gse_to_download"] = gse_id

    # Always show trusted GSE library
    st.sidebar.markdown("### ⭐ Trusted Human Stroke Datasets")
    for gse_id, desc in TRUSTED_GEO.items():
        if st.sidebar.button(f"⬇️ {gse_id} — {desc}", key=f"trusted_{gse_id}"):
            fetch_geo_series(gse_id)

    if "gse_to_download" in st.session_state:
        gse_id = st.session_state.pop("gse_to_download")
        st.sidebar.write(f"⏳ Downloading {gse_id} ...")
        fetch_geo_series(gse_id)

    if st.sidebar.button("⬇️ Download GSE16561 (Test)"):
        fetch_geo_series("GSE16561")

    if st.sidebar.button("Search Refine.bio"):
        results = search_refinebio(query)
        titles = list_refinebio_titles(results)
        st.sidebar.markdown("**Top Refine.bio Results:**")
        for title in titles:
            st.sidebar.write(title)
