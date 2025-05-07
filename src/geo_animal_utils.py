import os
import GEOparse
from Bio import Entrez
import pandas as pd
import streamlit as st

Entrez.email = "your_email@example.com"  # Update if needed

@st.cache_data(show_spinner=False)
def search_animal_geo_datasets(keyword="stroke", organism="Mus musculus", max_results=15):
    query = f"{keyword} AND {organism}[Organism] AND gse[Entry Type]"
    handle = Entrez.esearch(db="gds", term=query, retmax=max_results)
    record = Entrez.read(handle)
    handle.close()
    ids = record.get("IdList", [])
    summaries = []

    for gid in ids:
        summary_handle = Entrez.esummary(db="gds", id=gid)
        summary = Entrez.read(summary_handle)
        summary_handle.close()
        if summary:
            item = summary[0]
            summaries.append({
                "GSE": item.get("Accession", "N/A"),
                "Title": item.get("title", ""),
                "Samples": item.get("n_samples", 0),
                "Platform": item.get("GPL", ""),
                "Organism": item.get("taxon", ""),
                "ReleaseDate": item.get("PDAT", "")
            })

    return pd.DataFrame(summaries)

@st.cache_data(show_spinner=True)
def download_animal_dataset(gse_id):
    gse = GEOparse.get_GEO(geo=gse_id, destdir="animal_models", annotate_gpl=True)
    expression_df = gse.pivot_samples("VALUE")
    expression_path = os.path.join("animal_models", f"{gse_id}_expression.csv")
    expression_df.to_csv(expression_path)
    return expression_df, gse

def animal_dataset_search_ui():
    st.markdown("### 🐭 Search GEO for Animal Model Datasets")
    keyword = st.text_input("Animal disease keyword (e.g., stroke, thrombosis)", value="stroke")
    organism = st.selectbox("Select organism:", ["Mus musculus", "Rattus norvegicus", "Danio rerio"])

    if st.button("🔍 Search GEO for Animal Datasets"):
        with st.spinner("Searching GEO..."):
            results_df = search_animal_geo_datasets(keyword=keyword, organism=organism)
        if results_df.empty:
            st.warning("No datasets found.")
        else:
            st.dataframe(results_df)
            selected = st.multiselect("Select dataset(s) to download:", results_df["GSE"].tolist())

            for gse_id in selected:
                st.write(f"⬇️ Downloading {gse_id}...")
                df, _ = download_animal_dataset(gse_id)
                st.success(f"✅ {gse_id} saved to animal_models/")
