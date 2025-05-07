import os
import GEOparse
from Bio import Entrez
import pandas as pd
import streamlit as st

Entrez.email = "your_email@example.com"  # Replace with your actual email

# Curated list of stroke-related keywords for smarter relevance ranking
KEYWORDS = ["stroke", "MCAO", "tMCAO", "ischemia", "photothrombosis", "middle cerebral artery"]

@st.cache_data(show_spinner=False)
def search_animal_geo_datasets(keyword="stroke", organism="Mus musculus", max_results=25):
    query = f"({' OR '.join(KEYWORDS)}) AND {organism}[Organism] AND gse[Entry Type]"
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
            title = item.get("title", "")
            desc = item.get("summary", "")
            combined_text = f"{title} {desc}".lower()

            relevance_score = sum(kw.lower() in combined_text for kw in KEYWORDS)

            summaries.append({
                "GSE": item.get("Accession", "N/A"),
                "Title": title,
                "Description": desc,
                "Samples": item.get("n_samples", 0),
                "Platform": item.get("GPL", ""),
                "Organism": item.get("taxon", ""),
                "ReleaseDate": item.get("PDAT", ""),
                "Score": relevance_score
            })

    return pd.DataFrame(summaries).sort_values("Score", ascending=False)

@st.cache_data(show_spinner=True)
def download_animal_dataset(gse_id):
    gse = GEOparse.get_GEO(geo=gse_id, destdir="animal_models", annotate_gpl=True)
    expression_df = gse.pivot_samples("VALUE")
    expression_path = os.path.join("animal_models", f"{gse_id}_expression.csv")
    expression_df.to_csv(expression_path)
    return expression_df, gse

def animal_dataset_search_ui():
    st.markdown("### 🐭 Search GEO for Animal Stroke Model Datasets")
    keyword = st.text_input("Keyword(s) (e.g., stroke, MCAO, ischemia):", value="stroke")
    organism = st.selectbox("Select organism:", ["Mus musculus", "Rattus norvegicus", "Danio rerio"])

    if st.button("🔍 Search GEO"):
        with st.spinner("Searching GEO..."):
            results_df = search_animal_geo_datasets(keyword=keyword, organism=organism)

        if results_df.empty:
            st.warning("No datasets found.")
        else:
            st.dataframe(results_df[["GSE", "Title", "Samples", "Platform", "Organism", "Score"]])
            selected = st.multiselect("Select dataset(s) to download:", results_df["GSE"].tolist())

            for gse_id in selected:
                st.write(f"⬇️ Downloading {gse_id}...")
                df, _ = download_animal_dataset(gse_id)
                st.success(f"✅ {gse_id} saved to animal_models/")
