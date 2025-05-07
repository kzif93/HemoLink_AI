import os
import GEOparse
from Bio import Entrez
import pandas as pd
import streamlit as st

Entrez.email = "your_email@example.com"  # Replace with your actual email

PREFERRED_PLATFORMS = ["RNA-Seq", "GPL21103", "GPL17021"]

CURATED_GSES = {
    "GSE233813": "⭐ Curated",
    "GSE162072": "⭐ Curated",
    "GSE137482": "⭐ Curated",
    "GSE36010": "⭐ Curated",
    "GSE78731": "⭐ Curated",
    "GSE16561": "⭐ Curated",
    "GSE22255": "⭐ Curated",
    "GSE58294": "⭐ Curated",
    "GSE37587": "⭐ Curated",
    "GSE162955": "⭐ Curated"
}

def extract_keywords_from_query(query):
    return [kw.strip().lower() for kw in query.replace(",", " ").split() if len(kw) > 2]

@st.cache_data(show_spinner=False)
def pubmed_to_geo(pmids):
    geo_ids = set()
    for pmid in pmids:
        try:
            handle = Entrez.elink(dbfrom="pubmed", db="gds", id=pmid, linkname="pubmed_gds")
            records = Entrez.read(handle)
            handle.close()
            for link in records[0].get("LinkSetDb", []):
                for ref in link["Link"]:
                    geo_ids.add(ref["Id"])
        except Exception:
            continue
    return list(geo_ids)

@st.cache_data(show_spinner=False)
def smart_search_animal_geo(keyword="stroke", organism="", max_results=50):
    org_clause = f" AND {organism}[Organism]" if organism else ""
    query = f"{keyword}{org_clause} AND gse[Entry Type]"

    all_ids = set()
    for db in ["gds", "gse"]:
        try:
            handle = Entrez.esearch(db=db, term=query, retmax=max_results)
            record = Entrez.read(handle)
            handle.close()
            ids = record.get("IdList", [])
            all_ids.update(ids)
        except Exception:
            continue

    gse_summaries = {}
    user_keywords = extract_keywords_from_query(keyword)

    for gid in all_ids:
        try:
            summary_handle = Entrez.esummary(db="gds", id=gid)
            summary = Entrez.read(summary_handle)
            summary_handle.close()
            if summary:
                item = summary[0]
                gse_id = item.get("Accession", "")
                title = item.get("title", "")
                desc = item.get("summary", "")
                combined = f"{title} {desc}".lower()
                score = 0
                score += sum(kw in combined for kw in user_keywords) * 2
                if any(pl.lower() in combined for pl in PREFERRED_PLATFORMS):
                    score += 2
                if gse_id in CURATED_GSES:
                    score += 10
                gse_summaries[gse_id] = {
                    "GSE": gse_id,
                    "Title": title,
                    "Description": desc,
                    "Samples": item.get("n_samples", 0),
                    "Platform": item.get("GPL", ""),
                    "Organism": item.get("taxon", ""),
                    "ReleaseDate": item.get("PDAT", ""),
                    "Score": score,
                    "Tag": CURATED_GSES.get(gse_id, "")
                }
        except Exception:
            continue

    for gse_id, tag in CURATED_GSES.items():
        if gse_id not in gse_summaries:
            gse_summaries[gse_id] = {
                "GSE": gse_id,
                "Title": f"[Manual Insert] {gse_id}",
                "Description": "Manually curated dataset.",
                "Samples": "?",
                "Platform": "?",
                "Organism": "?",
                "ReleaseDate": "?",
                "Score": 15,
                "Tag": tag
            }

    return pd.DataFrame(list(gse_summaries.values())).sort_values("Score", ascending=False)

@st.cache_data(show_spinner=True)
def download_animal_dataset(gse_id):
    gse = GEOparse.get_GEO(geo=gse_id, destdir="animal_models", annotate_gpl=True)
    expression_df = gse.pivot_samples("VALUE")
    expression_path = os.path.join("animal_models", f"{gse_id}_expression.csv")
    expression_df.to_csv(expression_path)
    return expression_df, gse

def smart_animal_dataset_search_ui():
    st.markdown("### 🧠 Smart Animal GEO Dataset Discovery")
    keyword = st.text_input("Keyword or PubMed/PMC ID (e.g., stroke, thrombosis, PMC10369109):", value="stroke")
    organism = st.text_input("Species (e.g., Mus musculus, Rattus norvegicus — leave blank to search all species)", value="")

    if st.button("🔍 Run Smart Search"):
        geo_ids_from_pub = []
        if any(tag in keyword.lower() for tag in ["pmc", "pmid"]):
            pmid_clean = keyword.lower().replace("pmc", "").replace("pmid", "").strip()
            if pmid_clean.isdigit():
                geo_ids_from_pub = pubmed_to_geo([pmid_clean])

        results_df = smart_search_animal_geo(keyword=keyword, organism=organism)
        if geo_ids_from_pub:
            results_df["PubLinked"] = results_df["GSE"].apply(lambda x: x in geo_ids_from_pub)
            results_df["Score"] += results_df["PubLinked"].astype(int) * 5
            results_df = results_df.sort_values(by="Score", ascending=False)

        if results_df.empty:
            st.warning("No GEO datasets found.")
        else:
            st.dataframe(results_df[["GSE", "Title", "Platform", "Samples", "Organism", "Tag", "Score"]])
            selected = st.multiselect("Select dataset(s) to download:", results_df["GSE"].tolist())
            for gse_id in selected:
                st.write(f"⬇️ Downloading {gse_id}...")
                df, _ = download_animal_dataset(gse_id)
                st.success(f"✅ {gse_id} saved to animal_models/")
