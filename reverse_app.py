
import os
import sys
import streamlit as st
import pandas as pd
import re
from Bio import Entrez
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from preprocessing import preprocess_dataset
from model_training import train_model
from prediction import test_model_on_dataset
from reverse_modeling import load_multiple_datasets
from curated_sets import curated_registry

Entrez.email = "your_email@example.com"

KEYWORDS = ["stroke", "ischemia", "thrombosis", "vte", "dvt", "aps", "control", "healthy", "is"]

def extract_keywords_from_query(query):
    return [w.strip().lower() for w in re.split(r"[\s,]+", query)]

def smart_search_animal_geo(query, species=None, max_results=100):
    try:
        keywords = extract_keywords_from_query(query)
        search_term = f"{' OR '.join(keywords)} AND (gse[ETYP] OR gds[ETYP])"
        if species:
            search_term += f" AND {species}"
        handle = Entrez.esearch(db="gds", term=search_term, retmax=max_results)
        record = Entrez.read(handle)
        ids = record["IdList"]
        summaries = []
        for gds_id in ids:
            summary = Entrez.esummary(db="gds", id=gds_id)
            docsum = Entrez.read(summary)[0]
            summaries.append({
                "GSE": docsum.get("Accession", "?"),
                "Title": docsum.get("title", "?"),
                "Description": docsum.get("summary", "?"),
                "Samples": docsum.get("n_samples", "?"),
                "Platform": docsum.get("gpl", "?"),
                "Organism": docsum.get("taxon", "?"),
                "ReleaseDate": docsum.get("PDAT", "?"),
                "Score": 0,
                "Tag": "GEO"
            })
        return summaries
    except Exception as e:
        print(f"[smart_search_animal_geo] Error: {e}")
        return []

def download_and_prepare_dataset(gse):
    import GEOparse
    from probe_mapper import download_platform_annotation, map_probes_to_genes

    out_path = os.path.join("data", f"{gse}_expression.csv")
    label_out = os.path.join("data", f"{gse}_labels.csv")
    if os.path.exists(out_path):
        return out_path

    geo = GEOparse.get_GEO(geo=gse, destdir="data", annotate_gpl=True)
    gpl_name = list(geo.gpls.keys())[0] if geo.gpls else None
    df = pd.DataFrame({gsm: sample.table.set_index("ID_REF")["VALUE"] for gsm, sample in geo.gsms.items()})
    df.to_csv(out_path)

    if df.index.str.endswith("_at").sum() / len(df.index) > 0.5 and gpl_name:
        gpl_path = download_platform_annotation(gse)
        mapped = map_probes_to_genes(out_path, gpl_path)
        mapped = mapped.T
        mapped.to_csv(out_path)

    try:
        sample_titles = pd.Series({gsm: sample.metadata.get("title", [""])[0] for gsm, sample in geo.gsms.items()})
        labels = sample_titles.str.lower().map(lambda x: 1 if any(k in x for k in ["stroke", "is"]) else 0)
        labels.name = "label"
        label_dist = labels.value_counts().to_dict()
        st.warning(f"⚠️ Label distribution: {label_dist}")
        if labels.nunique() < 2:
            st.error("❌ Only one class found. Cannot proceed with training.")
        labels.to_csv(label_out)
    except Exception as e:
        print(f"[Labeling error] {e}")

    return out_path
