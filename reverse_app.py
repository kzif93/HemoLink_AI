import os
import sys
import streamlit as st
import pandas as pd
import re
from Bio import Entrez
from datetime import datetime

# Ensure src/ is in path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from preprocessing import preprocess_dataset
from model_training import train_model
from prediction import test_model_on_dataset
from explainability import extract_shap_values, compare_shap_vectors
from reverse_modeling import list_animal_datasets, load_multiple_datasets
from curated_sets import curated_registry
from probe_mapper import download_platform_annotation, map_probes_to_genes
import GEOparse

# ---- SMART GEO SEARCH ----
Entrez.email = "your_email@example.com"

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
    out_path = os.path.join("data", f"{gse}_expression.csv")
    label_out = os.path.join("data", f"{gse}_labels.csv")
    if os.path.exists(out_path) and os.path.exists(label_out):
        return out_path, label_out

    geo = GEOparse.get_GEO(geo=gse, destdir="data", annotate_gpl=True)
    gpl_name = list(geo.gpls.keys())[0] if geo.gpls else None
    df = geo.pivot_samples("VALUE")
    df.to_csv(out_path)

    probe_ids = df.index.to_series()
    looks_like_probes = probe_ids.str.endswith("_at").sum() / len(probe_ids) > 0.9
    if looks_like_probes and gpl_name:
        gpl_path = download_platform_annotation(gse)
        mapped = map_probes_to_genes(out_path, gpl_path)
        mapped = mapped.T
        mapped.to_csv(out_path)

    # Auto-labeling
    try:
        metadata = pd.DataFrame({gsm: sample.metadata for gsm, sample in geo.gsms.items()}).T
        candidates = metadata.select_dtypes(include='object').apply(lambda col: col.astype(str).str.lower())
        for col in candidates.columns:
            if candidates[col].str.contains("case|control|patient|stroke|healthy").any():
                binary = candidates[col].str.contains("case|stroke|patient", case=False).astype(int)
                binary.name = "label"
                binary.to_csv(label_out)
                return out_path, label_out
    except Exception as e:
        print(f"Auto-labeling failed: {e}")
    return out_path, None

# ---- STREAMLIT UI ----
st.set_page_config(page_title="HemoLink_AI – Reverse Modeling", layout="wide")
st.title("Reverse Modeling – Match Human Data to Animal Models")

# Step 1: Search input
st.subheader("Step 1: Search for Human or Animal Datasets")
query = st.text_input("Enter disease keyword (e.g., stroke, thrombosis, APS):", value="stroke")
species_input = st.text_input("Species (optional, e.g., Mus musculus):")

keywords = extract_keywords_from_query(query)
if any("stroke" in k for k in keywords):
    selected_domain = "stroke"
elif any(k in ["vte", "thrombosis", "dvt"] for k in keywords):
    selected_domain = "vte"
elif any("aps" in k for k in keywords):
    selected_domain = "aps"
else:
    selected_domain = None

# Curated datasets
st.markdown("### 📦 Curated Datasets")
curated_df = pd.DataFrame()
if selected_domain:
    try:
        curated = curated_registry[selected_domain]
        curated_df = pd.DataFrame(curated)
        curated_df.columns = curated_df.columns.astype(str).str.strip()
        if "Organism" in curated_df.columns:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Curated Animal Datasets**")
                st.dataframe(curated_df[curated_df["Organism"] != "Human"].reset_index(drop=True))
            with col2:
                st.markdown("**Curated Human Datasets**")
                st.dataframe(curated_df[curated_df["Organism"] == "Human"].reset_index(drop=True))
    except Exception as e:
        st.error(f"❌ Failed to load curated datasets: {e}")

# Smart search
st.markdown("### 🔍 Smart GEO Dataset Discovery")
search_results_df = pd.DataFrame()
if st.button("Run smart search"):
    try:
        with st.spinner("Searching GEO..."):
            results = smart_search_animal_geo(query, species_input)
        search_results_df = pd.DataFrame(results)
        if not search_results_df.empty:
            if "Organism" in search_results_df.columns:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Found Animal Datasets**")
                    st.dataframe(search_results_df[search_results_df["Organism"] != "Homo sapiens"].reset_index(drop=True))
                with col2:
                    st.markdown("**Found Human Datasets**")
                    st.dataframe(search_results_df[search_results_df["Organism"] == "Homo sapiens"].reset_index(drop=True))
    except Exception as e:
        st.error(f"Search failed: {e}")

# Step 2: Select datasets
st.subheader("Step 2: Select Datasets")
combined_df = pd.concat([curated_df, search_results_df], ignore_index=True).dropna(subset=["GSE"]).drop_duplicates(subset="GSE")
selected_gses = st.multiselect("Select GSEs:", combined_df["GSE"].tolist())

if selected_gses:
    st.success(f"✅ Selected GSEs: {selected_gses}")

    # Download and prepare
    st.markdown("### 🔄 Downloading and Preparing")
    expression_dfs = []
    label_series = []

    for gse in selected_gses:
        exp_path = os.path.join("data", f"{gse}_expression.csv")
        lbl_path = os.path.join("data", f"{gse}_labels.csv")
        if not (os.path.exists(exp_path) and os.path.exists(lbl_path)):
            st.info(f"📥 Downloading {gse}...")
            exp_path, lbl_path = download_and_prepare_dataset(gse)
        if os.path.exists(exp_path) and os.path.exists(lbl_path):
            df = pd.read_csv(exp_path, index_col=0)
            labels = pd.read_csv(lbl_path, index_col=0).squeeze("columns")
            df["label"] = labels
            expression_dfs.append(df)
            st.success(f"✅ {gse} ready")

    # Combine all
    if expression_dfs:
        full_df = pd.concat(expression_dfs, axis=0, join="inner")
        st.write("🧬 Combined dataset preview:")
        st.dataframe(full_df.head())

        # Step 3: Train
        st.subheader("Step 3: Train Model")
        try:
            X, y = preprocess_dataset(full_df, label_column="label")
            model, metrics = train_model(X, y)
            st.success("✅ Model trained")
            st.json(metrics)
        except Exception as e:
            st.error(f"❌ Failed to train: {e}")
else:
    st.info("ℹ️ Select at least one dataset to proceed.")
