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
from reverse_modeling import load_multiple_datasets
from curated_sets import curated_registry
from probe_mapper import download_platform_annotation, map_probes_to_genes

Entrez.email = "your_email@example.com"

# Search helpers
def extract_keywords_from_query(query):
    return [w.strip().lower() for w in re.split(r"[\s,]+", query)]

def smart_search_geo(query, species=None, max_results=100):
    try:
        keywords = extract_keywords_from_query(query)
        term = f"{' OR '.join(keywords)} AND (gse[ETYP] OR gds[ETYP])"
        if species:
            term += f" AND {species}"
        handle = Entrez.esearch(db="gds", term=term, retmax=max_results)
        record = Entrez.read(handle)
        ids = record["IdList"]

        summaries = []
        for gds_id in ids:
            summary = Entrez.esummary(db="gds", id=gds_id)
            doc = Entrez.read(summary)[0]
            summaries.append({
                "GSE": doc.get("Accession", "?"),
                "Title": doc.get("title", "?"),
                "Description": doc.get("summary", "?"),
                "Samples": doc.get("n_samples", "?"),
                "Platform": doc.get("gpl", "?"),
                "Organism": doc.get("taxon", "?"),
                "ReleaseDate": doc.get("PDAT", "?"),
                "Tag": "GEO"
            })
        return pd.DataFrame(summaries)
    except Exception as e:
        st.error(f"Search error: {e}")
        return pd.DataFrame()

def download_and_prepare_dataset(gse):
    import GEOparse

    out_path = f"data/{gse}_expression.csv"
    label_out = f"data/{gse}_labels.csv"
    if os.path.exists(out_path):
        return out_path

    geo = GEOparse.get_GEO(geo=gse, destdir="data", annotate_gpl=True)
    gpl_name = list(geo.gpls.keys())[0] if geo.gpls else None
    df = geo.pivot_samples("VALUE")
    df.to_csv(out_path)

    probe_ids = df.index.to_series()
    if probe_ids.str.endswith("_at").sum() / len(probe_ids) > 0.9 and gpl_name:
        gpl_path = download_platform_annotation(gse)
        mapped = map_probes_to_genes(out_path, gpl_path)
        mapped = mapped.T
        mapped.to_csv(out_path)

    try:
        metadata = pd.DataFrame({gsm: s.metadata for gsm, s in geo.gsms.items()}).T
        if "disease state" in metadata.columns:
            labels = metadata["disease state"].astype(str).str.lower().map(lambda x: 1 if "case" in x or "stroke" in x else 0)
        elif "title" in metadata.columns:
            labels = metadata["title"].astype(str).str.lower().map(lambda x: 1 if "stroke" in x or "patient" in x else 0)
        else:
            labels = pd.Series([0] * df.shape[1], index=df.columns)
        labels.name = "label"
        labels.to_csv(label_out)
    except Exception as e:
        st.warning(f"Auto-labeling failed for {gse}: {e}")
    return out_path

# Streamlit UI
st.set_page_config(page_title="HemoLink_AI – Reverse Modeling", layout="wide")
st.markdown("<h1>Reverse Modeling – Match Human Data to Animal Models</h1>", unsafe_allow_html=True)

query = st.text_input("Step 1: Enter disease keyword (e.g., stroke, thrombosis, APS):", value="stroke")
species = st.text_input("Optional species (e.g., Mus musculus):")

selected_domain = None
kws = extract_keywords_from_query(query)
if any("stroke" in k for k in kws):
    selected_domain = "stroke"
elif any(k in ["vte", "thrombosis", "dvt"] for k in kws):
    selected_domain = "vte"
elif any("aps" in k for k in kws):
    selected_domain = "aps"

curated_df = pd.DataFrame()
if selected_domain and selected_domain in curated_registry:
    curated_df = pd.DataFrame(curated_registry[selected_domain])
    curated_df.columns = curated_df.columns.astype(str).str.strip()

st.markdown("### 📦 Curated Datasets")
if not curated_df.empty and "Organism" in curated_df.columns:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Curated Animal Datasets**")
        st.dataframe(curated_df[curated_df["Organism"] != "Human"])
    with col2:
        st.markdown("**Curated Human Datasets**")
        st.dataframe(curated_df[curated_df["Organism"] == "Human"])

st.markdown("### 🔍 Smart GEO Dataset Discovery")
if st.button("Run smart search"):
    with st.spinner("Searching GEO..."):
        smart_df = smart_search_geo(query, species)
    if not smart_df.empty:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Smart Animal Datasets**")
            st.dataframe(smart_df[smart_df["Organism"] != "Homo sapiens"])
        with col2:
            st.markdown("**Smart Human Datasets**")
            st.dataframe(smart_df[smart_df["Organism"] == "Homo sapiens"])
else:
    smart_df = pd.DataFrame()

st.markdown("## Step 2: Select Datasets")
combined_df = pd.concat([curated_df, smart_df], ignore_index=True).dropna(subset=["GSE"]).drop_duplicates(subset="GSE")
selected = st.multiselect("Select GSEs:", combined_df["GSE"].tolist())

if selected:
    st.success(f"✅ Selected GSEs: {selected}")
    curated_human_gses = set(curated_df[curated_df["Organism"] == "Human"]["GSE"].str.lower())
    human_gses = [g for g in selected if g.lower() in curated_human_gses or g == "GSE16561"]
    animal_gses = [g for g in selected if g not in human_gses]

    # Step 3: Download and Train
    st.markdown("### 🔄 Downloading and Preparing")
    for gse in selected:
        try:
            download_and_prepare_dataset(gse)
            st.info(f"✅ {gse} ready")
        except Exception as e:
            st.error(f"❌ {gse} failed: {e}")

    st.markdown("## Step 3: Train Model")
    if human_gses:
        try:
            df, labels = load_multiple_datasets(human_gses)
            if isinstance(labels, pd.DataFrame):
                labels = labels.iloc[:, 0]
            X, y = preprocess_dataset(df, label_column=labels.name)
            model, metrics = train_model(X, y)
            st.success("Model trained successfully.")
            st.json(metrics)
        except Exception as e:
            st.error(f"❌ Failed to train: {e}")
    else:
        st.warning("⚠️ No human datasets selected.")

    # Step 4: Evaluate
    st.markdown("## Step 4: Evaluate on Animal Data")
    if animal_gses:
        try:
            animal_df, meta = load_multiple_datasets(animal_gses)
            preds = test_model_on_dataset(model, animal_df, meta)
            st.dataframe(preds)
        except Exception as e:
            st.error(f"❌ Evaluation failed: {e}")
    else:
        st.warning("⚠️ No animal datasets selected.")
