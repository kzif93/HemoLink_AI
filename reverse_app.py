import os
import sys
import streamlit as st
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from preprocessing import preprocess_dataset
from model_training import train_model
from prediction import test_model_on_dataset
from explainability import extract_shap_values, compare_shap_vectors
from reverse_modeling import list_animal_datasets, load_multiple_datasets
from smart_geo_animal_search import (
    smart_search_animal_geo,
    download_animal_dataset,
    extract_keywords_from_query
)

st.set_page_config(page_title="HemoLink_AI – Reverse Modeling", layout="wide")

st.markdown("""
    <h1 style='margin-bottom: 5px;'>Reverse Modeling – Match Human Data to Animal Models</h1>
    <p style='color: gray;'>Upload your own dataset or search GEO to train on multiple datasets and evaluate against preclinical models.</p>
""", unsafe_allow_html=True)

# --- Step 1: Search ---
st.markdown("## Step 1: Search for Human or Animal Datasets")
query = st.text_input("Enter disease keyword (e.g., stroke, thrombosis, APS):", value="stroke")
species_input = st.text_input("Species (optional, e.g., Mus musculus):")

# Curated datasets (standard structure)
curated_registry = {
  ...  # Curated data remains unchanged from earlier update
}

keywords = extract_keywords_from_query(query)
if any("stroke" in k for k in keywords):
    selected_domain = "stroke"
elif any(k in ["vte", "thrombosis", "dvt"] for k in keywords):
    selected_domain = "vte"
elif any("aps" in k for k in keywords):
    selected_domain = "aps"
else:
    selected_domain = None

# -- Show Curated Datasets --
st.markdown("### 📦 Curated Datasets")
curated_df = pd.DataFrame()
if selected_domain:
    try:
        curated_df = pd.DataFrame(curated_registry[selected_domain])
        curated_df.columns = curated_df.columns.astype(str).str.strip()
        if "Organism" in curated_df.columns:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Curated Animal Datasets**")
                st.dataframe(curated_df[curated_df["Organism"] != "Human"].reset_index(drop=True))
            with col2:
                st.markdown("**Curated Human Datasets**")
                st.dataframe(curated_df[curated_df["Organism"] == "Human"].reset_index(drop=True))
        else:
            st.warning("⚠️ 'Organism' column not found in curated dataset.")
    except Exception as e:
        st.error(f"❌ Failed to load curated datasets: {e}")

# --- Smart GEO Search ---
st.markdown("### 🔍 Smart Animal GEO Dataset Discovery")
search_results_df = pd.DataFrame()
if st.button("Run smart search"):
    try:
        results = smart_search_animal_geo(query, species_input)
        if results is not None and not pd.DataFrame(results).empty:
            search_results_df = pd.DataFrame(results)
            if "Organism" in search_results_df.columns:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Found Animal Datasets**")
                    st.dataframe(search_results_df[search_results_df["Organism"] != "Homo sapiens"].reset_index(drop=True))
                with col2:
                    st.markdown("**Found Human Datasets**")
                    st.dataframe(search_results_df[search_results_df["Organism"] == "Homo sapiens"].reset_index(drop=True))
            else:
                st.warning("⚠️ 'Organism' column not found in smart search results.")
        else:
            st.warning("No datasets found.")
    except Exception as e:
        st.error(f"Search failed: {e}")

# --- Step 2: Select Datasets to Train On ---
st.markdown("## Step 2: Select Dataset(s) for Modeling")

combined_sources = pd.concat([curated_df, search_results_df], ignore_index=True).drop_duplicates(subset=["GSE"])
if not combined_sources.empty:
    selected_gses = st.multiselect("Select dataset(s) to use for training:", combined_sources["GSE"].dropna().unique())

    if selected_gses:
        st.success(f"You selected: {selected_gses}")
        st.info("🧬 In future steps, these datasets will be downloaded/preprocessed and used for model training.")
    else:
        st.info("Select one or more datasets above to continue.")
else:
    st.warning("No datasets available to select from.")
