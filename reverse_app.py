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

# Curated datasets
curated_registry = {
  "stroke": [...],  # Your curated stroke dataset entries
  "vte": [...],     # Your curated VTE dataset entries
  "aps": [...]      # Your curated APS dataset entries
}

# Determine domain
keywords = extract_keywords_from_query(query)
if any("stroke" in k for k in keywords):
    selected_domain = "stroke"
elif any(k in ["vte", "thrombosis", "dvt"] for k in keywords):
    selected_domain = "vte"
elif any("aps" in k for k in keywords):
    selected_domain = "aps"
else:
    selected_domain = None

# Show curated results
st.markdown("### 📦 Curated Datasets")
if selected_domain:
    domain_df = pd.DataFrame(curated_registry[selected_domain])
    domain_df.columns = domain_df.columns.astype(str).str.strip()
    st.write("🔍 Curated columns:", list(domain_df.columns))
    if any(c.lower() == "organism" for c in domain_df.columns):
        org_col = [c for c in domain_df.columns if c.lower() == "organism"][0]
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Curated Animal Datasets**")
            st.dataframe(domain_df[domain_df[org_col] != "Human"].reset_index(drop=True))
        with col2:
            st.markdown("**Curated Human Datasets**")
            st.dataframe(domain_df[domain_df[org_col] == "Human"].reset_index(drop=True))
    else:
        st.warning("⚠️ 'Organism' column not found in curated dataset.")
else:
    st.info("No curated datasets matched your keyword. Try 'stroke', 'VTE', or 'APS'.")

# Smart GEO Search
st.markdown("### 🔍 Smart Animal GEO Dataset Discovery")
if st.button("Run smart search"):
    try:
        results = smart_search_animal_geo(query, species_input)
        if results is not None and not pd.DataFrame(results).empty:
            results_df = pd.DataFrame(results)
            st.write("Smart search result columns:", results_df.columns.tolist())
            if "organism" in results_df.columns:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Found Animal Datasets**")
                    st.dataframe(results_df[results_df["organism"] != "Homo sapiens"].reset_index(drop=True))
                with col2:
                    st.markdown("**Found Human Datasets**")
                    st.dataframe(results_df[results_df["organism"] == "Homo sapiens"].reset_index(drop=True))
            else:
                st.warning("⚠️ 'organism' column not found in smart search results.")
        else:
            st.warning("No datasets found.")
    except Exception as e:
        st.error(f"Search failed: {e}")
