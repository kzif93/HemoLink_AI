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

# Curated datasets (hardcoded, trimmed for brevity)
curated_registry = {
  "stroke": [
    {"GSE": "GSE16561", "Organism": "Human", "Model": "Ischemic Stroke", "Platform": "GPL570", "Description": "Whole blood transcriptome profiling of stroke patients.", "Tag": "⭐ Curated"},
    {"GSE": "GSE233813", "Organism": "Mouse", "Model": "MCAO", "Platform": "RNA-Seq", "Description": "Mouse MCAO model", "Tag": "⭐ Curated"},
  ],
  "vte": [
    {"GSE": "GSE19151", "Organism": "Human", "Model": "VTE", "Platform": "Microarray", "Description": "VTE expression dataset.", "Tag": "⭐ Curated"},
    {"GSE": "GSE125965", "Organism": "Mouse", "Model": "IVC DVT", "Platform": "Microarray", "Description": "Mouse vein thrombosis model", "Tag": "⭐ Curated"},
  ]
}

keywords = extract_keywords_from_query(query)
if any("stroke" in k for k in keywords):
    selected_domain = "stroke"
elif any(k in ["vte", "thrombosis", "dvt"] for k in keywords):
    selected_domain = "vte"
else:
    selected_domain = None

# --- Curated Section ---
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

# --- Smart Search ---
st.markdown("### 🔍 Smart Animal GEO Dataset Discovery")
search_results_df = pd.DataFrame()
if st.button("Run smart search"):
    try:
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

# --- Step 2: Select Datasets ---
st.markdown("## Step 2: Select Dataset(s) for Modeling")
combined_df = pd.concat([curated_df, search_results_df], ignore_index=True).dropna(subset=["GSE"]).drop_duplicates(subset="GSE")
if not combined_df.empty:
    selected_gses = st.multiselect("Select datasets to use for training:", combined_df["GSE"].tolist())

    if selected_gses:
        st.success(f"✅ Selected GSEs: {selected_gses}")

        # --- Step 3: Load & Train ---
        st.markdown("## Step 3: Train Model on Selected Human Data")
        human_gses = [g for g in selected_gses if g.lower() in curated_df[curated_df["Organism"] == "Human"]["GSE"].str.lower().tolist()]
        try:
            if human_gses:
                human_df, labels = load_multiple_datasets(human_gses, kind="human")
                st.write(f"📂 Loaded Human Training Data: {human_df.shape}")
                X, y = preprocess_dataset(human_df, labels)
                model, metrics = train_model(X, y)
                st.json(metrics)
            else:
                st.warning("⚠️ No human training datasets among selection.")
        except Exception as e:
            st.error(f"❌ Failed to train: {e}")

        st.markdown("## Step 4: Evaluate on Animal Datasets")
        animal_gses = [g for g in selected_gses if g.lower() not in human_gses]
        try:
            if animal_gses:
                eval_dfs, meta = load_multiple_datasets(animal_gses, kind="animal")
                results = test_model_on_dataset(model, eval_dfs, meta)
                st.dataframe(results)
            else:
                st.warning("⚠️ No animal datasets selected for evaluation.")
        except Exception as e:
            st.error(f"❌ Evaluation failed: {e}")
else:
    st.info("ℹ️ No datasets available to select.")
