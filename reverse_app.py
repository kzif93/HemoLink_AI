import os
import sys
import streamlit as st
import pandas as pd
import time

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from preprocessing import preprocess_dataset
from model_training import train_model
from prediction import test_model_on_dataset
from explainability import extract_shap_values, compare_shap_vectors
from reverse_modeling import list_animal_datasets, load_multiple_datasets
from smart_geo_animal_search import (
    smart_search_animal_geo,
    download_animal_dataset,
    extract_keywords_from_query,
    download_and_prepare_dataset
)
from curated_sets import curated_registry

st.set_page_config(page_title="HemoLink_AI – Reverse Modeling", layout="wide")

st.markdown("""
    <h1 style='margin-bottom: 5px;'>Reverse Modeling – Match Human Data to Animal Models</h1>
    <p style='color: gray;'>Upload your own dataset or search GEO to train on multiple datasets and evaluate against preclinical models.</p>
""", unsafe_allow_html=True)

# --- Step 1: Search ---
st.markdown("## Step 1: Search for Human or Animal Datasets")
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
        with st.spinner("Searching GEO datasets..."):
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
    selected_gses = st.multiselect("Select datasets to use for modeling:", combined_df["GSE"].tolist())

    if selected_gses:
        st.success(f"✅ Selected GSEs: {selected_gses}")

        curated_humans = set(curated_df[curated_df["Organism"] == "Human"]["GSE"].str.lower())
        human_gses = [g for g in selected_gses if g.lower() in curated_humans]
        animal_gses = [g for g in selected_gses if g.lower() not in curated_humans]

        # --- Attempt download of missing datasets ---
        st.markdown("### 🔄 Downloading and Preparing Missing Data")
        with st.spinner("Checking and downloading missing datasets..."):
            for gse in selected_gses:
                exp_path = os.path.join("data", f"{gse}_expression.csv")
                lab_path = os.path.join("data", f"{gse}_labels.csv")
                if not os.path.exists(exp_path):
                    st.info(f"📥 Attempting download for {gse}...")
                    try:
                        download_and_prepare_dataset(gse)
                        st.success(f"✅ Downloaded and saved {gse}")
                    except Exception as e:
                        st.error(f"❌ Failed to download {gse}: {e}")
                else:
                    st.info(f"✅ {gse} already exists.")

        # --- Step 3: Train Model ---
        st.markdown("## Step 3: Train Model on Selected Human Data")
        try:
            if human_gses:
                result = load_multiple_datasets(human_gses)
                if not result or len(result) != 2:
                    raise ValueError("Returned data is empty or malformed.")
                human_df, labels = result
                if human_df.empty or labels.empty:
                    raise ValueError("Loaded data or labels are empty.")

                st.write(f"📂 Loaded Human Training Data: {human_df.shape}")
                X, y = preprocess_dataset(human_df, labels)
                model, metrics = train_model(X, y)
                st.success("✅ Model training complete")
                st.json(metrics)
            else:
                st.warning("⚠️ No human datasets selected for training.")
        except Exception as e:
            st.error(f"❌ Failed to train: {e}")

        # --- Step 4: Evaluate ---
        st.markdown("## Step 4: Evaluate on Animal Datasets")
        try:
            if animal_gses:
                result = load_multiple_datasets(animal_gses)
                if not result or len(result) != 2:
                    raise ValueError("Returned data is empty or malformed.")
                eval_dfs, meta = result
                results = test_model_on_dataset(model, eval_dfs, meta)
                st.dataframe(results)
            else:
                st.warning("⚠️ No animal datasets selected for evaluation.")
        except Exception as e:
            st.error(f"❌ Evaluation failed: {e}")
else:
    st.info("ℹ️ No datasets available to select.")
