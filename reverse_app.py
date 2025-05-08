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

# --- Unified Search UI ---
st.markdown("## Step 1: Search Datasets")
query = st.text_input("🔍 Disease keyword or PubMed/PMC ID (e.g., stroke, thrombosis, PMC10369109):", value="stroke")
species_input = st.text_input("🧬 Species (e.g., Homo sapiens, Mus musculus — leave blank for all):")

# --- Curated Dataset Registry ---
curated_registry = { ... }  # Keep full curated_registry from earlier

# Match domain
keywords = extract_keywords_from_query(query)
if any("stroke" in k for k in keywords):
    selected_domain = "stroke"
elif any(k in ["vte", "thrombosis", "dvt"] for k in keywords):
    selected_domain = "vte"
elif any("aps" in k for k in keywords):
    selected_domain = "aps"
else:
    selected_domain = None

# --- Show curated datasets ---
st.markdown("### 📦 Curated Datasets")
if selected_domain:
    domain_df = pd.DataFrame(curated_registry[selected_domain])
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Curated Animal Datasets**")
        st.dataframe(domain_df[domain_df["Organism"] != "Human"].reset_index(drop=True))
    with col2:
        st.markdown("**Curated Human Datasets**")
        st.dataframe(domain_df[domain_df["Organism"] == "Human"].reset_index(drop=True))
else:
    st.info("No curated datasets matched your keyword. Try 'stroke', 'VTE', or 'APS'.")

# --- Run Smart GEO Search ---
st.markdown("### 🌐 GEO Search Results")
if st.button("🔍 Run GEO Search"):
    results = smart_search_animal_geo(query, species_input)
    if results:
        results_df = pd.DataFrame(results)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Found Animal Datasets**")
            st.dataframe(results_df[results_df["organism"] != "Homo sapiens"].reset_index(drop=True))
        with col2:
            st.markdown("**Found Human Datasets**")
            st.dataframe(results_df[results_df["organism"] == "Homo sapiens"].reset_index(drop=True))
    else:
        st.warning("No results found. Try refining your keyword or species.")

# --- Optional Upload UI ---
st.markdown("## Step 2: (Optional) Upload Your Own Dataset(s)")
st.markdown("You can upload one or more expression CSV files for custom training.\nIf no files are uploaded, the app will use the most recent downloaded dataset in the data/ folder.")
uploaded_files = st.file_uploader("Upload expression CSV files:", type=["csv"], accept_multiple_files=True)
human_files = uploaded_files if uploaded_files else []
human_paths = []

if human_files:
    for file in human_files:
        path = os.path.join("data", file.name)
        with open(path, "wb") as f:
            f.write(file.getbuffer())
        human_paths.append(path)
else:
    # Fallback to latest human dataset in data/
    local_files = sorted([f for f in os.listdir("data") if f.endswith("_expression.csv")], reverse=True)
    if local_files:
        latest_file = os.path.join("data", local_files[0])
        human_paths = [latest_file]
        st.info(f"Using latest local expression dataset: {os.path.basename(latest_file)}")
    else:
        st.warning("No uploaded or local expression datasets found.")

# --- Train model ---
if human_paths:
    st.markdown("## Step 3: Train Model on Data")
    label_col = st.text_input("Name of binary label column (leave blank to auto-detect)")
    all_human_dfs = []
    all_labels = []
    for path in human_paths:
        df = pd.read_csv(path, index_col=0)
        y = df[label_col] if label_col in df.columns else None
        if y is not None:
            df = df.drop(columns=[label_col])
            all_human_dfs.append(df)
            all_labels.append(y)

    if all_labels:
        X_all = pd.concat(all_human_dfs)
        y_all = pd.concat(all_labels)
        model, metrics = train_model(X_all, y_all)
        st.json(metrics)

        st.markdown("## Step 4: Evaluate on Animal Models")
        try:
            animal_files = list_animal_datasets("animal_models")
            animal_datasets = load_multiple_datasets(animal_files)
            leaderboard = []
            for animal_name, animal_df in animal_datasets.items():
                try:
                    preds, auc = test_model_on_dataset(model, animal_df, return_auc=True)
                    shap_vec = extract_shap_values(model, animal_df)
                    similarity = compare_shap_vectors(shap_vec, shap_vec)
                    leaderboard.append({
                        "Dataset": animal_name,
                        "AUC": auc,
                        "SHAP Similarity": similarity
                    })
                except Exception as e:
                    st.warning(f"Failed on {animal_name}: {e}")
            st.dataframe(pd.DataFrame(leaderboard))
        except FileNotFoundError:
            st.error("No animal model folder or datasets found. Please use the GEO search to download models.")
