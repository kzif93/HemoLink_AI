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
  "stroke": [...],
  "vte": [...],
  "aps": [...]
}  # Actual dataset entries defined previously

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
    domain_df.columns = domain_df.columns.str.strip()
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Curated Animal Datasets**")
        st.dataframe(domain_df[domain_df["Organism"] != "Human"].reset_index(drop=True))
    with col2:
        st.markdown("**Curated Human Datasets**")
        st.dataframe(domain_df[domain_df["Organism"] == "Human"].reset_index(drop=True))
else:
    st.info("No curated datasets matched your keyword. Try 'stroke', 'VTE', or 'APS'.")

# Smart GEO Search
st.markdown("### 🔍 Smart Animal GEO Dataset Discovery")
if st.button("Run smart search"):
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
        st.warning("No datasets found.")

# --- Step 2: Upload (Optional) ---
st.markdown("## Step 2: Upload Dataset(s) for Training")
st.markdown("This step is optional. If nothing is uploaded, the latest dataset from /data/ will be used.")
uploaded_files = st.file_uploader("Upload one or more expression CSV files (training data):", type=["csv"], accept_multiple_files=True)
human_paths = []

if uploaded_files:
    for f in uploaded_files:
        dest = os.path.join("data", f.name)
        with open(dest, "wb") as out:
            out.write(f.getbuffer())
        human_paths.append(dest)
else:
    candidates = sorted([f for f in os.listdir("data") if f.endswith("_expression.csv")], reverse=True)
    if candidates:
        fallback = os.path.join("data", candidates[0])
        human_paths = [fallback]
        st.info(f"Using fallback file: {os.path.basename(fallback)}")
    else:
        st.warning("No available expression files found.")

# --- Step 3: Train model ---
if human_paths:
    st.markdown("## Step 3: Train Model on Human Dataset(s)")
    label_col = st.text_input("Label column (binary, leave empty to auto-detect):")
    dfs = []
    labels = []
    for path in human_paths:
        df = pd.read_csv(path, index_col=0)
        if label_col and label_col in df.columns:
            y = df[label_col]
            df = df.drop(columns=[label_col])
        else:
            st.warning(f"No label found in {os.path.basename(path)}. Skipping.")
            continue
        dfs.append(df)
        labels.append(y)

    if dfs:
        X = pd.concat(dfs)
        y = pd.concat(labels)
        model, metrics = train_model(X, y)
        st.json(metrics)

        # --- Step 4: Evaluate ---
        st.markdown("## Step 4: Evaluate on Animal Models")
        try:
            animal_files = list_animal_datasets("animal_models")
            animal_data = load_multiple_datasets(animal_files)
            leaderboard = []
            for name, animal_df in animal_data.items():
                try:
                    preds, auc = test_model_on_dataset(model, animal_df, return_auc=True)
                    shap_vec = extract_shap_values(model, animal_df)
                    sim = compare_shap_vectors(shap_vec, shap_vec)
                    leaderboard.append({"Dataset": name, "AUC": auc, "SHAP Similarity": sim})
                except Exception as e:
                    st.error(f"Failed on {name}: {e}")
            st.dataframe(pd.DataFrame(leaderboard))
        except Exception as e:
            st.error(f"Animal model loading failed: {e}")
