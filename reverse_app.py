import os
import sys
import streamlit as st
import pandas as pd
import numpy as np

# Extend sys path to access src/
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from preprocessing import preprocess_dataset
from model_training import train_model
from prediction import test_model_on_dataset
from ortholog_mapper import map_human_to_model_genes
from explainability import extract_shap_values, compare_shap_vectors
from reverse_modeling import list_animal_datasets
from data_fetcher import dataset_search_ui

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

st.set_page_config(page_title="HemoLink_AI – Reverse Modeling", layout="wide")

# -------------------- HEADER --------------------
st.markdown("""
    <div style="display: flex; align-items: center; gap: 20px;">
        <img src="https://raw.githubusercontent.com/kzif93/HemoLink_AI/main/assets/logo.png" width="200">
        <div>
            <h1 style="margin-bottom: 0;">HemoLink_AI</h1>
            <h3 style="margin-top: 0; color: #ccc;">🔁 Reverse Modeling – Match Human Signatures to Animal Models</h3>
        </div>
    </div>
""", unsafe_allow_html=True)

# -------------------- GEO UI --------------------
dataset_search_ui()

st.markdown("### 🧬 Step 1: Choose Human Dataset(s)")

# Find all downloaded GEO expression datasets
expression_files = [f for f in os.listdir("data") if f.endswith("_expression.csv")]
selected_files = st.multiselect("Select one or more human datasets:", expression_files)

# Helper function to load and label datasets
def load_and_label_human_dataset(file):
    df = pd.read_csv(os.path.join("data", file), index_col=0).T  # samples as rows
    potential_labels = [col for col in df.columns if col.lower() in ["disease", "condition", "status", "phenotype"] or df[col].nunique() <= 5]

    label_col = None
    y = None
    if potential_labels:
        label_col = st.selectbox(f"Select label column for {file}", potential_labels, key=file)
        y = df[label_col].replace({"control": 0, "healthy": 0, "case": 1, "stroke": 1}).astype(int)
        df = df.drop(columns=[label_col])
    else:
        st.warning(f"⚠️ No clear label column found in {file}. Skipping.")
    return df, y

X_list = []
y_list = []

if selected_files:
    for file in selected_files:
        df, y = load_and_label_human_dataset(file)
        if df is not None and y is not None:
            X_list.append(df)
            y_list.append(y)

if X_list and y_list:
    X_human = pd.concat(X_list, axis=0, join="inner")
    y_human = pd.concat(y_list, axis=0)

    st.success(f"✅ Combined {len(X_list)} dataset(s), shape: {X_human.shape}")

    # ------------------ MODEL ------------------
    st.markdown("### 🧠 Step 2: Train Human Model")
    model_choice = st.selectbox("Select model:", ["RandomForest", "XGBoost", "LogisticRegression"])

    if model_choice == "RandomForest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_choice == "XGBoost":
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    elif model_choice == "LogisticRegression":
        model = LogisticRegression(max_iter=1000)

    model.fit(X_human, y_human)

    # ------------------ EVALUATION ------------------
    st.markdown("### 🐭 Step 3: Evaluate on Animal Datasets")
    animal_files = list_animal_datasets("animal_models")
    selected_animals = st.multiselect("Select animal datasets to test on:", animal_files, default=animal_files)

    results = []
    for file in selected_animals:
        try:
            animal_df = pd.read_csv(os.path.join("animal_models", file))
            shared_genes, X_animal = map_human_to_model_genes(
                human_genes=X_human.columns,
                animal_df=animal_df,
                ortholog_path="data/mouse_to_human_orthologs.csv",
                filename_hint=file
            )

            if len(shared_genes) < 10:
                continue

            auc_score, _ = test_model_on_dataset(model, X_animal[shared_genes])
            shap_human = extract_shap_values(model, X_human[shared_genes])
            shap_animal = extract_shap_values(model, X_animal[shared_genes])
            shap_similarity = compare_shap_vectors(shap_human, shap_animal)

            results.append({
                "Animal Model": file,
                "Shared Genes": len(shared_genes),
                "AUC": round(auc_score, 3),
                "SHAP Similarity": round(shap_similarity, 3)
            })

        except Exception as e:
            st.warning(f"⚠️ Skipping {file}: {e}")

    if results:
        st.markdown("### 📊 Results")
        st.dataframe(pd.DataFrame(results).sort_values(by="SHAP Similarity", ascending=False))
    else:
        st.info("No compatible animal datasets found.")
