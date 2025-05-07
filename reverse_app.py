import os
import sys
import streamlit as st
import pandas as pd
import numpy as np

# Add src path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# --- Import HemoLink modules ---
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

# --- Page Config ---
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

st.markdown("### 🧬 Step 1: Upload Human Labels or Select a Column")

uploaded_labels = st.file_uploader("📂 Upload CSV with sample labels", type=["csv"])
label_col = st.text_input("🔠 Name of binary label column (e.g. 'disease')")

# -------------------- Load Expression --------------------
st.info("📂 Looking for latest downloaded GEO expression file...")

try:
    # Get latest human expression file
    files = [f for f in os.listdir("data") if f.endswith("_expression.csv")]
    files.sort(reverse=True)
    latest_file = os.path.join("data", files[0])
    human_df = pd.read_csv(latest_file, index_col=0)

    st.success(f"✅ Using human dataset: {os.path.basename(latest_file)}")

    # Load labels if provided
    if uploaded_labels is not None:
        labels_df = pd.read_csv(uploaded_labels)
        if label_col not in labels_df.columns:
            st.error(f"Column '{label_col}' not found in uploaded file.")
            st.stop()
        y_human = labels_df[label_col]
    elif label_col in human_df.columns:
        y_human = human_df[label_col]
        human_df = human_df.drop(columns=[label_col])
    else:
        st.warning("⚠️ Please provide a valid label column.")
        st.stop()

    # Preprocess
    X_human, y_human = preprocess_dataset(human_df, label_col=None)

    # Model choice
    st.markdown("### 🧠 Step 2: Train Human Model")
    model_choice = st.selectbox("Select a model", ["RandomForest", "XGBoost", "LogisticRegression"])

    if model_choice == "RandomForest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_choice == "XGBoost":
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    elif model_choice == "LogisticRegression":
        model = LogisticRegression(max_iter=1000)

    model.fit(X_human, y_human)

    # Animal model evaluation
    st.markdown("### 🧪 Step 3: Evaluate on Animal Datasets")
    results = []
    for file in list_animal_datasets("animal_models"):
        try:
            animal_df = pd.read_csv(os.path.join("animal_models", file))
            shared_genes, X_animal = map_human_to_model_genes(
                human_genes=X_human.columns,
                animal_df=animal_df,
                ortholog_path="data/mouse_to_human_orthologs.csv",
                filename_hint=file
            )

            if len(shared_genes) < 10:
                st.warning(f"⚠️ Skipping {file}: too few shared genes.")
                continue

            auc, y_pred = test_model_on_dataset(model, X_animal[shared_genes])
            shap_human = extract_shap_values(model, X_human[shared_genes])
            shap_animal = extract_shap_values(model, X_animal[shared_genes])
            shap_similarity = compare_shap_vectors(shap_human, shap_animal)

            results.append({
                "Animal Model File": file,
                "Shared Genes": len(shared_genes),
                "AUC": round(auc, 3),
                "SHAP Similarity": round(shap_similarity, 3),
            })
        except Exception as e:
            st.warning(f"⚠️ Error processing {file}: {e}")
            continue

    if results:
        st.markdown("### 📊 Model Transferability Leaderboard")
        st.dataframe(pd.DataFrame(results).sort_values(by="SHAP Similarity", ascending=False))
    else:
        st.info("No compatible animal datasets found.")

except Exception as e:
    st.error("❌ Something went wrong during data processing.")
    st.exception(e)
