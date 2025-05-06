import streamlit as st
import pandas as pd
import os
import sys

# Extend sys.path to access src/ folder
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Imports from src
from model_training import train_model
from prediction import test_model_on_dataset
from ortholog_mapper import map_human_to_model_genes
from explainability import extract_shap_values, compare_shap_vectors
from reverse_modeling import list_animal_datasets
from preprocessing import preprocess_dataset
from geo_utils import search_geo_datasets, fetch_geo_platform_matrix

# --- App Config ---
st.set_page_config(
    page_title="HemoLink_AI – Reverse Modeling",
    layout="wide",
    page_icon="🧬",
)

st.title("🧬 Reverse Modeling – Match Human Data to Animal Models")

st.markdown("""
Upload your own dataset or search for **GEO datasets** to train on **multiple human datasets** and evaluate against **multiple animal models**.
""")

# --- Sidebar: GEO Search & Selection ---
with st.sidebar:
    st.markdown("🔍 **Search GEO for Human Datasets**")
    disease_query = st.text_input("Disease keyword (e.g., stroke, thrombosis)")
    selected_geo_ids = []

    if disease_query:
        with st.spinner("Searching GEO..."):
            geo_ids = search_geo_datasets(disease_query, max_results=10)
        if geo_ids:
            selected_geo_ids = st.multiselect("Select GEO dataset(s)", geo_ids)
            if st.button("Download selected datasets"):
                st.session_state["geo_datasets"] = []
                for geo_id in selected_geo_ids:
                    try:
                        df = fetch_geo_platform_matrix(geo_id)
                        df["source"] = geo_id  # for tracking
                        st.session_state["geo_datasets"].append(df)
                        st.success(f"✓ Loaded {geo_id}")
                    except Exception as e:
                        st.warning(f"⚠️ Failed: {geo_id} – {e}")
        else:
            st.info("No datasets found.")

# --- File Upload (optional) ---
uploaded_file = st.file_uploader("📄 Or upload a single human dataset CSV", type=["csv"])
label_col = st.text_input("🔠 Name of binary label column (e.g. 'disease')")

# --- Combine Human Datasets ---
combined_df = None

if "geo_datasets" in st.session_state and st.session_state["geo_datasets"]:
    combined_df = pd.concat(st.session_state["geo_datasets"], axis=0, join="inner")
    st.success(f"✅ Combined {len(st.session_state['geo_datasets'])} GEO datasets")
elif uploaded_file:
    combined_df = pd.read_csv(uploaded_file)

# --- Run Modeling Pipeline ---
if combined_df is not None and label_col:
    try:
        X_human, y_human = preprocess_dataset(combined_df, label_col)

        # --- Select Model ---
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        import xgboost as xgb

        model_choice = st.selectbox("🧠 Select a model", ["RandomForest", "XGBoost", "LogisticRegression"])

        if model_choice == "RandomForest":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_choice == "XGBoost":
            model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        elif model_choice == "LogisticRegression":
            model = LogisticRegression(max_iter=1000)
        else:
            st.error("Invalid model selected.")
            st.stop()

        model.fit(X_human, y_human)

        # --- Animal Model Evaluation ---
        model_folder = "animal_models"
        animal_files = list_animal_datasets(model_folder)

        st.subheader("📊 Evaluation Results")
        results = []

        for file in animal_files:
            try:
                animal_path = os.path.join(model_folder, file)
                animal_df = pd.read_csv(animal_path)

                # Detect species and align orthologs
                shared_genes, X_animal = map_human_to_model_genes(
                    human_genes=X_human.columns,
                    animal_df=animal_df,
                    ortholog_path='data/mouse_to_human_orthologs.csv',
                    filename_hint=file
                )

                if len(shared_genes) < 10:
                    st.warning(f"⚠️ Skipping {file}: Only {len(shared_genes)} shared genes.")
                    continue

                # Evaluate
                auc_score, y_pred = test_model_on_dataset(model, X_animal[shared_genes])
                shap_human = extract_shap_values(model, X_human[shared_genes])
                shap_animal = extract_shap_values(model, X_animal[shared_genes])
                shap_similarity = compare_shap_vectors(shap_human, shap_animal)

                results.append({
                    "Animal Model File": file,
                    "Detected Species": file.split("_")[-1].replace(".csv", ""),
                    "Shared Genes": len(shared_genes),
                    "AUC": round(auc_score, 3),
                    "SHAP Similarity": round(shap_similarity, 3),
                })

            except Exception as e:
                st.warning(f"⚠️ Skipping {file}: {e}")
                continue

        if results:
            result_df = pd.DataFrame(results).sort_values(by="SHAP Similarity", ascending=False)
            st.dataframe(result_df)
        else:
            st.info("No valid animal model results.")

    except Exception as e:
        st.error(f"❌ Pipeline error: {e}")
