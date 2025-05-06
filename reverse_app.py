import streamlit as st
import pandas as pd
import os
import sys

# Extend sys.path to include the src folder
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Imports from src
from model_training import train_model
from prediction import test_model_on_dataset
from ortholog_mapper import map_human_to_model_genes
from explainability import extract_shap_values, compare_shap_vectors
from reverse_modeling import list_animal_datasets
from preprocessing import preprocess_dataset

# --- App Config ---
st.set_page_config(
    page_title="HemoLink_AI – Reverse Modeling",
    layout="wide",
    page_icon="🧬",
)

st.title("🧬 Reverse Modeling – Match Human Data to Animal Models")

st.markdown("""
Upload a **human transcriptomic dataset** and binary labels. HemoLink_AI will evaluate all available animal models and rank them by:
- Gene ortholog overlap
- Predictive performance (AUC)
- SHAP feature importance similarity
""")

# --- File Upload ---
uploaded_file = st.file_uploader("📄 Upload human expression CSV", type=["csv"])
label_col = st.text_input("🔠 Name of the binary label column (e.g. 'disease')")

if uploaded_file and label_col:
    try:
        # Load and preprocess human dataset
        human_df = pd.read_csv(uploaded_file)
        X_human, y_human = preprocess_dataset(human_df, label_col)

        # Train model on human data
        model = train_model(X_human, y_human)

        # Locate animal datasets
        model_folder = "animal_models"
        animal_files = list_animal_datasets(model_folder)

        st.subheader("📊 Evaluation Results")
        results = []

        for file in animal_files:
            try:
                animal_path = os.path.join(model_folder, file)
                animal_df = pd.read_csv(animal_path)

                # Automatically detect species from filename and map orthologs
                shared_genes, X_animal = map_human_to_model_genes(
                    human_genes=X_human.columns,
                    animal_df=animal_df,
                    ortholog_path='data/mouse_to_human_orthologs.csv',
                    filename_hint=file
                )

                if len(shared_genes) < 10:
                    st.warning(f"⚠️ Skipping {file}: Only {len(shared_genes)} shared genes.")
                    continue

                # Predict and evaluate AUC
                auc_score, y_pred = test_model_on_dataset(model, X_animal[shared_genes])

                # Compare SHAP values
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
            st.info("No suitable animal models found with sufficient shared orthologs.")

    except Exception as e:
        st.error(f"❌ Error: {e}")
