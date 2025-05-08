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
    smart_animal_dataset_search_ui,
    smart_search_animal_geo,
    download_animal_dataset,
    extract_keywords_from_query
)

st.set_page_config(page_title="HemoLink_AI – Reverse Modeling", layout="wide")

st.markdown("""
    <h1 style='margin-bottom: 5px;'>Reverse Modeling – Match Human Data to Animal Models</h1>
    <p style='color: gray;'>Upload your own dataset or search for GEO datasets to train on multiple human datasets and evaluate against multiple animal models.</p>
""", unsafe_allow_html=True)

# --- QUERY DETECTION ---
st.markdown("## Step 1: Search for Human or Animal Datasets")
query = st.text_input("Enter disease keyword (e.g., stroke, thrombosis, APS):", value="stroke")
organism = st.text_input("Species (optional, e.g., Mus musculus):", value="")

# --- CURATED DATASETS BY DOMAIN ---
curated_registry = {
    "stroke": [
        {"GSE": "GSE233813", "Organism": "Mouse", "Model": "MCAO (24h)", "Platform": "RNA-Seq", "Description": "Brain RNA-seq in mice post-MCAO"},
        {"GSE": "GSE162072", "Organism": "Rat", "Model": "tMCAO (3h)", "Platform": "Microarray", "Description": "Vascular transcriptomics in rats"},
        {"GSE": "GSE137482", "Organism": "Mouse", "Model": "Photothrombosis", "Platform": "RNA-Seq", "Description": "Ischemic cortex gene expression"},
        {"GSE": "PMC10369109", "Organism": "Rabbit", "Model": "Spatial vascular", "Platform": "Visium", "Description": "Spatial transcriptomics of vessels"},
        {"GSE": "GSE36010 / GSE78731", "Organism": "Rat", "Model": "MCAO (meta-analysis)", "Platform": "Microarray", "Description": "Integrated stroke meta-datasets"},
        {"GSE": "GSE16561", "Organism": "Human", "Model": "Ischemic stroke", "Platform": "Microarray", "Description": "Whole blood profiling"},
        {"GSE": "GSE22255", "Organism": "Human", "Model": "Stroke vs control", "Platform": "Microarray", "Description": "PBMCs from stroke patients"},
        {"GSE": "GSE58294", "Organism": "Human", "Model": "Acute stroke", "Platform": "Microarray", "Description": "Blood transcriptional profiling"},
        {"GSE": "GSE37587", "Organism": "Human", "Model": "Stroke outcomes", "Platform": "Microarray", "Description": "Post-stroke recovery data"},
        {"GSE": "GSE162955", "Organism": "Human", "Model": "Cerebral ischemia", "Platform": "RNA-Seq", "Description": "Gene expression in ischemia"}
    ],
    "vte": [
        {"GSE": "GSE125965", "Organism": "Mouse", "Model": "IVC stenosis DVT", "Platform": "Microarray", "Description": "Mouse vein wall and blood DVT data"},
        {"GSE": "GSE19151", "Organism": "Mouse", "Model": "TF-induced thrombosis", "Platform": "Microarray", "Description": "TF-driven procoagulant signatures"},
        {"GSE": "GSE145993", "Organism": "Mouse", "Model": "FeCl₃ thrombosis", "Platform": "RNA-Seq", "Description": "Endothelial transcription after FeCl₃"},
        {"GSE": "GSE245276", "Organism": "Mouse", "Model": "Anti-P-selectin", "Platform": "RNA-Seq", "Description": "Treatment effects on thrombosis"},
        {"GSE": "GSE46265", "Organism": "Rat", "Model": "IVC ligation", "Platform": "Microarray", "Description": "Rat transcriptome post-ligation"},
        {"GSE": "GSE19151", "Organism": "Human", "Model": "VTE vs control", "Platform": "Microarray", "Description": "Blood profiling in idiopathic VTE"},
        {"GSE": "GSE48000", "Organism": "Human", "Model": "Factor V Leiden", "Platform": "Microarray", "Description": "Mutation-driven thrombosis risk"},
        {"GSE": "GSE17078", "Organism": "Human", "Model": "VTE leukocytes", "Platform": "Microarray", "Description": "Immune profiles in VTE"},
        {"GSE": "GSE26787", "Organism": "Human", "Model": "APS/VTE comparison", "Platform": "Microarray", "Description": "Coagulopathy patient groups"},
        {"GSE": "GSE158312", "Organism": "Human", "Model": "EV miRNAs in PE", "Platform": "RNA-Seq", "Description": "EVs from VTE and PE patients"}
    ],
    "aps": [
        {"GSE": "GSE139342", "Organism": "Mouse", "Model": "Obstetric APS", "Platform": "RNA-Seq", "Description": "Placenta and brain in anti-β2GPI"},
        {"GSE": "GSE99329", "Organism": "Mouse", "Model": "aPL + LPS thrombosis", "Platform": "Microarray", "Description": "Inflammatory APS model"},
        {"GSE": "GSE172935", "Organism": "Mouse", "Model": "Pregnancy APS", "Platform": "RNA-Seq", "Description": "Placental stages in APS mice"},
        {"GSE": "GSE61616", "Organism": "Rat", "Model": "LPS-enhanced injury", "Platform": "Microarray", "Description": "Thrombo-inflammatory injury"},
        {"GSE": "PRJNA640011 (SRA)", "Organism": "Macaque", "Model": "LPS coagulopathy", "Platform": "RNA-Seq", "Description": "Systemic vascular response"},
        {"GSE": "GSE50395", "Organism": "Human", "Model": "APS vs healthy", "Platform": "Microarray", "Description": "Whole blood APS profiles"},
        {"GSE": "GSE26787", "Organism": "Human", "Model": "APS/VTE/healthy", "Platform": "Microarray", "Description": "Coagulopathy comparison"},
        {"GSE": "GSE123116", "Organism": "Human", "Model": "Primary APS monocytes", "Platform": "RNA-Seq", "Description": "Immune cell transcriptomics"},
        {"GSE": "GSE219228", "Organism": "Human", "Model": "APS pregnancy", "Platform": "RNA-Seq", "Description": "PBMCs from pregnant APS patients"},
        {"GSE": "GSE168136", "Organism": "Human", "Model": "APS vs SLE-APS", "Platform": "RNA-Seq", "Description": "Autoimmune coagulopathy comparison"}
    ]
}

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

# --- Show matching curated datasets ---
st.markdown("## Matched Curated Datasets")
if selected_domain:
    domain_df = pd.DataFrame(curated_registry[selected_domain])
    st.write(f"**{selected_domain.upper()} curated datasets**")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Animal**")
        st.dataframe(domain_df[domain_df["Organism"] != "Human"].reset_index(drop=True))
    with col2:
        st.markdown("**Human**")
        st.dataframe(domain_df[domain_df["Organism"] == "Human"].reset_index(drop=True))
else:
    st.info("No curated datasets matched your keyword. Try 'stroke', 'VTE', or 'APS'.")

# --- GEO search ---
smart_animal_dataset_search_ui()

# --- Upload human datasets ---
st.markdown("## Step 2: Upload Your Human Dataset(s)")
uploaded_files = st.file_uploader("Upload one or more human expression CSV files:", type=["csv"], accept_multiple_files=True)
human_files = uploaded_files if uploaded_files else []
human_paths = []

if human_files:
    for file in human_files:
        path = os.path.join("data", file.name)
        with open(path, "wb") as f:
            f.write(file.getbuffer())
        human_paths.append(path)

# --- Train model ---
if human_paths:
    st.markdown("## Step 3: Train Model on Human Dataset(s)")
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
