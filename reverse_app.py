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
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Found Animal Datasets**")
                st.dataframe(results_df[results_df["organism"] != "Homo sapiens"].reset_index(drop=True))
            with col2:
                st.markdown("**Found Human Datasets**")
                st.dataframe(results_df[results_df["organism"] == "Homo sapiens"].reset_index(drop=True))
        else:
            st.warning("No datasets found.")
    except Exception as e:
        st.error(f"Search failed: {e}")
