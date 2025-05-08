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

# Curated datasets (standard structure)
curated_registry = {
  "stroke": [
    {"GSE": "GSE16561", "Organism": "Human", "Model": "Ischemic Stroke", "Platform": "GPL570", "Description": "Whole blood transcriptome profiling of stroke patients.", "Tag": "⭐ Curated"},
    {"GSE": "GSE22255", "Organism": "Human", "Model": "Stroke", "Platform": "GPL570", "Description": "PBMCs from stroke vs controls.", "Tag": "⭐ Curated"},
    {"GSE": "GSE58294", "Organism": "Human", "Model": "Stroke onset", "Platform": "Microarray", "Description": "Gene expression in acute stroke.", "Tag": "⭐ Curated"},
    {"GSE": "GSE37587", "Organism": "Human", "Model": "Stroke outcomes", "Platform": "Microarray", "Description": "Recovery dynamics post-stroke.", "Tag": "⭐ Curated"},
    {"GSE": "GSE162955", "Organism": "Human", "Model": "Cerebral ischemia", "Platform": "RNA-Seq", "Description": "Whole blood RNA-seq in ischemic stroke.", "Tag": "⭐ Curated"},
    {"GSE": "GSE233813", "Organism": "Mouse", "Model": "MCAO (24h)", "Platform": "RNA-Seq", "Description": "High-quality RNA-seq of mouse brain after stroke.", "Tag": "⭐ Curated"},
    {"GSE": "GSE162072", "Organism": "Rat", "Model": "tMCAO (3h)", "Platform": "Microarray", "Description": "Early transcriptomic changes in stroke.", "Tag": "⭐ Curated"},
    {"GSE": "GSE137482", "Organism": "Mouse", "Model": "Photothrombosis", "Platform": "RNA-Seq", "Description": "Cortex transcriptome 24h post-stroke.", "Tag": "⭐ Curated"},
    {"GSE": "GSE36010 / GSE78731", "Organism": "Rat", "Model": "Stroke Meta-analysis", "Platform": "Microarray", "Description": "Integrated rat model studies.", "Tag": "⭐ Curated"},
    {"GSE": "PMC10369109", "Organism": "Rabbit", "Model": "Spatial vascular study", "Platform": "Visium", "Description": "Spatial transcriptomics of brain vessels.", "Tag": "⭐ Curated"}
  ],
  "vte": [
    {"GSE": "GSE19151", "Organism": "Human", "Model": "Idiopathic VTE", "Platform": "Microarray", "Description": "Whole blood gene profiles in VTE patients.", "Tag": "⭐ Curated"},
    {"GSE": "GSE48000", "Organism": "Human", "Model": "Factor V Leiden", "Platform": "Microarray", "Description": "Asymptomatic carriers of thrombophilic mutations.", "Tag": "⭐ Curated"},
    {"GSE": "GSE17078", "Organism": "Human", "Model": "VTE Immune", "Platform": "Microarray", "Description": "Leukocyte transcriptomics in VTE.", "Tag": "⭐ Curated"},
    {"GSE": "GSE26787", "Organism": "Human", "Model": "APS + VTE", "Platform": "Microarray", "Description": "Comparative APS/VTE/healthy profiles.", "Tag": "⭐ Curated"},
    {"GSE": "GSE158312", "Organism": "Human", "Model": "PE vs VTE", "Platform": "RNA-Seq", "Description": "EV miRNA in PE and VTE.", "Tag": "⭐ Curated"},
    {"GSE": "GSE125965", "Organism": "Mouse", "Model": "IVC stenosis", "Platform": "Microarray", "Description": "Mouse DVT vein wall transcriptomics.", "Tag": "⭐ Curated"},
    {"GSE": "GSE145993", "Organism": "Mouse", "Model": "FeCl3 thrombosis", "Platform": "RNA-Seq", "Description": "Endothelial response to FeCl3 DVT.", "Tag": "⭐ Curated"},
    {"GSE": "GSE245276", "Organism": "Mouse", "Model": "Anti-P-selectin DVT", "Platform": "RNA-Seq", "Description": "Antibody treatment effects in DVT.", "Tag": "⭐ Curated"},
    {"GSE": "GSE46265", "Organism": "Rat", "Model": "IVC ligation", "Platform": "Microarray", "Description": "Post-ligation gene changes in rats.", "Tag": "⭐ Curated"}
  ],
  "aps": [
    {"GSE": "GSE50395", "Organism": "Human", "Model": "APS vs Healthy", "Platform": "Microarray", "Description": "Large-scale whole blood study.", "Tag": "⭐ Curated"},
    {"GSE": "GSE26787", "Organism": "Human", "Model": "APS, VTE, Healthy", "Platform": "Microarray", "Description": "Shared and distinct APS/VTE expression.", "Tag": "⭐ Curated"},
    {"GSE": "GSE123116", "Organism": "Human", "Model": "Primary APS", "Platform": "RNA-Seq", "Description": "Monocyte transcriptomics in APS.", "Tag": "⭐ Curated"},
    {"GSE": "GSE219228", "Organism": "Human", "Model": "APS Pregnancy", "Platform": "RNA-Seq", "Description": "PBMCs from pregnant APS patients.", "Tag": "⭐ Curated"},
    {"GSE": "GSE168136", "Organism": "Human", "Model": "APS vs SLE-APS", "Platform": "RNA-Seq", "Description": "Comparison of APS and lupus-related APS.", "Tag": "⭐ Curated"},
    {"GSE": "GSE139342", "Organism": "Mouse", "Model": "Obstetric APS", "Platform": "RNA-Seq", "Description": "Placenta and fetal brain profiles.", "Tag": "⭐ Curated"},
    {"GSE": "GSE99329", "Organism": "Mouse", "Model": "Thrombotic APS", "Platform": "Microarray", "Description": "aPL + LPS 2nd-hit APS model.", "Tag": "⭐ Curated"},
    {"GSE": "GSE172935", "Organism": "Mouse", "Model": "Pregnancy APS", "Platform": "RNA-Seq", "Description": "Gestation stage–specific profiles.", "Tag": "⭐ Curated"},
    {"GSE": "GSE61616", "Organism": "Rat", "Model": "Immune Vascular Injury", "Platform": "Microarray", "Description": "LPS-enhanced vascular thrombosis.", "Tag": "⭐ Curated"}
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
    try:
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
    except Exception as e:
        st.error(f"❌ Failed to load curated datasets: {e}")
else:
    st.info("No curated datasets matched your keyword. Try 'stroke', 'VTE', or 'APS'.")

# Smart GEO Search
st.markdown("### 🔍 Smart Animal GEO Dataset Discovery")
if st.button("Run smart search"):
    try:
        results = smart_search_animal_geo(query, species_input)
        if results is not None and not pd.DataFrame(results).empty:
            results_df = pd.DataFrame(results)
            st.write("Smart search result columns:", results_df.columns.tolist())
            if "Organism" in results_df.columns:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Found Animal Datasets**")
                    st.dataframe(results_df[results_df["Organism"] != "Homo sapiens"].reset_index(drop=True))
                with col2:
                    st.markdown("**Found Human Datasets**")
                    st.dataframe(results_df[results_df["Organism"] == "Homo sapiens"].reset_index(drop=True))
            else:
                st.warning("⚠️ 'Organism' column not found in smart search results.")
        else:
            st.warning("No datasets found.")
    except Exception as e:
        st.error(f"Search failed: {e}")
