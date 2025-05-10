
import os
import sys
import streamlit as st
import pandas as pd
import re
from Bio import Entrez
from datetime import datetime

# Ensure src/ is in path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from preprocessing import preprocess_dataset
from prediction import test_model_on_dataset
from explainability import extract_shap_values, compare_shap_vectors
from reverse_modeling import list_animal_datasets
from curated_sets import curated_registry

Entrez.email = "your_email@example.com"
KEYWORDS = ["stroke", "ischemia", "thrombosis", "vte", "dvt", "aps", "antiphospholipid", "control", "healthy", "normal"]

from typing import List, Tuple
def load_multiple_datasets(gse_list: List[str]) -> Tuple[pd.DataFrame, pd.Series]:
    dfs = []
    labels_list = []
    for gse in gse_list:
        exp_path = os.path.join("data", f"{gse}_expression.csv")
        label_path = os.path.join("data", f"{gse}_labels.csv")
        if not os.path.exists(exp_path) or not os.path.exists(label_path):
            return None
        try:
            df = pd.read_csv(exp_path, index_col=0).T
        except Exception as e:
            return None
        try:
            labels = pd.read_csv(label_path, index_col=0).squeeze()
        except Exception as e:
            return None
        dfs.append(df)
        labels_list.append(labels)
    try:
        full_df = pd.concat(dfs, axis=1)
        full_labels = pd.concat(labels_list)
        return full_df, full_labels
    except Exception as e:
        return None

def extract_keywords_from_query(query):
    return [w.strip().lower() for w in re.split(r"[\s,]+", query)]

def smart_search_animal_geo(query, species=None, max_results=100):
    try:
        keywords = extract_keywords_from_query(query)
        search_term = f"{' OR '.join(keywords)} AND (gse[ETYP] OR gds[ETYP])"
        if species:
            search_term += f" AND {species}"
        handle = Entrez.esearch(db="gds", term=search_term, retmax=max_results)
        record = Entrez.read(handle)
        ids = record["IdList"]
        summaries = []
        for gds_id in ids:
            summary = Entrez.esummary(db="gds", id=gds_id)
            docsum = Entrez.read(summary)[0]
            summaries.append({
                "GSE": docsum.get("Accession", "?"),
                "Title": docsum.get("title", "?"),
                "Description": docsum.get("summary", "?"),
                "Samples": docsum.get("n_samples", "?"),
                "Platform": docsum.get("gpl", "?"),
                "Organism": docsum.get("taxon", "?"),
                "ReleaseDate": docsum.get("PDAT", "?"),
                "Score": 0,
                "Tag": "GEO"
            })
        return summaries
    except Exception as e:
        return []

def download_and_prepare_dataset(gse):
    import GEOparse
    from probe_mapper import download_platform_annotation, map_probes_to_genes
    out_path = os.path.join("data", f"{gse}_expression.csv")
    label_out = os.path.join("data", f"{gse}_labels.csv")
    if os.path.exists(out_path):
        return out_path
    geo = GEOparse.get_GEO(geo=gse, destdir="data", annotate_gpl=True)
    gpl_name = list(geo.gpls.keys())[0] if geo.gpls else None
    df = pd.DataFrame({gsm: sample.table.set_index("ID_REF")["VALUE"] for gsm, sample in geo.gsms.items()})
    df.to_csv(out_path)
    if df.index.str.endswith("_at").sum() / len(df.index) > 0.5 and gpl_name:
        gpl_path = download_platform_annotation(gse)
        mapped = map_probes_to_genes(out_path, gpl_path)
        mapped = mapped.T
        mapped.index = df.columns
        mapped.index.name = "Sample"
        mapped.to_csv(out_path)
    try:
        sample_titles = pd.Series({gsm: sample.metadata.get("title", [""])[0] for gsm, sample in geo.gsms.items()})
        labels = sample_titles.str.lower().map(lambda x: 1 if "pbmcs_is" in x else (0 if "pbmcs_control" in x else None)).dropna()
        if labels.nunique() == 2:
            labels.name = "label"
            labels.to_csv(label_out)
            st.success("✅ Labels generated.")
            return out_path
        else:
            labels = pd.Series([0] * df.shape[1], index=df.columns, name="label")
            labels.to_csv(label_out)
    except Exception as e:
        st.error(f"❌ Labeling failed: {e}")
    return out_path
    
def train_model(X, y):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score, classification_report
    import numpy as np
    import streamlit as st

    try:
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = y.values.ravel()
        y = np.asarray(y).astype(int)

        if isinstance(X, pd.DataFrame):
            X = X.values

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)

        preds = model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, preds)

        y_pred = (preds > 0.5).astype(int)

        # FORCE y and y_pred to be flat arrays
        y = np.array(y).flatten()
        y_pred = np.array(y_pred).flatten()

        report = classification_report(y, y_pred, output_dict=True)

        metrics = {
            "roc_auc": round(auc, 4),
            "classification_report": report,
        }
        return model, metrics

    except Exception as e:
        import traceback
        st.error("❌ Training failed!")
        st.text(traceback.format_exc())
        raise RuntimeError(f"Training failed: {e}")
# ---- STREAMLIT UI ----
st.set_page_config(page_title="HemoLink_AI – Reverse Modeling", layout="wide")

st.markdown("""
    <h1 style='margin-bottom: 5px;'>Reverse Modeling – Match Human Data to Animal Models</h1>
    <p style='color: gray;'>Upload your own dataset or search GEO to train on multiple datasets and evaluate against preclinical models.</p>
""", unsafe_allow_html=True)

# Step 1: Search input
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

# Curated datasets
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

# Smart search
st.markdown("### 🔍 Smart GEO Dataset Discovery")
search_results_df = pd.DataFrame()
if st.button("Run smart search"):
    try:
        with st.spinner("Searching GEO..."):
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

# Step 2: Dataset selection
st.markdown("## Step 2: Select Dataset(s) for Modeling")
combined_df = pd.concat([curated_df, search_results_df], ignore_index=True).dropna(subset=["GSE"]).drop_duplicates(subset="GSE")
if not combined_df.empty:
    selected_gses = st.multiselect("Select datasets to use for modeling:", combined_df["GSE"].tolist())
    if selected_gses:
        st.success(f"✅ Selected GSEs: {selected_gses}")
        curated_humans = set(curated_df[curated_df["Organism"] == "Human"]["GSE"].str.lower())
        human_gses = [g for g in selected_gses if g.lower() in curated_humans]
        animal_gses = [g for g in selected_gses if g.lower() not in curated_humans]

        # Download
        st.markdown("### 🔄 Downloading and Preparing Missing Data")
        with st.spinner("Checking and downloading..."):
            for gse in selected_gses:
                exp_path = os.path.join("data", f"{gse}_expression.csv")
                if not os.path.exists(exp_path):
                    try:
                        st.info(f"📥 Downloading {gse}...")
                        download_and_prepare_dataset(gse)
                        st.success(f"✅ {gse} downloaded")
                    except Exception as e:
                        st.error(f"❌ Failed to download {gse}: {e}")
                else:
                    st.info(f"✅ {gse} already exists")

# Step 3: Train
import numpy as np
st.markdown("## Step 3: Train Model")
try:
    if human_gses:
        result = load_multiple_datasets(human_gses)
        if not result or len(result) != 2:
            raise ValueError("Returned data is empty or malformed.")
        human_df, labels = result

        # 1. Ensure labels are numeric (0/1)
        if not pd.api.types.is_numeric_dtype(labels):
            st.warning("Labels are not numeric! Attempting conversion...")
            labels = labels.map({"control": 0, "healthy": 0, "case": 1, "disease": 1}).astype(int)

        # 2. Align labels with feature matrix
        human_df = human_df.T
        common_samples = labels.index.intersection(human_df.index)
        if len(common_samples) < len(labels):
            st.warning(f"Dropping {len(labels) - len(common_samples)} mismatched samples")
        labels = labels.loc[common_samples]
        human_df = human_df.loc[common_samples]

        # 3. Convert labels to flat array
        if isinstance(labels, pd.DataFrame):
            labels = labels.squeeze()
        y = labels.values.ravel()

        # Debug output
        st.write("✅ Post-cleaning label summary:")
        st.write(f"Unique values: {np.unique(y)}")
        st.write(f"Class counts: {pd.Series(y).value_counts().to_dict()}")

        X, y = preprocess_dataset(human_df, labels)
        model, metrics = train_model(X, y)
        st.success("✅ Model training complete")
        st.json(metrics)
    else:
        st.warning("⚠️ No human datasets selected.")
except Exception as e:
    st.error(f"❌ Failed to train: {e}")

           
    # Step 4: Evaluate
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
                st.warning("⚠️ No animal datasets selected.")
    except Exception as e:
            st.error(f"❌ Evaluation failed: {e}")
    else:
        st.info("ℹ️ No datasets available to select.")
