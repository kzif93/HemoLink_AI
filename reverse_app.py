
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
from model_training import train_model
from prediction import test_model_on_dataset
from explainability import extract_shap_values, compare_shap_vectors
from reverse_modeling import list_animal_datasets, load_multiple_datasets
from curated_sets import curated_registry

Entrez.email = "your_email@example.com"

KEYWORDS = ["stroke", "ischemia", "thrombosis", "vte", "dvt", "aps", "antiphospholipid", "control", "healthy", "normal"]

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
        print(f"[smart_search_animal_geo] Error: {e}")
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
        mapped.to_csv(out_path)

    try:
        metadata = pd.DataFrame({gsm: sample.metadata for gsm, sample in geo.gsms.items()}).T
        sample_titles = pd.Series({gsm: sample.metadata.get("title", [""])[0] for gsm, sample in geo.gsms.items()})
        labels = sample_titles.str.lower().map(lambda x: 1 if "stroke" in x or "is" in x else 0)
        if labels.nunique() == 2:
            labels.name = "label"
            labels.to_csv(label_out)
            if st.checkbox("🔍 Preview labels before proceeding"):
                st.dataframe(pd.DataFrame({"Sample": labels.index, "Label": labels.values}))
                st.warning("These labels will be used for training.")
                if st.checkbox("✏️ Manually edit labels?", key="edit_labels"):
                    edited = st.data_editor(pd.DataFrame({"Sample": labels.index, "Label": labels.values}), num_rows="dynamic")
                    if "Label" in edited.columns and edited["Label"].nunique() == 2:
                        labels = edited.set_index("Sample")["Label"]
                        labels.to_csv(label_out)
                        st.success("✅ Updated labels saved.")
                    else:
                        st.error("❌ Edited labels must contain exactly two classes.")
            st.success("✅ Labels successfully parsed from sample titles (row metadata).")
            return out_path
        else:
            st.warning("⚠️ Only one class found in labels from sample titles.")
        st.write("🧠 Available metadata columns:", list(metadata.columns))

        # === CUSTOM LABEL LOGIC FOR GSE22255 ===
        # === SMART LABELING ===
        label_found = False
        for colname in ["title", "characteristics_ch1"]:
            if colname in metadata.columns:
                try:
                    values = metadata[colname].astype(str).str.lower()
                    labels = values.map(lambda x: 1 if "stroke" in x or "is" in x else 0)
                    if labels.nunique() == 2:
                        labels.name = "label"
                        labels.to_csv(label_out)
            if st.checkbox("🔍 Preview labels before proceeding"):
                st.dataframe(pd.DataFrame({"Sample": labels.index, "Label": labels.values}))
                st.warning("These labels will be used for training.")
                if st.checkbox("✏️ Manually edit labels?", key="edit_labels"):
                    edited = st.data_editor(pd.DataFrame({"Sample": labels.index, "Label": labels.values}), num_rows="dynamic")
                    if "Label" in edited.columns and edited["Label"].nunique() == 2:
                        labels = edited.set_index("Sample")["Label"]
                        labels.to_csv(label_out)
                        st.success("✅ Updated labels saved.")
                    else:
                        st.error("❌ Edited labels must contain exactly two classes.")
                        st.success(f"✅ Labels generated from {colname}.")
                        return out_path
                except Exception as e:
                    st.warning(f"⚠️ Could not label from {colname}: {e}")

        # Manual column selection if auto fails
            st.dataframe(pd.DataFrame({selected_col: values}).head(10))
            if labels.nunique() == 2:
                labels.name = "label"
                labels.to_csv(label_out)
            if st.checkbox("🔍 Preview labels before proceeding"):
                st.dataframe(pd.DataFrame({"Sample": labels.index, "Label": labels.values}))
                st.warning("These labels will be used for training.")
                if st.checkbox("✏️ Manually edit labels?", key="edit_labels"):
                    edited = st.data_editor(pd.DataFrame({"Sample": labels.index, "Label": labels.values}), num_rows="dynamic")
                    if "Label" in edited.columns and edited["Label"].nunique() == 2:
                        labels = edited.set_index("Sample")["Label"]
                        labels.to_csv(label_out)
                        st.success("✅ Updated labels saved.")
                    else:
                        st.error("❌ Edited labels must contain exactly two classes.")
                st.success(f"✅ Labels generated from selected column: {selected_col}")
                return out_path
            else:
                st.warning("⚠️ Still only one class found.")
        except Exception as e:
            st.error(f"❌ Failed to label from selected column: {e}")
        success = False
        for col in metadata.columns:
            try:
                values = metadata[col].astype(str).str.lower()
                labels = values.map(lambda x: 1 if any(k in x for k in KEYWORDS) else 0)
                if labels.nunique() == 2:
                    labels.name = "label"
                    labels.to_csv(label_out)
            if st.checkbox("🔍 Preview labels before proceeding"):
                st.dataframe(pd.DataFrame({"Sample": labels.index, "Label": labels.values}))
                st.warning("These labels will be used for training.")
                if st.checkbox("✏️ Manually edit labels?", key="edit_labels"):
                    edited = st.data_editor(pd.DataFrame({"Sample": labels.index, "Label": labels.values}), num_rows="dynamic")
                    if "Label" in edited.columns and edited["Label"].nunique() == 2:
                        labels = edited.set_index("Sample")["Label"]
                        labels.to_csv(label_out)
                        st.success("✅ Updated labels saved.")
                    else:
                        st.error("❌ Edited labels must contain exactly two classes.")
                    print(f"[Auto-labeling] ✅ Used column: {col}")
                    print(f"[Label distribution] {labels.value_counts().to_dict()}")
                    success = True
                    break
            except Exception:
                continue

        if not success:
            st.warning("⚠️ Auto-labeling failed. Assigning default label 0 to all.")
            try:
                st.warning("⚠️ Showing metadata preview (first 5 columns × 10 samples):")
                st.dataframe(metadata.iloc[:, :5].head(10))
            except Exception as preview_err:
                st.error(f"⚠️ Metadata preview failed: {preview_err}")
            labels = pd.Series([0] * df.shape[1], index=df.columns, name="label")
            labels.to_csv(label_out)
            if st.checkbox("🔍 Preview labels before proceeding"):
                st.dataframe(pd.DataFrame({"Sample": labels.index, "Label": labels.values}))
                st.warning("These labels will be used for training.")
                if st.checkbox("✏️ Manually edit labels?", key="edit_labels"):
                    edited = st.data_editor(pd.DataFrame({"Sample": labels.index, "Label": labels.values}), num_rows="dynamic")
                    if "Label" in edited.columns and edited["Label"].nunique() == 2:
                        labels = edited.set_index("Sample")["Label"]
                        labels.to_csv(label_out)
                        st.success("✅ Updated labels saved.")
                    else:
                        st.error("❌ Edited labels must contain exactly two classes.")
    except Exception as e:
        print(f"[Auto-labeling failed] ❌ {e}")

    return out_path

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
        st.markdown("## Step 3: Train Model")
        try:
            if human_gses:
                result = load_multiple_datasets(human_gses)
                if not result or len(result) != 2:
                    raise ValueError("Returned data is empty or malformed.")
                human_df, labels = result
                if human_df.empty or labels.empty:
                    raise ValueError("Loaded data or labels are empty.")
                if isinstance(labels, pd.DataFrame):
                    labels = labels.iloc[:, 0]

                labels.index = labels.index.astype(str).str.strip()
                human_df.columns = human_df.columns.astype(str).str.strip()
                unmatched = [idx for idx in labels.index if idx not in human_df.columns]
                if unmatched:
                    st.warning(f"⚠️ Unmatched label samples: {unmatched[:5]}... (+{len(unmatched)-5} more)" if len(unmatched) > 5 else f"⚠️ Unmatched label samples: {unmatched}")
                labels = labels[labels.index.isin(human_df.columns)]
                human_df = human_df[labels.index]

                st.warning(f"⚠️ Label distribution: {labels.value_counts().to_dict()}")
                if labels.nunique() < 2:
                    raise ValueError("Only one class found in labels.")

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
