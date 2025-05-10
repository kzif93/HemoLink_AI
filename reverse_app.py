
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

    # === Safe preview/edit block ===
                    st.success("✅ Updated labels saved.")
                else:
                    st.error("❌ Edited labels must contain exactly two classes.")
    except Exception as e:
        st.error(f"❌ Error during label preview/edit: {e}")
        # === OPTIONAL LABEL PREVIEW UI ===
                        st.success("✅ Updated labels saved.")
                    else:
                        st.error("❌ Edited labels must contain exactly two classes.")
        except Exception as preview_error:
            st.error(f"❌ Label preview/edit failed: {preview_error}")
            if st.checkbox("✏️ Manually edit labels?", key="edit_labels"):
                edited = st.data_editor(pd.DataFrame({"Sample": labels.index, "Label": labels.values}), num_rows="dynamic")
                if "Label" in edited.columns and edited["Label"].nunique() == 2:
                    labels = edited.set_index("Sample")["Label"]
                    labels.to_csv(label_out)
        # === Label Preview and Optional Manual Edit ===
        if st.checkbox("🔍 Preview labels before proceeding"):
            st.dataframe(pd.DataFrame({"Sample": labels.index, "Label": labels.values}))
            st.warning("These labels will be used for training.")
            if st.checkbox("✏️ Manually edit labels?", key="edit_labels"):
                edited = st.data_editor(
                    pd.DataFrame({"Sample": labels.index, "Label": labels.values}),
                    num_rows="dynamic"
                )
                if "Label" in edited.columns and edited["Label"].nunique() == 2:
                    labels = edited.set_index("Sample")["Label"]
                    labels.to_csv(label_out)
                    st.success("✅ Updated labels saved.")
                else:
                    st.error("❌ Edited labels must contain exactly two classes.")
        # === Label Preview and Optional Manual Edit ===
            st.dataframe(pd.DataFrame({"Sample": labels.index, "Label": labels.values}))
            st.warning("These labels will be used for training.")
            if st.checkbox("✏️ Manually edit labels?", key="edit_labels"):
                edited = st.data_editor(
                    pd.DataFrame({"Sample": labels.index, "Label": labels.values}),
                    num_rows="dynamic"
                )
                if "Label" in edited.columns and edited["Label"].nunique() == 2:
                    labels = edited.set_index("Sample")["Label"]
                else:
    # === Label Preview and Optional Manual Edit ===
                else:
                st.warning("These labels will be used for training.")
                if st.checkbox("✏️ Manually edit labels?", key="edit_labels"):
                    edited = st.data_editor(pd.DataFrame({"Sample": labels.index, "Label": labels.values}), num_rows="dynamic")
                    if "Label" in edited.columns and edited["Label"].nunique() == 2:
                        labels = edited.set_index("Sample")["Label"]
                    else:
            return out_path
        else:
            st.warning("⚠️ Only one class found in labels from sample titles.")
        st.write("🧠 Available metadata columns:", list(metadata.columns))

        # === CUSTOM LABEL LOGIC FOR GSE22255 ===
        # === SMART LABELING ===
        label_found = False
        for colname in ["title", "characteristics_ch1"]:
            if colname in metadata.columns:
        # === OPTIONAL LABEL PREVIEW UI ===
                        st.success("✅ Updated labels saved.")
                    else:
                        st.error("❌ Edited labels must contain exactly two classes.")
        except Exception as preview_error:
            st.error(f"❌ Label preview/edit failed: {preview_error}")
            if st.checkbox("✏️ Manually edit labels?", key="edit_labels"):
                edited = st.data_editor(pd.DataFrame({"Sample": labels.index, "Label": labels.values}), num_rows="dynamic")
                if "Label" in edited.columns and edited["Label"].nunique() == 2:
                    labels = edited.set_index("Sample")["Label"]
                    labels.to_csv(label_out)
                    st.success("✅ Updated labels saved.")
                else:
                    st.error("❌ Edited labels must contain exactly two classes.")
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
            if labels.nunique() == 2:
                labels.name = "label"
                labels.to_csv(label_out)
        # === OPTIONAL LABEL PREVIEW UI ===
                        st.success("✅ Updated labels saved.")
                    else:
                        st.error("❌ Edited labels must contain exactly two classes.")
        except Exception as preview_error:
            st.error(f"❌ Label preview/edit failed: {preview_error}")
            if st.checkbox("✏️ Manually edit labels?", key="edit_labels"):
                edited = st.data_editor(pd.DataFrame({"Sample": labels.index, "Label": labels.values}), num_rows="dynamic")
                if "Label" in edited.columns and edited["Label"].nunique() == 2:
                    labels = edited.set_index("Sample")["Label"]
                    labels.to_csv(label_out)
                    st.success("✅ Updated labels saved.")
                else:
                    st.error("❌ Edited labels must contain exactly two classes.")
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
        # === OPTIONAL LABEL PREVIEW UI ===
                        st.success("✅ Updated labels saved.")
                    else:
                        st.error("❌ Edited labels must contain exactly two classes.")
        except Exception as preview_error:
            st.error(f"❌ Label preview/edit failed: {preview_error}")
            if st.checkbox("✏️ Manually edit labels?", key="edit_labels"):
                edited = st.data_editor(pd.DataFrame({"Sample": labels.index, "Label": labels.values}), num_rows="dynamic")
                if "Label" in edited.columns and edited["Label"].nunique() == 2:
                    labels = edited.set_index("Sample")["Label"]
                    labels.to_csv(label_out)
                    st.success("✅ Updated labels saved.")
                else:
                    st.error("❌ Edited labels must contain exactly two classes.")
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
                st.error(f"⚠️ Metadata preview failed: {preview_err}")
            labels = pd.Series([0] * df.shape[1], index=df.columns, name="label")
            labels.to_csv(label_out)
        # === OPTIONAL LABEL PREVIEW UI ===
                        st.success("✅ Updated labels saved.")
                    else:
                        st.error("❌ Edited labels must contain exactly two classes.")
        except Exception as preview_error:
            st.error(f"❌ Label preview/edit failed: {preview_error}")
            if st.checkbox("✏️ Manually edit labels?", key="edit_labels"):
                edited = st.data_editor(pd.DataFrame({"Sample": labels.index, "Label": labels.values}), num_rows="dynamic")
                if "Label" in edited.columns and edited["Label"].nunique() == 2:
                    labels = edited.set_index("Sample")["Label"]
                    labels.to_csv(label_out)
                    st.success("✅ Updated labels saved.")
                else:
                    st.error("❌ Edited labels must contain exactly two classes.")
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
        st.error(f"❌ Failed to load curated datasets: {e}")

# Smart search
st.markdown("### 🔍 Smart GEO Dataset Discovery")
search_results_df = pd.DataFrame()
if st.button("Run smart search"):
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
                        st.error(f"❌ Failed to download {gse}: {e}")
                else:
                    st.info(f"✅ {gse} already exists")

        # Step 3: Train
        st.markdown("## Step 3: Train Model")
            st.error(f"❌ Failed to train: {e}")

        # Step 4: Evaluate
        st.markdown("## Step 4: Evaluate on Animal Datasets")
            st.error(f"❌ Evaluation failed: {e}")
else:
    st.info("ℹ️ No datasets available to select.")
