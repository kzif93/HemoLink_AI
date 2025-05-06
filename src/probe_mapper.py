# src/probe_mapper.py

import os
import pandas as pd
import streamlit as st
import GEOparse


def download_platform_annotation(gse_id: str, out_dir: str = "data") -> str:
    """
    Downloads the platform (GPL) annotation file for a given GSE ID using GEOparse.
    Returns the path to the annotation file as a CSV.
    """
    os.makedirs(out_dir, exist_ok=True)

    st.info(f"🔍 Getting platform info for {gse_id}...")
    gse = GEOparse.get_GEO(geo=gse_id, destdir=out_dir, include_data=False)
    gpl_ids = list(gse.gpls.keys())

    if not gpl_ids:
        raise ValueError(f"❌ No platform (GPL) found for {gse_id}")

    gpl_id = gpl_ids[0]
    st.success(f"✅ Found platform: {gpl_id}")

    gpl = GEOparse.get_GEO(geo=gpl_id, destdir=out_dir)
    gpl_table = gpl.table

    out_path = os.path.join(out_dir, f"{gpl_id}_annotation.csv")
    gpl_table.to_csv(out_path, index=False)
    st.success(f"✅ Saved annotation file to {out_path}")
    return out_path


def map_probes_to_genes(expr_file: str, annotation_file: str,
                        probe_col: str = "ID", gene_col: str = "Gene Symbol") -> pd.DataFrame:
    """
    Maps probe IDs in an expression matrix to gene symbols using an annotation file.
    """
    st.info("🔧 Mapping probes to gene symbols...")
    expr_df = pd.read_csv(expr_file, index_col=0)
    ann_df = pd.read_csv(annotation_file)

    if probe_col not in ann_df.columns or gene_col not in ann_df.columns:
        raise ValueError("❌ Required columns not found in annotation file")

    # Drop missing gene symbols and deduplicate
    ann_df = ann_df[[probe_col, gene_col]].dropna()
    ann_df = ann_df.drop_duplicates(subset=probe_col)

    # Merge
    merged = expr_df.merge(ann_df.set_index(probe_col), left_index=True, right_index=True, how="inner")
    merged = merged.reset_index(drop=True)
    merged = merged.groupby(gene_col).mean()

    st.success("✅ Probes mapped to gene symbols.")
    return merged


# Optional: Run as standalone script
if __name__ == "__main__":
    gse_id = "GSE16561"
    expr_path = f"data/{gse_id}_expression.csv"
    ann_path = download_platform_annotation(gse_id)
    mapped_df = map_probes_to_genes(expr_path, ann_path)
    mapped_df.to_csv(f"data/{gse_id}_gene_symbols.csv")
    print("✅ Saved final matrix with gene symbols.")

