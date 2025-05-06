# src/probe_mapper.py

import os
import pandas as pd
import streamlit as st
import GEOparse


def download_platform_annotation(gse_id: str, out_dir: str = "data") -> str:
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


def map_probes_to_genes(expr_file: str, annotation_file: str) -> pd.DataFrame:
    st.info("🔧 Mapping probes to gene symbols...")
    expr_df = pd.read_csv(expr_file, index_col=0)
    ann_df = pd.read_csv(annotation_file)

    # Heuristically detect columns
    possible_probe_cols = ["ID", "ID_REF", "Probe ID"]
    possible_gene_cols = ["Gene Symbol", "GENE_SYMBOL", "SYMBOL", "Gene symbol"]

    probe_col = next((col for col in possible_probe_cols if col in ann_df.columns), None)
    gene_col = next((col for col in possible_gene_cols if col in ann_df.columns), None)

    if not probe_col or not gene_col:
        st.warning("⚠️ Auto-detection failed. Please select columns manually:")
        st.write("🧬 Annotation Columns:", list(ann_df.columns))

        probe_col = st.selectbox("Select probe ID column:", ann_df.columns, key="probe")
        gene_col = st.selectbox("Select gene symbol column:", ann_df.columns, key="gene")

    st.info(f"🧬 Using columns: {probe_col} → {gene_col}")

    ann_df = ann_df[[probe_col, gene_col]].dropna()
    ann_df = ann_df.drop_duplicates(subset=probe_col)

    merged = expr_df.merge(ann_df.set_index(probe_col), left_index=True, right_index=True, how="inner")
    merged = merged.reset_index(drop=True)
    merged = merged.groupby(gene_col).mean()

    st.success("✅ Probes mapped to gene symbols.")
    return merged


if __name__ == "__main__":
    gse_id = "GSE16561"
    expr_path = f"data/{gse_id}_expression.csv"
    ann_path = download_platform_annotation(gse_id)
    mapped_df = map_probes_to_genes(expr_path, ann_path)
    mapped_df.to_csv(f"data/{gse_id}_gene_symbols.csv")
    print("✅ Saved final matrix with gene symbols.")
