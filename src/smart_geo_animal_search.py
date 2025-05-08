import os
import re
import pandas as pd
from Bio import Entrez
from datetime import datetime

Entrez.email = "your_email@example.com"  # Replace with a valid email

KEYWORDS = ["stroke", "ischemia", "thrombosis", "vte", "dvt", "aps", "antiphospholipid"]
TISSUE_HINTS = ["brain", "cortex", "hippocampus", "vein", "blood"]


# Fallback gold-standard datasets by domain
gold_standard_datasets = {
    "stroke": [
        {"GSE": "GSE233813", "Organism": "Mouse", "Model": "MCAO (24h)", "Platform": "RNA-Seq",
         "Description": "High-quality RNA-seq of ischemic brain tissue in male and female mice post-MCAO; includes sham controls."},
        {"GSE": "GSE162072", "Organism": "Rat", "Model": "tMCAO (3h)", "Platform": "Microarray",
         "Description": "Early vascular transcriptomic response in female rats. Focus on middle cerebral arteries post-reperfusion."},
        {"GSE": "GSE137482", "Organism": "Mouse", "Model": "Photothrombosis (24h)", "Platform": "RNA-Seq",
         "Description": "Bulk RNA-seq of ischemic cortex; distinct stroke mechanism (photochemical) compared to MCAO."},
        {"GSE": "PMC10369109", "Organism": "Rabbit", "Model": "Vascular Spatial Transcriptomics", "Platform": "Visium",
         "Description": "Spatially resolved transcriptomic analysis of intracranial vessels in rabbit brain."},
        {"GSE": "GSE36010", "Organism": "Rat", "Model": "MCAO", "Platform": "Microarray",
         "Description": "Part of integrated rat meta-study for biomarker discovery in stroke."},
    ],
    "vte": [
        {"GSE": "GSE125965", "Organism": "Mouse", "Model": "IVC stenosis-induced DVT", "Platform": "Microarray",
         "Description": "Comprehensive profiling of blood and vein wall tissue in mouse inferior vena cava DVT model."},
        {"GSE": "GSE19151", "Organism": "Mouse", "Model": "TF-induced thrombosis", "Platform": "Microarray",
         "Description": "Mouse model mimicking procoagulant state via tissue factor; includes platelet and leukocyte activation signals."},
        {"GSE": "GSE145993", "Organism": "Mouse", "Model": "Ferric chloride DVT", "Platform": "RNA-Seq",
         "Description": "Gene expression in mouse vein endothelial cells post FeCl₃-induced thrombosis."},
        {"GSE": "GSE245276", "Organism": "Mouse", "Model": "DVT with anti-P-selectin", "Platform": "RNA-Seq",
         "Description": "Effect of anti-P-selectin antibody on venous thrombosis; includes immune–thrombus interaction data."},
        {"GSE": "GSE46265", "Organism": "Rat", "Model": "IVC ligation", "Platform": "Microarray",
         "Description": "Rat gene expression profiling post-ligation; adds cross-species value."},
    ],
    "aps": [
        {"GSE": "GSE139342", "Organism": "Mouse", "Model": "Obstetric APS", "Platform": "RNA-Seq",
         "Description": "Placenta and fetal brain transcriptomes in APS model."},
        {"GSE": "GSE99329", "Organism": "Mouse", "Model": "aPL + LPS thrombosis", "Platform": "Microarray",
         "Description": "Mimics thrombotic APS using inflammatory second hit; blood transcriptomics post-induction."},
        {"GSE": "GSE172935", "Organism": "Mouse", "Model": "APS pregnancy complications", "Platform": "RNA-Seq",
         "Description": "Placental transcriptomes from APS pregnancies at different gestation stages."},
        {"GSE": "GSE61616", "Organism": "Rat", "Model": "LPS-enhanced thrombo-inflammation", "Platform": "Microarray",
         "Description": "Simulates second-hit immune-driven vascular injury typical of APS."},
        {"GSE": "PRJNA640011", "Organism": "Rhesus macaque", "Model": "LPS + coagulopathy", "Platform": "RNA-Seq",
         "Description": "Non-human primate model of immune-thrombotic injury; surrogate for APS-like inflammation."},
    ]
}


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

        # Add fallback curated datasets
        domain = "stroke" if "stroke" in keywords else "vte" if "thrombosis" in keywords else "aps" if "aps" in keywords else None
        if domain:
            curated = gold_standard_datasets.get(domain, [])
            for entry in curated:
                entry = entry.copy()
                entry.update({"Tag": "⭐ Curated", "Score": 15})
                summaries.append(entry)

        return summaries
    except Exception as e:
        print(f"[smart_search_animal_geo] Error: {e}")
        return []


def download_and_prepare_dataset(gse_id):
    # Placeholder: Implement actual GEO download + preprocessing logic
    raise NotImplementedError("Automatic GEO dataset download is not yet implemented for this GSE.")
