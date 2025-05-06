# reverse_modeling.py

import os
import sys
import pandas as pd
from sklearn.metrics import roc_auc_score

# Ensure src modules are importable
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.ortholog_mapper import map_orthologs
from src.preprocessing import clean_and_scale
from src.model_training import train_model
from src.prediction import predict_on_human
from src.explainability import extract_shap_values


def evaluate_mouse_models(human_df, y_human, ortholog_df, model_dir="animal_models"):
    results = []

    # Train on human
    X_human = clean_and_scale(human_df)
    model, metrics = train_model(X_human, y_human)

    for fname in os.listdir(model_dir):
        if not fname.endswith(".csv"):
            continue

        path = os.path.join(model_dir, fname)
        mouse_df = pd.read_csv(path, index_col=0)

        try:
            # Normalize casing
            mouse_df.columns = mouse_df.columns.str.upper()
            human_df.columns = human_df.columns.str.upper()
            ortholog_df["mouse_symbol"] = ortholog_df["mouse_symbol"].str.upper()
            ortholog_df["human_symbol"] = ortholog_df["human_symbol"].str.upper()

            # Align genes
            mouse_aligned, human_aligned = map_orthologs(mouse_df, human_df, ortholog_df)
            X_mouse = clean_and_scale(mouse_aligned)

            # Predict
            preds = predict_on_human(model, X_mouse)
            pred_probs = preds.iloc[:, 1] if preds.shape[1] > 1 else preds.iloc[:, 0]
            auc = roc_auc_score([0] * len(preds), pred_probs)  # dummy labels

            # SHAP similarity (optional)
            human_shap = extract_shap_values(model, X_human)
            mouse_shap = extract_shap_values(model, X_mouse)
            shared_genes = list(set(human_shap.columns) & set(mouse_shap.columns))
            similarity = (
                human_shap[shared_genes].mean().corr(mouse_shap[shared_genes].mean())
                if shared_genes else 0.0
            )

            results.append({
                "model_file": fname,
                "shared_genes": len(shared_genes),
                "AUC": round(auc, 3),
                "SHAP_similarity": round(similarity, 3)
            })

        except Exception as e:
            results.append({
                "model_file": fname,
                "shared_genes": 0,
                "AUC": None,
                "SHAP_similarity": None,
                "error": str(e)
            })

    return pd.DataFrame(results).sort_values("SHAP_similarity", ascending=False)
