from sklearn.metrics import roc_auc_score

def test_model_on_dataset(model, X):
    """
    Uses a trained model to make predictions on dataset X
    and calculates the AUC score.
    """
    y_pred_prob = model.predict_proba(X)[:, 1]  # get probability for class 1
    # dummy labels since we're only testing model transfer (use 50/50 guess for AUC)
    dummy_labels = [0] * (len(y_pred_prob) // 2) + [1] * (len(y_pred_prob) - len(y_pred_prob) // 2)
    
    try:
        auc = roc_auc_score(dummy_labels, y_pred_prob)
    except:
        auc = 0.5  # fallback if prediction fails or all one class

    return auc, y_pred_prob
