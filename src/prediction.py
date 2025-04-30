import streamlit as st
import pandas as pd
import numpy as np

def predict_and_display(model, X_test, y_test):
    st.subheader("🔬 Predictions")

    try:
        proba = model.predict_proba(X_test)

        # Handle binary vs single-class models
        if proba.shape[1] == 1:
            probs = [round(p[0], 3) for p in proba]
        else:
            probs = [round(p[1], 3) for p in proba]  # Class 1 = "positive" (e.g., VTE)

        preds = model.predict(X_test)

        # Combine results
        results_df = pd.DataFrame({
            "Predicted Label": preds,
            "True Label": y_test,
            "Probability (Class 1)": probs
        }, index=X_test.index)

        st.dataframe(results_df)

        # Show probability bars
        st.write("📈 Prediction Probabilities:")
        st.bar_chart(pd.Series(probs, index=X_test.index))

    except Exception as e:
        st.warning(f"⚠️ Could not generate prediction probabilities: {e}")
