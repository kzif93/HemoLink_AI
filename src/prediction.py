import streamlit as st
import pandas as pd

def predict_and_display(model, X_test, y_test):
    probs = model.predict_proba(X_test)[:, 1]
    df = pd.DataFrame({
        "Prediction Probability": probs,
        "True Label": y_test
    })
    st.bar_chart(df["Prediction Probability"])
    st.write("Prediction Table", df)
