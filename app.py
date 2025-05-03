# upload_test_app.py

import streamlit as st
import pandas as pd

# Increase max upload size (in megabytes)
st.set_option("server.maxUploadSize", 1000)

st.set_page_config(page_title="Upload Debug Test", layout="centered")
st.title("🧪 Streamlit File Upload Debugger")

# File uploader
uploaded_file = st.file_uploader(
    "Upload a CSV, TXT, or GZ file", type=["csv", "txt", "gz"]
)

# Show what was received
st.write("📁 Raw Uploaded File Object:", uploaded_file)

if uploaded_file:
    st.success("✅ File received!")
    try:
        # Try reading file
        df = pd.read_csv(uploaded_file, sep=None, engine="python", compression="infer")
        st.write("🔍 Parsed File Preview:")
        st.dataframe(df.head())
    except Exception as e:
        st.error("❌ Failed to parse file:")
        st.exception(e)
else:
    st.info("📤 Please upload a file above.")

