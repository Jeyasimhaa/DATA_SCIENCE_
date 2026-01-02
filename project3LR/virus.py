import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Page config
st.set_page_config(page_title="My Streamlit App", layout="wide")

# Title
st.title("📊 Data Science / ML Streamlit App")

# File upload
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # ✅ READ THE UPLOADED FILE
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    st.subheader("📌 Dataset Info")
    st.write(df.describe())

    # Select target column
    target = st.selectbox("Select Target Column", df.columns)

    # Feature selection
    X = df.drop(columns=[target])
    y = df[target]

    st.subheader("🔍 Feature Columns")
    st.write(X.columns.tolist())

    # Correlation heatmap
    st.subheader("📈 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.success("Data loaded successfully ✅")
else:
    st.warning("Please upload a CSV file to continue")
