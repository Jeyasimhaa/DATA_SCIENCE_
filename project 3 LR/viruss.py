import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Page config
st.set_page_config(page_title="My Streamlit App", layout="wide")

# Title
st.title("📊 Data Science / ML Streamlit App")

# File upload
uploaded_file = st.file_uploader("C:\Users\rjey0\Downloads\practise\project 3 LR\framingham_heart_disease.csv")

if uploaded_file is not None:
    # Load data
    df = pd.read_csv("C:\Users\rjey0\Downloads\practise\project 3 LR\framingham_heart_disease.csv)

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
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.success("Data loaded successfully ✅")
else:
    st.warning("Please upload a CSV file to continue")
