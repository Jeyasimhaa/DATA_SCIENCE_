# ------------------------------
# Matplotlib backend (Cloud-safe)
# ------------------------------
import matplotlib
matplotlib.use("Agg")

# ------------------------------
# Imports
# ------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

# ------------------------------
# Page Config
# ------------------------------
st.set_page_config(
    page_title="Mall Customer Segmentation",
    layout="centered"
)

# ------------------------------
# Load Models Safely
# ------------------------------
BASE_DIR = Path(__file__).resolve().parent

@st.cache_resource
def load_models():
    scaler = joblib.load(BASE_DIR / "scaler.pkl")
    kmeans = joblib.load(BASE_DIR / "kmeans_model.pkl")
    return scaler, kmeans

scaler, kmeans = load_models()

# ------------------------------
# App Title
# ------------------------------
st.title("🛍️ Mall Customer Segmentation using K-Means")
st.write("Predicts the **customer cluster** based on income and spending score.")

# ------------------------------
# Sidebar Inputs
# ------------------------------
st.sidebar.header("Enter Customer Details")

annual_income = st.sidebar.slider(
    "Annual Income (k$)", min_value=0, max_value=150, value=50
)

spending_score = st.sidebar.slider(
    "Spending Score (1-100)", min_value=1, max_value=100, value=50
)

# ------------------------------
# Prediction
# ------------------------------
input_data = pd.DataFrame({
    "Annual Income (k$)": [annual_income],
    "Spending Score (1-100)": [spending_score]
})

input_scaled = scaler.transform(input_data)
cluster = kmeans.predict(input_scaled)[0]

st.subheader("📌 Prediction Result")
st.success(f"Customer belongs to **Cluster {cluster}**")

# ------------------------------
# Visualization (Optional Upload)
# ------------------------------
st.subheader("📊 Customer Clusters Visualization")

uploaded_file = st.file_uploader(
    "Upload Mall_Customers.csv (optional)",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    required_cols = {"Annual Income (k$)", "Spending Score (1-100)"}
    if not required_cols.issubset(df.columns):
        st.error("CSV must contain: Annual Income (k$) and Spending Score (1-100)")
    else:
        X = df[list(required_cols)]
        X_scaled = scaler.transform(X)
        df["Cluster"] = kmeans.predict(X_scaled)

        fig, ax = plt.subplots(figsize=(8, 6))

        for c in range(kmeans.n_clusters):
            ax.scatter(
                df[df["Cluster"] == c]["Annual Income (k$)"],
                df[df["Cluster"] == c]["Spending Score (1-100)"],
                label=f"Cluster {c}"
            )

        ax.scatter(
            annual_income,
            spending_score,
            marker="X",
            s=200,
            label="New Customer"
        )

        ax.set_xlabel("Annual Income (k$)")
        ax.set_ylabel("Spending Score (1-100)")
        ax.set_title("Customer Segmentation")
        ax.legend()

        st.pyplot(fig)
else:
    st.info("Upload a CSV file to visualize customer clusters.")
