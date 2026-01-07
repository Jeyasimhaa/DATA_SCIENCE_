import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

# ===================== PAGE CONFIG =====================
st.set_page_config(page_title="Mall Customer Segmentation", layout="centered")

# ===================== PATH HANDLING =====================
BASE_DIR = Path(__file__).resolve().parent
SCALER_PATH = BASE_DIR / "scaler.pkl"
MODEL_PATH = BASE_DIR / "kmeans_model.pkl"
CSV_PATH = BASE_DIR / "Mall_Customers.csv"

# ===================== LOAD MODEL & SCALER =====================
if not SCALER_PATH.exists():
    st.error(f"❌ scaler.pkl not found in {BASE_DIR.name}")
    st.stop()

if not MODEL_PATH.exists():
    st.error(f"❌ kmeans_model.pkl not found in {BASE_DIR.name}")
    st.stop()

scaler = joblib.load(SCALER_PATH)
kmeans = joblib.load(MODEL_PATH)

# ===================== APP UI =====================
st.title("🛍️ Mall Customer Segmentation using K-Means")
st.write("Predict the **customer cluster** using income and spending score.")

# ===================== SIDEBAR INPUTS =====================
st.sidebar.header("Enter Customer Details")

annual_income = st.sidebar.slider(
    "Annual Income (k$)", 0, 150, 50
)

spending_score = st.sidebar.slider(
    "Spending Score (1–100)", 1, 100, 50
)

# ===================== PREDICTION =====================
input_data = pd.DataFrame({
    "Annual Income (k$)": [annual_income],
    "Spending Score (1-100)": [spending_score]
})

input_scaled = scaler.transform(input_data)
cluster = kmeans.predict(input_scaled)[0]

st.subheader("📌 Prediction Result")
st.success(f"Customer belongs to **Cluster {cluster}**")

# ===================== VISUALIZATION =====================
st.subheader("📊 Customer Clusters Visualization")

if CSV_PATH.exists():
    df = pd.read_csv(CSV_PATH)

    X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
    X_scaled = scaler.transform(X)
    df['Cluster'] = kmeans.predict(X_scaled)

    fig, ax = plt.subplots(figsize=(8, 6))

    for c in range(kmeans.n_clusters):
        ax.scatter(
            df[df['Cluster'] == c]['Annual Income (k$)'],
            df[df['Cluster'] == c]['Spending Score (1-100)'],
            label=f'Cluster {c}'
        )

    ax.scatter(
        annual_income,
        spending_score,
        c='black',
        s=200,
        marker='X',
        label='New Customer'
    )

    ax.set_xlabel("Annual Income (k$")
    ax.set_ylabel("Spending Score (1–100)")
    ax.set_title("Customer Segmentation")
    ax.legend()

    st.pyplot(fig)
else:
    st.warning("⚠️ Mall_Customers.csv not found. Visualization skipped.")
