import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# Load trained model and scaler
scaler = joblib.load("scaler.pkl")
kmeans = joblib.load("kmeans_model.pkl")

st.set_page_config(page_title="Mall Customer Segmentation", layout="centered")

st.title("🛍️ Mall Customer Segmentation using K-Means")
st.write("This app predicts the **customer cluster** based on income and spending score.")

# Sidebar inputs
st.sidebar.header("Enter Customer Details")

annual_income = st.sidebar.slider(
    "Annual Income (k$)", min_value=0, max_value=150, value=50
)

spending_score = st.sidebar.slider(
    "Spending Score (1-100)", min_value=1, max_value=100, value=50
)

# Prepare input data
input_data = pd.DataFrame({
    "Annual Income (k$)": [annual_income],
    "Spending Score (1-100)": [spending_score]
})

# Scale input
input_scaled = scaler.transform(input_data)

# Predict cluster
cluster = kmeans.predict(input_scaled)[0]

st.subheader("📌 Prediction Result")
st.success(f"Customer belongs to **Cluster {cluster}**")

# Optional: Load dataset to visualize clusters
st.subheader("📊 Customer Clusters Visualization")

try:
    df = pd.read_csv("Mall_Customers.csv")

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

    ax.set_xlabel("Annual Income (k$)")
    ax.set_ylabel("Spending Score (1-100)")
    ax.set_title("Customer Segmentation")
    ax.legend()

    st.pyplot(fig)

except FileNotFoundError:
    st.warning("Mall_Customers.csv not found. Upload it to visualize clusters.")
