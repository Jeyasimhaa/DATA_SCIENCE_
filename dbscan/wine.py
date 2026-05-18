import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import os

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Wine DBSCAN Clustering", layout="centered")

st.title("🍷 Wine Clustering using DBSCAN")
st.write("Direct deployment using wine_clustering_data.csv")

# -------------------------------
# Load Dataset Directly
# -------------------------------
@st.cache_data
def load_data():
    csv_path = os.path.join(os.path.dirname(__file__), "wine_clustering_data.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    else:
        return None

df = load_data()

if df is None:
    st.error("❌ CSV file not found. Please ensure 'wine_clustering_data.csv' is in the same folder as this script.")
    st.stop()

# -------------------------------
# Dataset Preview
# -------------------------------
st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# -------------------------------
# Feature Selection
# -------------------------------
st.subheader("🔢 Feature Selection")
feature_cols = df.columns[:2]
X = df[feature_cols].values

# -------------------------------
# Scaling
# -------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# DBSCAN Parameters
# -------------------------------
st.subheader("⚙️ DBSCAN Parameters")
eps = st.slider("Epsilon (eps)", 0.1, 5.0, 0.5)
min_samples = st.slider("Min Samples", 1, 20, 5)

dbscan = DBSCAN(eps=eps, min_samples=min_samples)
labels = dbscan.fit_predict(X_scaled)
df["Cluster"] = labels

# -------------------------------
# Cluster Summary
# -------------------------------
st.subheader("📈 Cluster Summary")
st.write(df["Cluster"].value_counts())

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
st.success(f"Number of clusters (excluding noise): {n_clusters}")

# -------------------------------
# Visualization
# -------------------------------
st.subheader("📉 Cluster Visualization")
fig, ax = plt.subplots(figsize=(7, 5))
scatter = ax.scatter(
    X_scaled[:, 0],
    X_scaled[:, 1],
    c=labels,
    cmap="viridis",
    s=60
)
ax.set_xlabel(feature_cols[0])
ax.set_ylabel(feature_cols[1])
ax.set_title("DBSCAN Wine Clustering (Noise = -1)")
plt.colorbar(scatter, ax=ax)
st.pyplot(fig)

# -------------------------------
# Test New Data
# -------------------------------
st.subheader("🧪 Test New Wine Data")
f1 = st.number_input(f"{feature_cols[0]}", value=float(X[:, 0].mean()))
f2 = st.number_input(f"{feature_cols[1]}", value=float(X[:, 1].mean()))

if st.button("Test Point"):
    new_point = np.array([[f1, f2]])
    new_point_scaled = scaler.transform(new_point)
    combined_data = np.vstack([X_scaled, new_point_scaled])
    new_labels = dbscan.fit_predict(combined_data)
    new_label = new_labels[-1]

    st.info(f"Assigned Cluster: {new_label}")
    st.warning("DBSCAN does not truly predict — it reclusters the data.")
