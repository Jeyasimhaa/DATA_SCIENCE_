import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="DBSCAN Clustering App", layout="centered")

st.title("🍷 DBSCAN Wine Clustering")
st.write("Density-Based Clustering using DBSCAN")

# -------------------------------
# Load Pickle Files
# -------------------------------
@st.cache_resource
def load_models():
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open("dbscan_model.pkl", "rb") as f:
        dbscan = pickle.load(f)

    return scaler, dbscan


scaler, dbscan = load_models()

# -------------------------------
# Upload Dataset
# -------------------------------
st.subheader("📂 Upload CSV File")

uploaded_file = st.file_uploader("Upload wine_clustering_data.csv", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.dataframe(df.head())

    # Select first two columns
    X = df.iloc[:, [0, 1]].values

    # Scale
    X_scaled = scaler.transform(X)

    # DBSCAN clustering
    labels = dbscan.fit_predict(X_scaled)
    df["Cluster"] = labels

    # -------------------------------
    # Cluster Info
    # -------------------------------
    st.subheader("📊 Cluster Summary")
    st.write(df["Cluster"].value_counts())

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    st.success(f"Number of clusters (excluding noise): {n_clusters}")

    # -------------------------------
    # Plot Clusters
    # -------------------------------
    st.subheader("📈 Cluster Visualization")

    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(
        X_scaled[:, 0],
        X_scaled[:, 1],
        c=labels,
        cmap="viridis",
        s=50
    )
    ax.set_xlabel("Feature 1 (Scaled)")
    ax.set_ylabel("Feature 2 (Scaled)")
    ax.set_title("DBSCAN Clustering (Noise = -1)")
    plt.colorbar(scatter, ax=ax)

    st.pyplot(fig)

# -------------------------------
# New Data Prediction
# -------------------------------
st.subheader("🧪 Test New Data Point")

f1 = st.number_input("Feature 1", value=13.5)
f2 = st.number_input("Feature 2", value=2.3)

if st.button("Cluster New Point"):
    new_data = np.array([[f1, f2]])
    new_data_scaled = scaler.transform(new_data)

    # DBSCAN reclustering (important limitation)
    new_label = dbscan.fit_predict(new_data_scaled)

    st.info(f"Cluster Label: {new_label[0]}")
    st.warning("Note: DBSCAN does NOT truly predict. It reclusters data.")
