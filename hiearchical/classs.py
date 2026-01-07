# ===================== IMPORT LIBRARIES =====================
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import numpy as np

from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage

# ===================== PAGE CONFIG =====================
st.set_page_config(page_title="Hierarchical Clustering App", layout="centered")
st.title("📊 Hierarchical Clustering (Age vs Income)")

# ===================== FILE UPLOADER =====================
st.subheader("📂 Upload Excel Dataset")

uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])

if uploaded_file is None:
    st.warning("Please upload an Excel file to continue.")
    st.stop()

# ===================== LOAD DATA =====================
df = pd.read_excel(
    uploaded_file,
    names=["name", "age", "income"]
)

st.subheader("Dataset Preview")
st.dataframe(df)

# ===================== FEATURES =====================
X = df[["age", "income"]]

# ===================== FEATURE SCALING =====================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ===================== SCATTER PLOT =====================
st.subheader("Scatter Plot (Age vs Income)")
fig1, ax1 = plt.subplots()
sns.scatterplot(
    x=df["age"],
    y=df["income"],
    s=50,
    ax=ax1
)
ax1.set_xlabel("Age")
ax1.set_ylabel("Income")
st.pyplot(fig1)

# ===================== HIERARCHICAL CLUSTERING =====================
hc = AgglomerativeClustering(
    n_clusters=2,
    metric="euclidean",
    linkage="average"
)

df["cluster"] = hc.fit_predict(X_scaled)

# ===================== SAVE MODEL =====================
with open("hierarchical_model.pkl", "wb") as f:
    pickle.dump(hc, f)

# ===================== CLUSTERED SCATTER PLOT =====================
st.subheader("Cluster Visualization")
fig2, ax2 = plt.subplots()
sns.scatterplot(
    x=df["age"],
    y=df["income"],
    hue=df["cluster"],
    palette="Set1",
    s=60,
    ax=ax2
)
ax2.set_xlabel("Age")
ax2.set_ylabel("Income")
st.pyplot(fig2)

# ===================== SIDEBAR INPUT =====================
st.sidebar.header("🔮 New Data Point")

age_input = st.sidebar.number_input(
    "Enter Age", min_value=1, max_value=100, value=30
)
income_input = st.sidebar.number_input(
    "Enter Income", min_value=1000, max_value=200000, value=30000
)

# ===================== CLUSTER ASSIGNMENT (CORRECT LOGIC) =====================
if st.sidebar.button("Assign Cluster"):

    new_point = np.array([[age_input, income_input]])
    new_point_scaled = scaler.transform(new_point)

    # Combine with existing data
    temp_X = np.vstack([X_scaled, new_point_scaled])

    temp_model = AgglomerativeClustering(
        n_clusters=2,
        metric="euclidean",
        linkage="average"
    )

    temp_labels = temp_model.fit_predict(temp_X)
    predicted_cluster = temp_labels[-1]

    st.success(f"✅ Assigned Cluster: {predicted_cluster}")

    st.progress(100 if predicted_cluster == 1 else 50)

# ===================== CLUSTERED DATA =====================
st.subheader("Clustered Dataset")
st.dataframe(df)

# ===================== BAR CHART =====================
st.subheader("Cluster Distribution")
cluster_counts = df["cluster"].value_counts().sort_index()
st.bar_chart(cluster_counts)

# ===================== DENDROGRAM =====================
st.subheader("Dendrogram")
Z = linkage(X_scaled, method="average")

fig3, ax3 = plt.subplots(figsize=(8, 5))
dendrogram(
    Z,
    labels=df["name"].values,
    leaf_rotation=90,
    leaf_font_size=10,
    ax=ax3
)
ax3.set_xlabel("Samples")
ax3.set_ylabel("Distance")
st.pyplot(fig3)
