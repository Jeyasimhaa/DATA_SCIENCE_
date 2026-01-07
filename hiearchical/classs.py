# ===================== IMPORT LIBRARIES =====================
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import numpy as np
from pathlib import Path

from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage

# ===================== PAGE CONFIG =====================
st.set_page_config(page_title="Hierarchical Clustering App", layout="centered")
st.title("📊 Hierarchical Clustering (Age vs Income)")

# ===================== FILE PATH (CLOUD SAFE) =====================
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "income.xlsx"

# ===================== LOAD DATA =====================
@st.cache_data
def load_data(file_path):
    if not file_path.exists():
        st.error(f"❌ File not found: {file_path.name}")
        st.stop()
    return pd.read_excel(file_path, names=["name", "age", "income"])

df = load_data(DATA_FILE)

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
sns.scatterplot(x=df["age"], y=df["income"], s=60, ax=ax1)
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
with open(BASE_DIR / "hierarchical_model.pkl", "wb") as f:
    pickle.dump(hc, f)

# ===================== CLUSTER VISUALIZATION =====================
st.subheader("Cluster Visualization")
fig2, ax2 = plt.subplots()
sns.scatterplot(
    x=df["age"],
    y=df["income"],
    hue=df["cluster"],
    palette="Set1",
    s=70,
    ax=ax2
)
ax2.set_xlabel("Age")
ax2.set_ylabel("Income")
st.pyplot(fig2)

# ===================== SIDEBAR INPUT =====================
st.sidebar.header("🔮 Assign Cluster")

age_input = st.sidebar.number_input("Enter Age", 1, 100, 30)
income_input = st.sidebar.number_input("Enter Income", 1000, 200000, 30000)

# ===================== CLUSTER ASSIGNMENT =====================
if st.sidebar.button("Assign Cluster"):
    new_point = np.array([[age_input, income_input]])
    new_point_scaled = scaler.transform(new_point)

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

# ===================== CLUSTER DISTRIBUTION =====================
st.subheader("Cluster Distribution")
st.bar_chart(df["cluster"].value_counts().sort_index())

# ===================== DENDROGRAM =====================
st.subheader("Dendrogram")
Z = linkage(X_scaled, method="average")

fig3, ax3 = plt.subplots(figsize=(8, 5))
dendrogram(
    Z,
    labels=df["name"].astype(str).values,
    leaf_rotation=90,
    leaf_font_size=10,
    ax=ax3
)
ax3.set_xlabel("Samples")
ax3.set_ylabel("Distance")
st.pyplot(fig3)
