# ===================== IMPORT LIBRARIES =====================
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import numpy as np

from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# ===================== PAGE CONFIG =====================
st.set_page_config(page_title="Hierarchical Clustering App", layout="centered")
st.title("📊 Hierarchical Clustering (Age vs Income)")

# ===================== LOAD DATA =====================
df = pd.read_excel(
    r"C:\Users\rjey0\Downloads\practise\hiearchical\income (1).xlsx",
    names=["name", "age", "income"]
)

st.subheader("Dataset")
st.dataframe(df)

# ===================== SCATTER PLOT =====================
st.subheader("Scatter Plot")
fig1, ax1 = plt.subplots()
sns.scatterplot(data=df, x="age", y="income", s=30, ax=ax1)
st.pyplot(fig1)

# ===================== FEATURES =====================
X = df[["age", "income"]]

# ===================== HIERARCHICAL CLUSTERING =====================
hc = AgglomerativeClustering(
    n_clusters=2,
    metric="euclidean",
    linkage="average"
)

df["cluster"] = hc.fit_predict(X)

# ===================== SAVE MODEL =====================
with open("hierarchical_model.pkl", "wb") as f:
    pickle.dump(hc, f)

# ===================== SIDEBAR INPUT =====================
st.sidebar.header("🔮 Predict Cluster")

age_input = st.sidebar.number_input("Enter Age", min_value=1, max_value=100, value=30)
income_input = st.sidebar.number_input("Enter Income", min_value=1000, max_value=200000, value=30000)

# ===================== PREDICTION LOGIC =====================
if st.sidebar.button("Predict Cluster"):

    # Add new point to dataset
    new_point = pd.DataFrame([[age_input, income_input]], columns=["age", "income"])
    temp_X = pd.concat([X, new_point], ignore_index=True)

    temp_model = AgglomerativeClustering(
        n_clusters=2,
        metric="euclidean",
        linkage="average"
    )

    temp_labels = temp_model.fit_predict(temp_X)

    predicted_cluster = temp_labels[-1]

    st.success(f"✅ Predicted Cluster: {predicted_cluster}")

    # ===================== PROGRESS BAR =====================
    st.subheader("Prediction Confidence Bar")
    progress_value = 50 if predicted_cluster == 0 else 100
    st.progress(progress_value)

# ===================== CLUSTERED DATA =====================
st.subheader("Clustered Data")
st.dataframe(df)

# ===================== BAR CHART (CLUSTER COUNTS) =====================
st.subheader("Cluster Distribution")
cluster_counts = df["cluster"].value_counts().sort_index()
st.bar_chart(cluster_counts)

# ===================== DENDROGRAM =====================
st.subheader("Dendrogram")
Z = linkage(X, method="average")

fig2, ax2 = plt.subplots(figsize=(8, 5))
dendrogram(
    Z,
    labels=df["name"].values,
    leaf_rotation=90,
    leaf_font_size=10,
    ax=ax2
)
ax2.set_xlabel("Samples")
ax2.set_ylabel("Distance")
st.pyplot(fig2)
