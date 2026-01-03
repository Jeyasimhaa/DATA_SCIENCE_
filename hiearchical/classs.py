# ===================== IMPORT LIBRARIES =====================
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

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

# ===================== SAVE MODEL USING PICKLE =====================
with open("hierarchical_model.pkl", "wb") as f:
    pickle.dump(hc, f)

# ===================== LOAD MODEL =====================
with open("hierarchical_model.pkl", "rb") as f:
    loaded_model = pickle.load(f)

# ===================== SHOW CLUSTERED DATA =====================
st.subheader("Clustered Data")
st.dataframe(df)

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
