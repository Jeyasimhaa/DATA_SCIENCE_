import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="My Streamlit App", layout="wide")
st.title("📊 Data Science / ML Streamlit App")

df = pd.read_csv(
    r"C:\Users\rjey0\Downloads\practise\project3LR\framingham_heart_disease.csv"
)

st.subheader("📄 Dataset Preview")
st.dataframe(df.head())

st.subheader("📌 Dataset Info")
st.write(df.describe())

target = st.selectbox("Select Target Column", df.columns)

X = df.drop(columns=[target])
y = df[target]

st.subheader("🔍 Feature Columns")
st.write(X.columns.tolist())

st.subheader("📈 Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

st.success("Data loaded successfully ✅")
