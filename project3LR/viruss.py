
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(page_title="Heart Disease Prediction", layout="wide")
st.title("❤️ Framingham Heart Disease Prediction")

# Load dataset (RELATIVE PATH)
@st.cache_data
def load_data():
    return pd.read_csv("framingham_heart_disease.csv")

df = load_data()

st.subheader("📄 Dataset Preview")
st.dataframe(df.head())

st.subheader("📊 Dataset Info")
st.write(df.describe())

# Handle missing values
df = df.drop(columns=["education"])

cols = ['cigsPerDay', 'BPMeds', 'totChol', 'BMI', 'glucose', 'heartRate']
df[cols] = df[cols].fillna(df[cols].mean())

st.success("Missing values handled ✅")

# Split data
X = df.drop(columns=["TenYearCHD"])
y = df["TenYearCHD"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)

st.subheader("📈 Model Performance")
st.write("Accuracy:", accuracy_score(y_test, y_pred))
st.text(classification_report(y_test, y_pred))

# Correlation heatmap
st.subheader("🔥 Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(df.corr(), cmap="coolwarm", ax=ax)
st.pyplot(fig)
