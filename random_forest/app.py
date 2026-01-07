import streamlit as st
import pickle
import numpy as np
from pathlib import Path

# ===================== PAGE CONFIG (MUST BE FIRST) =====================
st.set_page_config(page_title="Cancer Prediction App", layout="centered")

# ===================== PATH HANDLING =====================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "cancer_rf_model.pkl"

# ===================== LOAD MODEL =====================
if not MODEL_PATH.exists():
    st.error(f"❌ Model file not found: {MODEL_PATH.name}")
    st.stop()

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

# ===================== APP UI =====================
st.title("🩺 Cancer Prediction App")
st.write("Enter patient feature values to predict diagnosis")

radius_mean = st.number_input("Radius Mean", value=10.0)
texture_mean = st.number_input("Texture Mean", value=10.0)
perimeter_mean = st.number_input("Perimeter Mean", value=50.0)
area_mean = st.number_input("Area Mean", value=500.0)
smoothness_mean = st.number_input("Smoothness Mean", value=0.1)

if st.button("Predict Cancer"):
    input_data = np.array([[
        radius_mean,
        texture_mean,
        perimeter_mean,
        area_mean,
        smoothness_mean
    ]])

    prediction = model.predict(input_data)

    if prediction[0] == 'M':
        st.error("🔴 Prediction: Malignant (Cancer)")
    else:
        st.success("🟢 Prediction: Benign (No Cancer)")
