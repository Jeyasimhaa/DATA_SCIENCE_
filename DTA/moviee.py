import streamlit as st
import pickle
import numpy as np
from pathlib import Path

# ===================== PAGE CONFIG =====================
st.set_page_config(page_title="Interest Prediction App", layout="centered")
st.title("🎯 Interest Prediction using Decision Tree")
st.write("Enter details to predict Interest")

# ===================== BASE DIRECTORY =====================
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "decision_tree_model.pkl"
ENCODER_PATH = BASE_DIR / "interest_label_encoder.pkl"

# ===================== LOAD MODEL =====================
if not MODEL_PATH.exists():
    st.error(f"❌ Model file not found: {MODEL_PATH.name}")
    st.stop()

if not ENCODER_PATH.exists():
    st.error(f"❌ Encoder file not found: {ENCODER_PATH.name}")
    st.stop()

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(ENCODER_PATH, "rb") as f:
    le = pickle.load(f)

# ===================== USER INPUTS =====================
age = st.number_input("Age", min_value=1, max_value=100, value=25)

gender = st.selectbox(
    "Gender",
    options=["Male", "Female"]
)

# Encode gender (same as training)
gender_encoded = 1 if gender == "Male" else 0

# ===================== PREDICTION =====================
if st.button("Predict Interest"):
    input_data = np.array([[age, gender_encoded]])
    prediction = model.predict(input_data)
    result = le.inverse_transform(prediction)

    st.success(f"✅ Predicted Interest: **{result[0]}**")
