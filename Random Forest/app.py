import streamlit as st
import pickle
import numpy as np
import os
# Page configuration (MUST be first)
st.set_page_config(page_title="Cancer Prediction App", layout="centered")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "cancer_rf_model.pkl")


with open("cancer_rf_model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("🩺 Cancer Prediction App")
st.write("Enter patient feature values to predict diagnosis")

# ⚠️ Enter ALL features used in training (example placeholders)
radius_mean = st.number_input("Radius Mean", value=10.0)
texture_mean = st.number_input("Texture Mean", value=10.0)
perimeter_mean = st.number_input("Perimeter Mean", value=50.0)
area_mean = st.number_input("Area Mean", value=500.0)
smoothness_mean = st.number_input("Smoothness Mean", value=0.1)

# ⚠️ Add remaining features here if your model was trained on more columns

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
