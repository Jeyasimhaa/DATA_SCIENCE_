import streamlit as st
import pickle
import numpy as np
import os

# --------------------------------------------------
# Page configuration (MUST be first Streamlit command)
# --------------------------------------------------
st.set_page_config(
    page_title="Cancer Prediction App",
    layout="centered"
)

# --------------------------------------------------
# Load model safely using absolute path
# --------------------------------------------------
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "cancer_rf_model.pkl")

    with open(model_path, "rb") as file:
        model = pickle.load(file)

    return model

model = load_model()

# --------------------------------------------------
# App UI
# --------------------------------------------------
st.title("🩺 Cancer Prediction App")
st.write("Enter patient feature values to predict diagnosis")

# --------------------------------------------------
# Input fields (MATCH your training features!)
# --------------------------------------------------
radius_mean = st.number_input("Radius Mean", min_value=0.0, value=10.0)
texture_mean = st.number_input("Texture Mean", min_value=0.0, value=10.0)
perimeter_mean = st.number_input("Perimeter Mean", min_value=0.0, value=50.0)
area_mean = st.number_input("Area Mean", min_value=0.0, value=500.0)
smoothness_mean = st.number_input("Smoothness Mean", min_value=0.0, value=0.1)

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if st.button("🔍 Predict Cancer"):
    input_data = np.array([[
        radius_mean,
        texture_mean,
        perimeter_mean,
        area_mean,
        smoothness_mean
    ]])

    prediction = model.predict(input_data)

    if prediction[0] == 'M':
        st.error("🔴 **Prediction: Malignant (Cancer Detected)**")
    else:
        st.success("🟢 **Prediction: Benign (No Cancer Detected)**")
