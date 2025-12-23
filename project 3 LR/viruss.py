import streamlit as st
import pickle
import numpy as np

# ---------------------------------
# Load the trained model & scaler
# ---------------------------------
with open("logistic_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ---------------------------------
# Streamlit UI
# ---------------------------------
st.set_page_config(page_title="Logistic Regression App", layout="centered")

st.title("Customer Purchase Prediction")
st.write("Logistic Regression Deployment using Streamlit")

# ---------------------------------
# User Inputs
# ---------------------------------
age = st.number_input(
    "Enter Age",
    min_value=1,
    max_value=100,
    value=30
)

salary = st.number_input(
    "Enter Estimated Salary",
    min_value=1000,
    max_value=200000,
    value=50000
)

# ---------------------------------
# Prediction
# ---------------------------------
if st.button("Predict"):

    input_data = np.array([[age, salary]])

    # Apply same scaling used during training
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.success("✅ Customer WILL purchase")
    else:
        st.error("❌ Customer will NOT purchase")
