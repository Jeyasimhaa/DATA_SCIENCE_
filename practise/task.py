import streamlit as st
import numpy as np
import pickle

# Load model and scaler
model = pickle.load(open("knn_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# App title
st.title("Social Network Ads Prediction")
st.write("KNN Model Prediction App")

# User inputs
age = st.number_input("Enter Age", min_value=18, max_value=100, value=30)
salary = st.number_input("Enter Estimated Salary", min_value=1000, max_value=200000, value=50000)

# Predict button
if st.button("Predict"):
    new_data = np.array([[age, salary]])
    new_data_scaled = scaler.transform(new_data)
    prediction = model.predict(new_data_scaled)

    if prediction[0] == 1:
        st.success("✅ User is likely to PURCHASE")
    else:
        st.warning("❌ User is NOT likely to purchase")
