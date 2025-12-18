import streamlit as st
import pickle
import os

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# App title
st.title("🛒 Shopping App")
st.write("Shopping Persons Prediction App")

# Load model and scaler
model = pickle.load(open(os.path.join(BASE_DIR, "student_final_model.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(BASE_DIR, "student_scaler.pkl"), "rb"))

# User inputs
age = st.number_input("Enter Age", min_value=1, max_value=100, value=30)
salary = st.number_input("Enter Estimated Salary", min_value=1000, max_value=200000, value=50000)

# Prediction
new_data = [[age, salary]]
new_data_scaled = scaler.transform(new_data)
prediction = model.predict(new_data_scaled)

# Display result
if prediction[0] == 1:
    st.success("✅ Person is likely to PURCHASE")
else:
    st.warning("❌ Person is NOT likely to purchase")
