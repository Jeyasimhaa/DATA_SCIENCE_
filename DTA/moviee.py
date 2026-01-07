import streamlit as st
import pickle
import numpy as np

# Load model
with open("decision_tree_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load label encoder
with open("interest_label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# Page config
st.set_page_config(page_title="Interest Prediction App", layout="centered")

st.title("🎯 Interest Prediction using Decision Tree")
st.write("Enter details to predict Interest")

# User inputs
age = st.number_input("Age", min_value=1, max_value=100, value=25)

gender = st.selectbox(
    "Gender",
    options=["Male", "Female"]
)

# Encode gender manually (same logic as training)
gender_encoded = 1 if gender == "Male" else 0

# Prediction button
if st.button("Predict Interest"):
    input_data = np.array([[age, gender_encoded]])
    prediction = model.predict(input_data)
    result = le.inverse_transform(prediction)

    st.success(f"✅ Predicted Interest: **{result[0]}**")
