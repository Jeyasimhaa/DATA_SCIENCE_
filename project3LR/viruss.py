import streamlit as st
import numpy as np
import pickle

# -----------------------------
# Load model and scaler
# -----------------------------
with open("logistic_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# -----------------------------
# App UI
# -----------------------------
st.set_page_config(page_title="Heart Disease Prediction")
st.title("❤️ Heart Disease Prediction")
st.write("Predict 10-year risk of heart disease")

st.divider()

# -----------------------------
# User Inputs (ALL 14 FEATURES)
# -----------------------------
age = st.number_input("Age", 20, 100, 45)

sex = st.selectbox("Sex", ["Male", "Female"])
currentSmoker = st.selectbox("Current Smoker", ["Yes", "No"])
cigsPerDay = st.number_input("Cigarettes Per Day", 0, 70, 0)

BPMeds = st.selectbox("On BP Medication", ["Yes", "No"])
prevalentStroke = st.selectbox("History of Stroke", ["Yes", "No"])
prevalentHyp = st.selectbox("Hypertension", ["Yes", "No"])
diabetes = st.selectbox("Diabetes", ["Yes", "No"])

totChol = st.number_input("Total Cholesterol", value=200)
sysBP = st.number_input("Systolic BP", value=120)
diaBP = st.number_input("Diastolic BP", value=80)
BMI = st.number_input("BMI", value=25.0)
heartRate = st.number_input("Heart Rate", value=75)
glucose = st.number_input("Glucose", value=90)

# -----------------------------
# Convert categorical → numeric
# -----------------------------
sex = 1 if sex == "Male" else 0
currentSmoker = 1 if currentSmoker == "Yes" else 0
BPMeds = 1 if BPMeds == "Yes" else 0
prevalentStroke = 1 if prevalentStroke == "Yes" else 0
prevalentHyp = 1 if prevalentHyp == "Yes" else 0
diabetes = 1 if diabetes == "Yes" else 0

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):
    X = np.array([[
        age, sex, currentSmoker, cigsPerDay, BPMeds,
        prevalentStroke, prevalentHyp, diabetes,
        totChol, sysBP, diaBP, BMI, heartRate, glucose
    ]])

    X_scaled = scaler.transform(X)

    probability = model.predict_proba(X_scaled)[0][1]
    prediction = 1 if probability >= 0.3 else 0

    st.divider()
    st.write("### Result")
    st.write(f"Risk Probability: **{probability:.2f}**")

    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")
