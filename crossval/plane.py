import streamlit as st
import pandas as pd
import pickle

# =========================
# Load model & scaler
# =========================
with open("travel_insurance_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("✈️ Travel Insurance Prediction App")

st.write("Enter customer details to predict Travel Insurance purchase")

# =========================
# User Inputs
# =========================
age = st.number_input("Age", min_value=18, max_value=100, value=30)
employment = st.selectbox(
    "Employment Type",
    ["Government Sector", "Private Sector/Self Employed"]
)
graduate = st.selectbox("Graduate", ["Yes", "No"])
income = st.number_input("Annual Income", min_value=0, value=500000)
family = st.number_input("Family Members", min_value=1, max_value=10, value=3)
chronic = st.selectbox("Chronic Diseases", [0, 1])
flyer = st.selectbox("Frequent Flyer", ["Yes", "No"])
abroad = st.selectbox("Ever Travelled Abroad", ["Yes", "No"])

# =========================
# Create input DataFrame
# =========================
input_data = pd.DataFrame({
    "Age": [age],
    "AnnualIncome": [income],
    "FamilyMembers": [family],
    "ChronicDiseases": [chronic],
    "Employment Type_Private Sector/Self Employed": [1 if employment == "Private Sector/Self Employed" else 0],
    "GraduateOrNot_Yes": [1 if graduate == "Yes" else 0],
    "FrequentFlyer_Yes": [1 if flyer == "Yes" else 0],
    "EverTravelledAbroad_Yes": [1 if abroad == "Yes" else 0],
})

# =========================
# Scale input
# =========================
input_scaled = scaler.transform(input_data)

# =========================
# Prediction
# =========================
if st.button("Predict"):
    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.success("✅ Customer is likely to buy Travel Insurance")
    else:
        st.error("❌ Customer is NOT likely to buy Travel Insurance")
