import streamlit as st
import pandas as pd
import pickle

# Load trained model
with open("travel_insurance_model.pkl", "rb") as file:
    model = pickle.load(file)

st.set_page_config(page_title="Travel Insurance Prediction", layout="centered")

st.title("✈️ Travel Insurance Prediction App")
st.write("Predict whether a customer will buy travel insurance")

# ---- User Input ----
age = st.number_input("Age", min_value=18, max_value=100, value=30)
employment_type = st.selectbox("Employment Type", ["Private Sector", "Government Sector"])
graduate = st.selectbox("Graduate", ["Yes", "No"])
annual_income = st.number_input("Annual Income", min_value=0, value=500000)
family_members = st.number_input("Family Members", min_value=1, max_value=10, value=3)
chronic_disease = st.selectbox("Chronic Disease", ["Yes", "No"])
frequent_flyer = st.selectbox("Frequent Flyer", ["Yes", "No"])
ever_travelled_abroad = st.selectbox("Ever Travelled Abroad", ["Yes", "No"])

# ---- Convert inputs to dataframe ----
input_data = pd.DataFrame({
    "Age": [age],
    "AnnualIncome": [annual_income],
    "FamilyMembers": [family_members],
    "Employment Type_Private Sector/Self Employed": [1 if employment_type == "Private Sector" else 0],
    "Graduate_Yes": [1 if graduate == "Yes" else 0],
    "ChronicDiseases_Yes": [1 if chronic_disease == "Yes" else 0],
    "FrequentFlyer_Yes": [1 if frequent_flyer == "Yes" else 0],
    "EverTravelledAbroad_Yes": [1 if ever_travelled_abroad == "Yes" else 0]
})

# ---- Prediction ----
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(f"✅ Customer is likely to buy travel insurance (Probability: {probability:.2f})")
    else:
        st.error(f"❌ Customer is not likely to buy travel insurance (Probability: {probability:.2f})")
