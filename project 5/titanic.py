import streamlit as st
import pandas as pd
import pickle

# Load model and encoder
with open("titanic_rf_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("sex_label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

st.set_page_config(page_title="Titanic Survival Prediction", layout="centered")

st.title("🚢 Titanic Survival Prediction App")
st.write("Enter passenger details to predict survival")

# User Inputs
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.number_input("Age", min_value=0.0, max_value=100.0, value=30.0)
sibsp = st.number_input("Siblings/Spouses aboard", min_value=0, max_value=10, value=0)
parch = st.number_input("Parents/Children aboard", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare", min_value=0.0, value=32.0)

# Encode Sex
sex_encoded = le.transform([sex])[0]

# Create input DataFrame
input_data = pd.DataFrame({
    "Pclass": [pclass],
    "Sex": [sex_encoded],
    "Age": [age],
    "SibSp": [sibsp],
    "Parch": [parch],
    "Fare": [fare]
})

# Prediction
if st.button("Predict Survival"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][prediction]

    if prediction == 1:
        st.success(f"✅ Passenger is likely to SURVIVE (Confidence: {probability:.2f})")
    else:
        st.error(f"❌ Passenger is likely to NOT SURVIVE (Confidence: {probability:.2f})")
