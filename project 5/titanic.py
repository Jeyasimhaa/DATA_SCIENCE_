import streamlit as st
import pandas as pd
import pickle
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Titanic Survival Prediction",
    layout="centered"
)

st.title("🚢 Titanic Survival Prediction App")
st.write("Enter passenger details to predict survival")

# ---------------- LOAD MODEL SAFELY ----------------
MODEL_PATH = "titanic_rf_model.pkl"

if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file 'titanic_rf_model.pkl' not found. Upload it to the same folder.")
    st.stop()

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# ---------------- USER INPUTS ----------------
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.number_input("Age", min_value=0.0, max_value=100.0, value=30.0)
sibsp = st.number_input("Siblings / Spouses aboard", min_value=0, max_value=10, value=0)
parch = st.number_input("Parents / Children aboard", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare", min_value=0.0, value=32.0)

# ---------------- ENCODE SEX (NO ENCODER FILE NEEDED) ----------------
sex_encoded = 1 if sex == "male" else 0

# ---------------- INPUT DATAFRAME ----------------
input_data = pd.DataFrame({
    "Pclass": [pclass],
    "Sex": [sex_encoded],
    "Age": [age],
    "SibSp": [sibsp],
    "Parch": [parch],
    "Fare": [fare]
})

# ---------------- PREDICTION ----------------
if st.button("Predict Survival"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][prediction]

    if prediction == 1:
        st.success(f"✅ Passenger is likely to SURVIVE (Confidence: {probability:.2f})")
    else:
        st.error(f"❌ Passenger is likely to NOT SURVIVE (Confidence: {probability:.2f})")
