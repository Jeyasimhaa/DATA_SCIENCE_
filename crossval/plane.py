import streamlit as st
import pandas as pd
import pickle

# --------------------------------------------------
# Load model safely (works for both pipeline-only
# pickle and dict-based pickle)
# --------------------------------------------------
with open("travelinsurancemodel.pkl", "rb") as f:
    bundle = pickle.load(f)

if isinstance(bundle, dict):
    model = bundle["model"]
    feature_names = bundle["features"]
else:
    model = bundle
    feature_names = model.feature_names_in_.tolist()

# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------
st.set_page_config(page_title="Travel Insurance Prediction", layout="centered")
st.title("✈️ Travel Insurance Prediction App")
st.write("Predict whether a customer will buy travel insurance")

# --------------------------------------------------
# User Inputs
# --------------------------------------------------
age = st.number_input("Age", min_value=18, max_value=100, value=30)
annual_income = st.number_input("Annual Income", min_value=0, value=500000)
family_members = st.number_input("Family Members", min_value=1, max_value=10, value=3)

employment = st.selectbox(
    "Employment Type",
    ["Private Sector/Self Employed", "Government Sector"]
)

graduate = st.selectbox("Graduate", ["Yes", "No"])
chronic = st.selectbox("Chronic Disease", ["Yes", "No"])
frequent = st.selectbox("Frequent Flyer", ["Yes", "No"])
abroad = st.selectbox("Ever Travelled Abroad", ["Yes", "No"])

# --------------------------------------------------
# Create raw input dataframe (MATCH TRAINING DATA)
# --------------------------------------------------
raw_input = pd.DataFrame({
    "Age": [age],
    "AnnualIncome": [annual_income],
    "FamilyMembers": [family_members],
    "Employment Type": [employment],
    "GraduateOrNot": [1 if graduate == "Yes" else 0],
    "ChronicDiseases": [1 if chronic == "Yes" else 0],
    "FrequentFlyer": [1 if frequent == "Yes" else 0],
    "EverTravelledAbroad": [1 if abroad == "Yes" else 0]
})

# --------------------------------------------------
# Apply SAME preprocessing as training
# --------------------------------------------------
input_df = pd.get_dummies(raw_input, drop_first=True)

# Align columns exactly with training features
input_df = input_df.reindex(columns=feature_names, fill_value=0)

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.success(f"✅ Customer WILL buy travel insurance\n\nProbability: {probability:.2f}")
    else:
        st.error(f"❌ Customer will NOT buy travel insurance\n\nProbability: {probability:.2f}")
