import streamlit as st
import pickle
import numpy as np
import os

# --------------------------------------------------
# Page config (MUST be first Streamlit command)
# --------------------------------------------------
st.set_page_config(
    page_title="Interest Prediction App",
    layout="centered"
)

# --------------------------------------------------
# Load model & encoder safely using absolute paths
# --------------------------------------------------
@st.cache_resource
def load_artifacts():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    model_path = os.path.join(base_dir, "decision_tree_model.pkl")
    encoder_path = os.path.join(base_dir, "interest_label_encoder.pkl")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(encoder_path, "rb") as f:
        le = pickle.load(f)

    return model, le

model, le = load_artifacts()

# --------------------------------------------------
# App UI
# --------------------------------------------------
st.title("🎯 Interest Prediction using Decision Tree")
st.write("Enter details to predict Interest")

# --------------------------------------------------
# User inputs
# --------------------------------------------------
age = st.number_input("Age", min_value=1, max_value=100, value=25)

gender = st.selectbox(
    "Gender",
    options=["Male", "Female"]
)

# Encoding gender (must match training)
gender_encoded = 1 if gender == "Male" else 0

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if st.button("🔍 Predict Interest"):
    input_data = np.array([[age, gender_encoded]])

    prediction = model.predict(input_data)
    result = le.inverse_transform(prediction)

    st.success(f"✅ Predicted Interest: **{result[0]}**")
