# -------------------------------
# Imports
# -------------------------------
import streamlit as st
import pandas as pd
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(
    page_title="Cancer Diagnosis Prediction",
    layout="centered"
)

st.title("🧬 Cancer Diagnosis Prediction")
st.write("Random Forest model using patient data")

# -------------------------------
# Load dataset & train model
# -------------------------------
@st.cache_resource
def load_and_train_model():
    BASE_DIR = Path(__file__).resolve().parent
    data_path = BASE_DIR / "cancer_data.xlsx"

    if not data_path.exists():
        st.error("❌ Dataset file 'cancer_data.xlsx' not found in the app directory.")
        st.stop()

    # Read Excel file
    df = pd.read_excel(data_path)

    # Expecting a column named 'diagnosis'
    if "diagnosis" not in df.columns:
        st.error("❌ Dataset must contain a 'diagnosis' column.")
        st.stop()

    X = df.drop("diagnosis", axis=1)
    y = df["diagnosis"]

    preprocessing = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline(steps=[
        ("preprocessing", preprocessing),
        ("model", model)
    ])

    pipeline.fit(X, y)
    return pipeline, X.columns


model, feature_names = load_and_train_model()

# -------------------------------
# User input section
# -------------------------------
st.header("🔍 Enter Patient Information")

age = st.slider("Age", 18, 90, 45)
gender = st.selectbox("Gender (0 = Female, 1 = Male)", [0, 1])
bmi = st.number_input("BMI", 10.0, 50.0, 25.0)
smoking = st.selectbox("Smoking (0 = No, 1 = Yes)", [0, 1])
genetic_risk = st.slider("Genetic Risk (0–3)", 0, 3, 1)
physical_activity = st.slider("Physical Activity (hours/week)", 0, 10, 4)
alcohol_intake = st.number_input("Alcohol Intake", 0.0, 10.0, 2.0)
cancer_history = st.selectbox("Family Cancer History (0 = No, 1 = Yes)", [0, 1])

# Input dataframe must match training columns
input_data = pd.DataFrame({
    "age": [age],
    "gender": [gender],
    "bmi": [bmi],
    "smoking": [smoking],
    "genetic_risk": [genetic_risk],
    "physical_activity": [physical_activity],
    "alcohol_intake": [alcohol_intake],
    "cancer_history": [cancer_history]
})

# -------------------------------
# Prediction
# -------------------------------
st.subheader("🔮 Prediction")

if st.button("Predict Diagnosis"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"⚠️ **Diagnosis: POSITIVE**\n\nProbability: **{probability:.2f}**")
    else:
        st.success(f"✅ **Diagnosis: NEGATIVE**\n\nProbability: **{probability:.2f}**")

# -------------------------------
# Feature importance
# -------------------------------
st.subheader("📊 Feature Importance")

importances = model.named_steps["model"].feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

st.bar_chart(importance_df.set_index("Feature"))
