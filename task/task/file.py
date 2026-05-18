import streamlit as st
import pickle
import pandas as pd

# LOAD MODEL
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("Employee Retention Prediction")

# INPUTS
satisfaction_level = st.slider("Satisfaction Level", 0.0, 1.0, 0.5)

last_evaluation = st.slider("Last Evaluation", 0.0, 1.0, 0.5)

number_project = st.number_input("Number of Projects", 1, 10, 3)

average_montly_hours = st.number_input("Average Monthly Hours", 50, 400, 150)

time_spend_company = st.number_input("Years in Company", 1, 10, 3)

Work_accident = st.selectbox("Work Accident", [0,1])

promotion_last_5years = st.selectbox("Promotion Last 5 Years", [0,1])

# INPUT DATA
input_data = pd.DataFrame([[
    satisfaction_level,
    last_evaluation,
    number_project,
    average_montly_hours,
    time_spend_company,
    Work_accident,
    promotion_last_5years,
    0,0,0,0,0,0,0,0,0,
    1,0
]], columns=model.feature_names_in_)

# PREDICT
if st.button("Predict"):

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("Employee Will Leave")
    else:
        st.success("Employee Will Stay")