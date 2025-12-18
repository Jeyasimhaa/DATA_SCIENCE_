import streamlit as st 
import pickle 
import numpy as np 
import os 

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
model = pickle.load(open(os.path.join(BASE_DIR, "student_final_model.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(BASE_DIR, "scaler.pkl"), "rb")) 

st.title(" Y Student Final Score Prediction App")  
st.write("Predict Final Score using academic and lifestyle factors") 

study_hours = st.number_input("Study Hours per Week", 0.0, 100.0, 10.0) 
attendance = st.number_input("Attendance Percentage", 0.0, 100.0, 75.0)  
previous_score = st.number_input("Previous Semester Score", 0.0, 100.0, 60.0) 
sleep_hours = st.number_input("Sleep Hours per Day", 0.0, 12.0, 7.0) 
travel_time = st.number_input("Travel Time (hours)", 0.0, 5.0, 1.0) 
library_usage = st.number_input("Library Usage per Week", 0.0, 20.0, 3.0) 

features = np.array([[study_hours,attendance,previous_score,
                      sleep_hours,travel_time,library_usage]]) 
if st.button("Predict Final Score"): 
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled) 
    st.success(f"Predicted Final Score: {prediction[0]:.2f}") 
