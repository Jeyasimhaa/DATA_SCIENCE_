import os
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "accident_model.pkl"))
encoders = joblib.load(os.path.join(BASE_DIR, "accident_label_encoders.pkl"))

sample = {
    "Weather": "Rain",
    "Speed": 95,
    "Road_Condition": "Wet",
    "Traffic": "Heavy",
    "Time": "Night",
    "Latitude": 13.0827,
    "Longitude": 80.2707
}

df = pd.DataFrame([sample])

for col in encoders:
    df[col] = encoders[col].transform(df[col])

prediction = model.predict(df)
probability = model.predict_proba(df)

print("Prediction :", prediction[0])
print("Probability :", probability[0][1] * 100)
