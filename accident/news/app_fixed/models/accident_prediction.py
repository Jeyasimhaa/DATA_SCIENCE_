# ==========================================
# AI Road Accident Prediction System
# Single Python File
# ==========================================

import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "..", "dataset", "accident_data.csv")

# ==========================================
# STEP 1 : Load Dataset
# ==========================================

df = pd.read_csv(DATASET_PATH)

print("Dataset Loaded Successfully")
print(df.head())

# ==========================================
# STEP 2 : Encode Categorical Columns
# ==========================================
# These must match the actual columns in accident_data.csv:
# Weather, Speed, Road_Condition, Traffic, Time, Latitude, Longitude, Accident

categorical_columns = [
    "Weather",
    "Road_Condition",
    "Traffic",
    "Time"
]

label_encoders = {}

for column in categorical_columns:
    encoder = LabelEncoder()
    df[column] = encoder.fit_transform(df[column])
    label_encoders[column] = encoder

# ==========================================
# STEP 3 : Split Features and Target
# ==========================================

X = df.drop("Accident", axis=1)
y = df["Accident"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ==========================================
# STEP 4 : Train Random Forest Model
# ==========================================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

print("\nModel Training Completed")

# ==========================================
# STEP 5 : Evaluate Model
# ==========================================

prediction = model.predict(X_test)

accuracy = accuracy_score(y_test, prediction)

print("\nAccuracy :", round(accuracy * 100, 2), "%")

print("\nClassification Report")
print(classification_report(y_test, prediction, zero_division=0))

# ==========================================
# STEP 6 : Save Model
# ==========================================

joblib.dump(model, os.path.join(BASE_DIR, "accident_model.pkl"))
joblib.dump(label_encoders, os.path.join(BASE_DIR, "accident_label_encoders.pkl"))

print("\nModel Saved Successfully")

# ==========================================
# STEP 7 : Sample User Input
# ==========================================

sample = {
    "Weather": "Rain",
    "Speed": 95,
    "Road_Condition": "Wet",
    "Traffic": "Heavy",
    "Time": "Night",
    "Latitude": 13.0827,
    "Longitude": 80.2707
}

sample_df = pd.DataFrame([sample])

# Encode categorical values
for column in categorical_columns:
    sample_df[column] = label_encoders[column].transform(sample_df[column])

# ==========================================
# STEP 8 : Predict Accident Risk
# ==========================================

prediction = model.predict(sample_df)

probability = model.predict_proba(sample_df)

risk = probability[0][1] * 100

print("\n==============================")
print("ACCIDENT PREDICTION RESULT")
print("==============================")

print(f"Accident Prediction : {'YES' if prediction[0] == 1 else 'NO'}")
print(f"Accident Probability : {risk:.2f}%")

# ==========================================
# STEP 9 : Risk Level
# ==========================================

if risk < 30:
    level = "LOW RISK"
elif risk < 60:
    level = "MEDIUM RISK"
elif risk < 80:
    level = "HIGH RISK"
else:
    level = "CRITICAL RISK"

print("Risk Level :", level)

# ==========================================
# STEP 10 : Safety Recommendation
# ==========================================

print("\nSafety Recommendation")

if level == "LOW RISK":
    print("Drive normally and follow traffic rules.")

elif level == "MEDIUM RISK":
    print("Maintain a safe distance and reduce speed if necessary.")

elif level == "HIGH RISK":
    print("Reduce speed immediately.")
    print("Turn on headlights if visibility is poor.")
    print("Be alert and avoid sudden braking.")

else:
    print("WARNING! High accident probability.")
    print("Reduce speed immediately.")
    print("Consider taking an alternate route.")
    print("Maintain a safe following distance.")
