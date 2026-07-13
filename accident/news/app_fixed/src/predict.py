import os
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")

# =====================================
# Load Model
# =====================================

model = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))

# =====================================
# Load Label Encoders
# =====================================

encoders = joblib.load(os.path.join(MODELS_DIR, "label_encoders.pkl"))

# =====================================
# Get User Input
# =====================================

print("=" * 60)
print(" AI Road Accident Prediction System ")
print("=" * 60)

weather = input("Weather (Rain/Sunny/Cloudy/Fog): ")

speed = float(input("Vehicle Speed (km/h): "))

road = input("Road Condition (Dry/Wet): ")

traffic = input("Traffic (Low/Medium/Heavy): ")

time = input("Time (Morning/Afternoon/Evening/Night): ")

latitude = float(input("Latitude: "))

longitude = float(input("Longitude: "))

# =====================================
# Encode Inputs
# =====================================

weather = encoders["Weather"].transform([weather])[0]

road = encoders["Road_Condition"].transform([road])[0]

traffic = encoders["Traffic"].transform([traffic])[0]

time = encoders["Time"].transform([time])[0]

# =====================================
# Create DataFrame
# =====================================

sample = pd.DataFrame({

    "Weather": [weather],

    "Speed": [speed],

    "Road_Condition": [road],

    "Traffic": [traffic],

    "Time": [time],

    "Latitude": [latitude],

    "Longitude": [longitude]

})

# =====================================
# Prediction
# =====================================

prediction = model.predict(sample)[0]

probability = model.predict_proba(sample)[0][1]

probability = probability * 100

# =====================================
# Risk Level
# =====================================

if probability < 40:

    risk = "LOW"

    advice = "Road conditions appear safe. Continue driving carefully."

elif probability < 70:

    risk = "MEDIUM"

    advice = "Drive carefully. Reduce speed and stay alert."

else:

    risk = "HIGH"

    advice = "High accident risk detected. Slow down immediately and drive with extra caution."

# =====================================
# Display Result
# =====================================

print("\n")

print("=" * 60)

print("Prediction Result")

print("=" * 60)

print(f"Accident Probability : {probability:.2f}%")

print(f"Risk Level           : {risk}")

print(f"Accident Prediction  : {'YES' if prediction == 1 else 'NO'}")

print("\nRecommendation")

print(advice)

print("=" * 60)
