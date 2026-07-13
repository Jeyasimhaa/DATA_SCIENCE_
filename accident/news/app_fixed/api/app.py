from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os

# ===========================
# Import Custom Modules
# ===========================

from notification import (
    desktop_notification,
    send_email,
    send_sms
)

from report_generator import generate_report
from weather_service import get_weather
from database import create_database, save_prediction

# Make sure the history table exists before we try to use it
create_database()

# ===========================
# Initialize Flask
# ===========================

app = Flask(__name__)
CORS(app)

# ===========================
# Load AI Model
# ===========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")

MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
ENCODERS_PATH = os.path.join(MODEL_DIR, "label_encoders.pkl")

if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODERS_PATH):
    raise FileNotFoundError(
        "Model files not found. Run the training pipeline first:\n"
        "  cd src\n"
        "  python preprocessing.py\n"
        "  python train_random_forest.py\n"
        "  python train_xgboost.py\n"
        "  python evaluate_models.py"
    )

model = joblib.load(MODEL_PATH)
encoders = joblib.load(ENCODERS_PATH)

# ===========================
# Home Route
# ===========================

@app.route("/")
def home():
    return jsonify({
        "message": "AI Road Accident Prediction API",
        "status": "Running"
    })

# ===========================
# Health Check
# ===========================

@app.route("/health")
def health():
    return jsonify({
        "status": "Healthy"
    })

# ===========================
# Prediction Route
# ===========================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # -------------------------
        # Receive JSON Data
        # -------------------------

        data = request.get_json()

        speed = data.get("speed")
        road_condition = data.get("road_condition")
        traffic = data.get("traffic")
        time = data.get("time")
        latitude = data.get("latitude")
        longitude = data.get("longitude")

        # -------------------------
        # Get Live Weather
        # -------------------------

        weather_data = get_weather(latitude, longitude)

        weather_name = weather_data["weather"]
        temperature = weather_data["temperature"]
        humidity = weather_data["humidity"]
        wind_speed = weather_data["wind_speed"]

        # -------------------------
        # Encode Features
        # -------------------------

        weather = encoders["Weather"].transform([weather_name])[0]
        road = encoders["Road_Condition"].transform([road_condition])[0]
        traffic_enc = encoders["Traffic"].transform([traffic])[0]
        time_enc = encoders["Time"].transform([time])[0]

        # -------------------------
        # Create DataFrame
        # -------------------------

        sample = pd.DataFrame({
            "Weather": [weather],
            "Speed": [speed],
            "Road_Condition": [road],
            "Traffic": [traffic_enc],
            "Time": [time_enc],
            "Latitude": [latitude],
            "Longitude": [longitude]
        })

        # -------------------------
        # AI Prediction
        # -------------------------

        prediction = model.predict(sample)[0]
        probability = model.predict_proba(sample)[0][1] * 100

        # -------------------------
        # Risk Analysis
        # -------------------------

        if probability < 40:

            risk = "LOW"
            recommendation = "Safe road conditions."

        elif probability < 70:

            risk = "MEDIUM"
            recommendation = "Drive carefully."

        else:

            risk = "HIGH"
            recommendation = "High accident risk. Reduce speed."

            desktop_notification(probability, risk)

            send_email(
                "user@gmail.com",
                probability,
                risk
            )

            send_sms(
                "+919876543210",
                probability,
                risk
            )

        # -------------------------
        # Save To History
        # -------------------------

        save_prediction(
            weather_name,
            speed,
            road_condition,
            traffic,
            time,
            latitude,
            longitude,
            probability,
            risk,
            recommendation
        )

        # -------------------------
        # Generate PDF Report
        # -------------------------

        pdf_path = generate_report(
            weather_name,
            speed,
            road_condition,
            traffic,
            time,
            latitude,
            longitude,
            probability,
            risk,
            recommendation
        )

        # -------------------------
        # Return Response
        # -------------------------

        return jsonify({

            "weather": weather_name,
            "temperature": temperature,
            "humidity": humidity,
            "wind_speed": wind_speed,

            "accident_prediction": int(prediction),
            "probability": round(probability, 2),
            "risk_level": risk,
            "recommendation": recommendation,

            "report_path": pdf_path

        })

    except Exception as e:

        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400


# ===========================
# Run Flask Server
# ===========================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)