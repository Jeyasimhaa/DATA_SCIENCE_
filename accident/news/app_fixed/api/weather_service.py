import os
import requests
from dotenv import load_dotenv

# Load variables from the .env file (WEATHER_API_KEY, etc.)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, "..", ".env"))

API_KEY = os.getenv("WEATHER_API_KEY")

# The trained model only knows the categories that exist in
# dataset/accident_data.csv: Sunny, Cloudy, Rain, Fog.
# OpenWeatherMap returns its own vocabulary (Clear, Clouds, Drizzle,
# Thunderstorm, Snow, Mist, Haze, Dust, Smoke, Squall, Tornado...),
# so live results are mapped onto the closest trained category here.
# Without this mapping, encoders["Weather"].transform(...) would raise
# "y contains previously unseen labels" for most real weather conditions.
OWM_TO_DATASET_WEATHER = {
    "Clear": "Sunny",
    "Clouds": "Cloudy",
    "Rain": "Rain",
    "Drizzle": "Rain",
    "Thunderstorm": "Rain",
    "Snow": "Rain",
    "Mist": "Fog",
    "Fog": "Fog",
    "Haze": "Fog",
    "Smoke": "Fog",
    "Dust": "Fog",
    "Sand": "Fog",
    "Ash": "Fog",
    "Squall": "Rain",
    "Tornado": "Rain",
}


def get_weather(latitude, longitude):

    if not API_KEY:
        raise ValueError(
            "WEATHER_API_KEY is not set. Add it to the .env file in the project root."
        )

    url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?lat={latitude}&lon={longitude}"
        f"&appid={API_KEY}&units=metric"
    )

    response = requests.get(url, timeout=10)
    data = response.json()

    if response.status_code != 200:
        message = data.get("message", "Unknown error from weather service")
        raise RuntimeError(f"Weather API error: {message}")

    raw_weather = data["weather"][0]["main"]
    weather = OWM_TO_DATASET_WEATHER.get(raw_weather, "Cloudy")
    temperature = data["main"]["temp"]
    humidity = data["main"]["humidity"]
    wind_speed = data["wind"]["speed"]

    return {
        "weather": weather,
        "temperature": temperature,
        "humidity": humidity,
        "wind_speed": wind_speed
    }
