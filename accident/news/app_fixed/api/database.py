import os
import sqlite3
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE = os.path.join(BASE_DIR, "..", "database", "accident_history.db")


def create_database():

    connection = sqlite3.connect(DATABASE)
    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            weather TEXT,
            speed REAL,
            road_condition TEXT,
            traffic TEXT,
            time TEXT,
            latitude REAL,
            longitude REAL,
            probability REAL,
            risk TEXT,
            recommendation TEXT
        )
    """)

    connection.commit()
    connection.close()


def save_prediction(
    weather,
    speed,
    road_condition,
    traffic,
    time,
    latitude,
    longitude,
    probability,
    risk,
    recommendation
):
    """Insert a prediction result into the history table."""

    connection = sqlite3.connect(DATABASE)
    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO prediction_history (
            date, weather, speed, road_condition, traffic, time,
            latitude, longitude, probability, risk, recommendation
        )
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        weather,
        speed,
        road_condition,
        traffic,
        time,
        latitude,
        longitude,
        probability,
        risk,
        recommendation
    ))

    connection.commit()
    connection.close()


if __name__ == "__main__":
    create_database()
    print("Database Created Successfully!")
