import os
import sqlite3
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "accident_history.db")


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

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

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

    conn.commit()
    conn.close()


def get_history():

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM prediction_history ORDER BY id DESC")
    data = cursor.fetchall()

    conn.close()
    return data
