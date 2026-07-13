import os
import sqlite3
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE = os.path.join(BASE_DIR, "..", "database", "accident_history.db")


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
    recommendation=""
):

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


def get_history():

    connection = sqlite3.connect(DATABASE)
    cursor = connection.cursor()

    cursor.execute("SELECT * FROM prediction_history ORDER BY id DESC")
    data = cursor.fetchall()

    connection.close()
    return data
