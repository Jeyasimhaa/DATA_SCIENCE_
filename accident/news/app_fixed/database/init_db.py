import sqlite3
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "accident_history.db")

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

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

conn.commit()
conn.close()

print("Database created successfully!")
print(DB_PATH)
