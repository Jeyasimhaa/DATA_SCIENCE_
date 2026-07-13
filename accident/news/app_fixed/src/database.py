import os
import sqlite3

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


if __name__ == "__main__":
    create_database()
    print("Database Created Successfully!")
