import os
import sqlite3
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE = os.path.join(BASE_DIR, "..", "database", "accident_history.db")

connection = sqlite3.connect(DATABASE)

df = pd.read_sql(
    "SELECT * FROM prediction_history",
    connection
)

connection.close()

print(df)
