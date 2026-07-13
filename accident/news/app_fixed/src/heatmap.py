import os
import sqlite3
import folium

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(BASE_DIR, "..")
DATABASE = os.path.join(PROJECT_ROOT, "database", "accident_history.db")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

connection = sqlite3.connect(DATABASE)
cursor = connection.cursor()

cursor.execute("""
SELECT latitude,
       longitude,
       risk,
       probability
FROM prediction_history
""")

rows = cursor.fetchall()

connection.close()

# Chennai as default center
m = folium.Map(
    location=[13.0827, 80.2707],
    zoom_start=11
)

for latitude, longitude, risk, probability in rows:

    if latitude is None or longitude is None:
        continue

    if risk == "HIGH":
        color = "red"

    elif risk == "MEDIUM":
        color = "orange"

    else:
        color = "green"

    folium.CircleMarker(
        location=[latitude, longitude],
        radius=8,
        popup=f"{risk} ({probability:.2f}%)",
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.7
    ).add_to(m)

m.save(os.path.join(OUTPUT_DIR, "heatmap.html"))

print("Heatmap Created Successfully!")
