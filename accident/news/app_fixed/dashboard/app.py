import os
import sqlite3
import requests
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium

# ===================================
# Configuration
# ===================================

st.set_page_config(
    page_title="AI Road Accident Prediction",
    page_icon="🚗",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.abspath(
    os.path.join(BASE_DIR, "..", "database", "accident_history.db")
)

API_URL = "http://127.0.0.1:5000/predict"

# ===================================
# Title
# ===================================

st.title("🚗 AI Road Accident Prediction System")
st.write("Predict road accident probability using Artificial Intelligence.")

# ===================================
# Sidebar
# ===================================

st.sidebar.header("Prediction Input")

speed = st.sidebar.slider(
    "Vehicle Speed (km/h)",
    0,
    150,
    60
)

road = st.sidebar.selectbox(
    "Road Condition",
    [
        "Dry",
        "Wet"
    ]
)

traffic = st.sidebar.selectbox(
    "Traffic",
    [
        "Low",
        "Medium",
        "Heavy"
    ]
)

time = st.sidebar.selectbox(
    "Time",
    [
        "Morning",
        "Afternoon",
        "Evening",
        "Night"
    ]
)

latitude = st.sidebar.number_input(
    "Latitude",
    value=13.0827,
    format="%.4f"
)

longitude = st.sidebar.number_input(
    "Longitude",
    value=80.2707,
    format="%.4f"
)

result = None

# ===================================
# Prediction
# ===================================

if st.sidebar.button("🚀 Predict Accident Risk"):

    payload = {

        "speed": speed,
        "road_condition": road,
        "traffic": traffic,
        "time": time,
        "latitude": latitude,
        "longitude": longitude

    }

    try:

        response = requests.post(
            API_URL,
            json=payload,
            timeout=20
        )

        response.raise_for_status()

        result = response.json()

    except requests.exceptions.ConnectionError:

        st.error("❌ Flask API is not running.")

    except Exception as e:

        st.error(e)

# ===================================
# Prediction Result
# ===================================

if result:

    st.divider()

    st.header("Prediction Result")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Accident Probability",
            f"{result['probability']} %"
        )

    with col2:

        st.metric(
            "Risk Level",
            result["risk_level"]
        )

    st.info(result["recommendation"])

    if result["risk_level"] == "HIGH":

        st.error("🚨 HIGH ACCIDENT RISK")

    elif result["risk_level"] == "MEDIUM":

        st.warning("⚠ MEDIUM ACCIDENT RISK")

    else:

        st.success("✅ LOW ACCIDENT RISK")

# ===================================
# Live Weather
# ===================================

    st.divider()

    st.header("🌤 Live Weather")

    c1, c2 = st.columns(2)
    c3, c4 = st.columns(2)

    with c1:

        st.metric(
            "Weather",
            result["weather"]
        )

    with c2:

        st.metric(
            "Temperature",
            f"{result['temperature']} °C"
        )

    with c3:

        st.metric(
            "Humidity",
            f"{result['humidity']} %"
        )

    with c4:

        st.metric(
            "Wind Speed",
            f"{result['wind_speed']} m/s"
        )

# ===================================
# Download PDF
# ===================================

    if os.path.exists(result["report_path"]):

        with open(result["report_path"], "rb") as pdf:

            st.download_button(

                "📄 Download PDF Report",

                pdf,

                file_name=os.path.basename(result["report_path"]),

                mime="application/pdf"

            )

# ===================================
# Prediction History
# ===================================

st.divider()

st.header("📜 Prediction History")

try:

    if os.path.exists(DB_PATH):

        conn = sqlite3.connect(DB_PATH)

        df = pd.read_sql(
            "SELECT * FROM prediction_history ORDER BY id DESC",
            conn
        )

        conn.close()

        if len(df):

            st.dataframe(
                df,
                use_container_width=True
            )

        else:

            st.info("No prediction history found.")

except Exception as e:

    st.warning(e)

# ===================================
# Accident Heatmap
# ===================================

st.divider()

st.header("🗺 Accident Heatmap")

try:

    if os.path.exists(DB_PATH):

        conn = sqlite3.connect(DB_PATH)

        df = pd.read_sql(

            """
            SELECT
                latitude,
                longitude,
                risk,
                probability
            FROM prediction_history
            """,

            conn

        )

        conn.close()

        if len(df):

            fmap = folium.Map(

                location=[13.0827, 80.2707],

                zoom_start=11

            )

            for _, row in df.iterrows():

                color = {

                    "LOW": "green",

                    "MEDIUM": "orange",

                    "HIGH": "red"

                }.get(row["risk"], "blue")

                folium.CircleMarker(

                    location=[
                        row["latitude"],
                        row["longitude"]
                    ],

                    radius=8,

                    popup=f"{row['risk']} ({row['probability']:.2f}%)",

                    color=color,

                    fill=True,

                    fill_color=color

                ).add_to(fmap)

            st_folium(

                fmap,

                width=1100,

                height=500

            )

        else:

            st.info("No accident locations found.")

except Exception as e:

    st.warning(e)