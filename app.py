## File: app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import numpy as np
from streamlit_folium import st_folium
import folium
from smart_agri.utils import fetch_weather_data

st.set_page_config(page_title="Smart Agriculture - Ozone Impact", layout="centered")

st.title("🌾 Smart Agriculture: Ground-Level Ozone Impact on Yield")

# Load ML model
try:
    model = joblib.load("model/yield_model.pkl")
except FileNotFoundError:
    model = None
    st.warning("Model not found. Please train the model first using model_training.py")

st.subheader("🌍 Select Your Location on Map")

m = folium.Map(location=[20.5937, 78.9629], zoom_start=4)
marker = folium.Marker([20.5937, 78.9629], tooltip="Your Location", draggable=True)
marker.add_to(m)
output = st_folium(m, height=400, width=700)

if output and output['last_clicked']:
    lat = output['last_clicked']['lat']
    lon = output['last_clicked']['lng']

    weather = fetch_weather_data(lat, lon)
    if weather:
        temp = weather['temp']
        rain = weather['rain']

        st.success("✅ Weather Data Retrieved")
        st.write(f"**Latitude:** {lat:.2f}, **Longitude:** {lon:.2f}")
        st.write(f"**Temperature:** {temp:.2f} °C")
        st.write(f"**Rainfall:** {rain:.2f} mm")

        ozone = st.slider("Input Ground-Level Ozone (ppb)", 30, 100, 60)
        soil = st.slider("Input Soil Moisture (m³/m³)", 0.1, 0.5, 0.25)

        if model:
            features = np.array([[ozone, temp, rain, soil]])
            prediction = model.predict(features)[0]
            st.success(f"📊 Predicted Crop Yield: **{prediction:.2f} tonnes/hectare**")

            st.subheader("📉 Ozone vs Yield Sensitivity")
            ozone_vals = np.linspace(30, 100, 50)
            predictions = model.predict(np.column_stack((ozone_vals, np.full(50, temp), np.full(50, rain), np.full(50, soil))))
            fig, ax = plt.subplots()
            ax.plot(ozone_vals, predictions, color='green')
            ax.set_xlabel("Ozone Level (ppb)")
            ax.set_ylabel("Predicted Yield (tonnes/ha)")
            ax.set_title("Ozone Impact on Yield")
            st.pyplot(fig)
        else:
            st.warning("Train a model first using model_training.py")
    else:
        st.error("Failed to fetch weather data. Check coordinates or network.")
else:
    st.info("Click on the map to select a location.")
