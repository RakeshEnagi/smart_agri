import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import numpy as np
from streamlit_folium import st_folium
import folium
from utils import fetch_weather_data, get_hourly_forecast, recommend_fertilizer, predict_stress_level

st.set_page_config(page_title="Smart Potato Farming", layout="wide")

st.title("🥔 Smart Potato Farming Dashboard")

# Sidebar Navigation
page = st.sidebar.radio("Choose a Feature", [
    "📈 Yield Prediction",
    "🕒 Best Fertilizer Window",
    "🧪 Fertilizer Recommendation",
    "⚠️ Crop Stress Level Prediction"
])

# Load models
yield_model = joblib.load("model/yield_model.pkl")
time_model = joblib.load("model/best_window_model.pkl")
fert_model = joblib.load("model/fertilizer_model.pkl")
stress_model = joblib.load("model/stress_model.pkl")

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
        humidity = weather['humidity']
        wind = weather['wind']

        st.success("✅ Weather Data Retrieved")
        st.write(f"**Latitude:** {lat:.2f}, **Longitude:** {lon:.2f}")
        st.write(f"**Temperature:** {temp:.2f} °C")
        st.write(f"**Rainfall:** {rain:.2f} mm")
        st.write(f"**Humidity:** {humidity:.2f}%")
        st.write(f"**Wind Speed:** {wind:.2f} m/s")

        ozone = st.slider("Input Ground-Level Ozone (ppb)", 30, 100, 60)
        soil = st.slider("Input Soil Moisture (m³/m³)", 0.1, 0.5, 0.25)

        if page == "📈 Yield Prediction":
            st.header("📊 Potato Yield Prediction")
            features = pd.DataFrame([[ozone, temp, rain, soil]], columns=["ozone", "temp", "rain", "soil"])
            prediction = yield_model.predict(features)[0]
            st.success(f"📊 Predicted Potato Yield: **{prediction:.2f} tonnes/hectare**")

            st.subheader("📉 Ozone vs Yield Sensitivity for Potato")
            ozone_vals = np.linspace(30, 100, 50)
            pred_df = pd.DataFrame({
                "ozone": ozone_vals,
                "temp": temp,
                "rain": rain,
                "soil": soil
            })
            predictions = yield_model.predict(pred_df)
            fig, ax = plt.subplots()
            ax.plot(ozone_vals, predictions, color='green')
            ax.set_xlabel("Ozone Level (ppb)")
            ax.set_ylabel("Predicted Yield (tonnes/ha)")
            ax.set_title("Ozone Impact on Potato Yield")
            st.pyplot(fig)

        elif page == "🕒 Best Fertilizer Window":
            st.header("🕒 Fertilizer Spray Timing Prediction")
            hourly_data = get_hourly_forecast(lat, lon)
            if not hourly_data.empty:
                st.write("### Hourly Forecast Preview:")
                st.dataframe(hourly_data)
                hourly_data['probability'] = time_model.predict_proba(
                    hourly_data[["hour", "temp", "humidity", "wind", "ozone", "rain"]]
                )[:, 1]

                best_window = None
                best_score = -1
                for i in range(len(hourly_data) - 2):
                    window = hourly_data.iloc[i:i+3]
                    avg_prob = window['probability'].mean()
                    if avg_prob > best_score:
                        best_score = avg_prob
                        best_window = f"{int(window.iloc[0]['hour'])}:00 to {int(window.iloc[2]['hour']) + 1}:00"

                if best_score >= 0.5:
                    st.success(f"✅ Best 3-hour window to spray: **{best_window}** (Confidence: {best_score:.2f})")
                else:
                    st.warning(f"⚠️ No ideal 3-hour window, but highest confidence: **{best_window}** (Confidence: {best_score:.2f})")
            else:
                st.warning("⚠️ No hourly forecast data available for prediction.")

        elif page == "🧪 Fertilizer Recommendation":
            st.header("🧪 AI-Based Fertilizer Recommendation")
            ph = st.slider("Soil pH", 4.5, 7.5, 5.8)
            stage = st.selectbox("Current Growth Stage", ["Pre-Planting", "Early Growth", "Tuberization", "Bulking"])
            input_df = pd.DataFrame([{
                "ozone": ozone,
                "temp": temp,
                "rain": rain,
                "soil": soil,
                "ph": ph,
                "stage": stage
            }])
            input_df = pd.get_dummies(input_df)
            for col in fert_model.feature_names_in_:
                if col not in input_df.columns:
                    input_df[col] = 0
            input_df = input_df[fert_model.feature_names_in_]
            prediction = fert_model.predict(input_df)[0]
            st.success(f"🌿 Recommended Fertilizer: **{prediction}**")

        elif page == "⚠️ Crop Stress Level Prediction":
            st.header("⚠️ Crop Stress Level Prediction")
            color = st.selectbox("Leaf Color Observation", ["Dark Green", "Yellowing", "Purple Tint", "Brown Spots"])
            symptom = st.selectbox("Symptom on Leaf/Plant", ["None", "Wilting", "Curling", "Stunted Growth"])
            input_df = pd.DataFrame([[ozone, temp, humidity, color, symptom]],
                                    columns=["ozone", "temp", "humidity", "color", "symptom"])
            level, explanation = predict_stress_level(stress_model, input_df)
            st.info(f"Stress Level: **{level}**")
            st.write(explanation)

    else:
        st.error("Failed to fetch weather data. Check coordinates or network.")
else:
    st.info("Click on the map to select a location.")
