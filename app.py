import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import numpy as np
from streamlit_folium import st_folium
import folium
from utils import fetch_weather_data, get_hourly_forecast, recommend_fertilizer, predict_stress_level, get_7_day_forecast, generate_weather_alerts
import os

# Load Crop Model
crop_model_path = os.path.join('model', 'crop_model.pkl')
if os.path.exists(crop_model_path):
    crop_model = joblib.load(crop_model_path)
    crop_model_loaded = True
else:
    crop_model_loaded = False

# Load Other Models
yield_model = joblib.load("model/yield_model.pkl")
time_model = joblib.load("model/best_window_model.pkl")
fert_model = joblib.load("model/fertilizer_model.pkl")
stress_model = joblib.load("model/stress_model.pkl")

st.set_page_config(page_title="Smart Potato Farming", layout="wide")

st.title("🥔 Smart Potato Farming Dashboard")

# Sidebar Navigation
page = st.sidebar.radio("Choose a Feature", [
    "📈 Yield Prediction",
    "🕒 Best Fertilizer Window",
    "🧪 Fertilizer Recommendation",
    "⚠️ Crop Stress Level Prediction",
    "🌱 Crop Recommendation"
])

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

        st.subheader("🌦️ Weather Alerts for Next 7 Days")
        forecast_df = get_7_day_forecast(lat, lon)
        if not forecast_df.empty:
            alerts = []
            total_rain = forecast_df['rain'].sum()
            if total_rain < 10:
                alerts.append(f"🌧️ Low rainfall expected (Total: {total_rain:.1f} mm)")
            if total_rain > 70:
                alerts.append(f"🌊 High rainfall alert (Total: {total_rain:.1f} mm)")
            if forecast_df['rain'].std() > 5:
                alerts.append("🌦️ Uneven rainfall pattern over next 7 days")
            if forecast_df['temp'].max() > 38:
                alerts.append(f"🌡️ High temperatures up to {forecast_df['temp'].max():.1f}°C expected")
            if forecast_df['wind'].max() > 7:
                alerts.append(f"🌬️ High wind speeds up to {forecast_df['wind'].max():.1f} m/s")
            fog_days = forecast_df[(forecast_df['humidity'] > 85) & (forecast_df['temp'] < 20)]
            if len(fog_days) >= 2:
                alerts.append("🌫️ Foggy conditions likely on multiple days")

            if alerts:
                for alert in alerts:
                    st.warning(alert)
            else:
                st.success("✅ No major weather threats detected in the next 7 days.")
            st.dataframe(forecast_df)
        else:
            st.warning("Could not fetch 7-day forecast data.")

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
            prediction = recommend_fertilizer(input_df, fert_model)
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

        elif page == "🌱 Crop Recommendation":
            st.header("🌱 Crop Recommendation")
            st.write("Enter soil and weather parameters to get the best crop recommendation.")
            N = st.number_input("Nitrogen (N)", min_value=0, max_value=200, value=50)
            P = st.number_input("Phosphorus (P)", min_value=0, max_value=200, value=50)
            K = st.number_input("Potassium (K)", min_value=0, max_value=200, value=50)
            temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=60.0, value=25.0)
            humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0)
            ph = st.number_input("Soil pH", min_value=3.0, max_value=10.0, value=6.5)
            rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=100.0)
            ozone = st.number_input("Ozone (ppb)", min_value=10, max_value=100, value=40)
            if crop_model_loaded:
                if st.button("Recommend Crop"):
                    features = [[N, P, K, temperature, humidity, ph, rainfall, ozone]]
                    try:
                        pred = crop_model.predict(features)[0]
                        st.success(f"Recommended Crop: **{pred}**")
                    except Exception as e:
                        st.warning("No preferred crop available for the given conditions.")
                        st.error(f"Prediction error: {e}")
            else:
                st.warning("Crop recommendation model not found. Please place crop_model.pkl in the model folder.")
    st.markdown("<span style='color:#388e3c'><b>ℹ️ Weather and model outputs will now auto-update and show as popups every minute (web dashboard only).</b></span>", unsafe_allow_html=True)
else:
    st.info("Click on the map to select a location.")
