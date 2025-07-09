import requests
import pandas as pd
import numpy as np
from datetime import datetime
import joblib

def fetch_weather_data(lat, lon):
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m,relative_humidity_2m,precipitation,windspeed_10m&current_weather=true"
        res = requests.get(url)
        data = res.json()

        if 'current_weather' in data:
            temp = data['current_weather']['temperature']
            wind = data['current_weather']['windspeed']
            hourly = data['hourly']
            humidity = hourly['relative_humidity_2m'][0]
            rain = hourly['precipitation'][0]
            return {
                "temp": temp,
                "humidity": humidity,
                "rain": rain,
                "wind": wind
            }
        else:
            print("No current weather data found.")
            return None
    except Exception as e:
        print("Error fetching weather data:", e)
        return None

def get_hourly_forecast(lat, lon):
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m,relative_humidity_2m,precipitation,windspeed_10m&timezone=auto"
        res = requests.get(url)
        data = res.json()

        hourly = data.get("hourly", {})
        df = pd.DataFrame({
            "hour": pd.to_datetime(hourly["time"]).hour,
            "temp": hourly["temperature_2m"],
            "humidity": hourly["relative_humidity_2m"],
            "rain": hourly["precipitation"],
            "wind": hourly["windspeed_10m"]
        })

        # Assume constant ozone value (to be updated from app.py input)
        df["ozone"] = 60  # Placeholder
        return df
    except Exception as e:
        print("Error fetching forecast:", e)
        return pd.DataFrame()

def get_7_day_forecast(lat, lon):
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max,relative_humidity_2m_max&timezone=auto"
        res = requests.get(url)
        data = res.json()

        daily = data.get("daily", {})
        if not daily:
            return pd.DataFrame()

        df = pd.DataFrame({
            "day": pd.to_datetime(daily["time"]).strftime("%a"),
            "temp": daily["temperature_2m_max"],
            "rain": daily["precipitation_sum"],
            "wind": daily["windspeed_10m_max"],
            "humidity": daily["relative_humidity_2m_max"]
        })

        return df
    except Exception as e:
        print("Error fetching 7-day forecast:", e)
        return pd.DataFrame()

def generate_weather_alerts(forecast_df):
    alerts = []

    total_rain = forecast_df['rain'].sum()
    if total_rain < 10:
        alerts.append(f"🌧️ Low rainfall expected (Total: {total_rain:.1f} mm)")

    if forecast_df['rain'].std() > 5:
        alerts.append("🌦️ Uneven rainfall pattern over next 7 days")

    if forecast_df['temp'].max() > 38:
        alerts.append(f"🌡️ High temperatures up to {forecast_df['temp'].max():.1f}°C expected")

    if forecast_df['wind'].max() > 7:
        alerts.append(f"🌬️ High wind speeds up to {forecast_df['wind'].max():.1f} m/s")

    fog_days = forecast_df[(forecast_df['humidity'] > 85) & (forecast_df['temp'] < 20)]
    if len(fog_days) >= 2:
        alerts.append("🌫️ Foggy conditions likely on multiple days")

    return alerts

def recommend_fertilizer(input_df, model):
    input_df = pd.get_dummies(input_df)
    for col in model.feature_names_in_:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[model.feature_names_in_]
    return model.predict(input_df)[0]

def predict_stress_level(model, input_df):
    input_df = pd.get_dummies(input_df)
    for col in model.feature_names_in_:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[model.feature_names_in_]

    prediction = model.predict(input_df)[0]
    explanation = {
        "Low": "Healthy plant: Dark green leaves, no visible symptoms.",
        "Medium": "Mild stress detected: Possible leaf curling or slight discoloration.",
        "High": "High stress detected: Brown spots, yellowing, stunted growth due to ozone or nutrient imbalance."
    }
    return prediction, explanation.get(prediction, "Unknown stress level.")
