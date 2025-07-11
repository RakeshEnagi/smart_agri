import requests
import pandas as pd
import joblib
from datetime import datetime

# ---------------------- CONFIG ----------------------
LAT = 15.3
LON = 75.7

MODEL_PATH = "risk_predictor_model.pkl"
DISEASE_ENCODER_PATH = "disease_label_encoder.pkl"
RISK_ENCODER_PATH = "risk_label_encoder.pkl"
# -----------------------------------------------------

# 🌿 Estimate Leaf Wetness (hrs)
def estimate_leaf_wetness(humidity, rainfall):
    if humidity > 90 and rainfall > 0:
        return 13 + (humidity - 90) * 0.1 + rainfall * 0.5
    elif humidity > 90:
        return 11 + (humidity - 90) * 0.2
    elif rainfall > 0:
        return 10 + rainfall * 0.5
    else:
        return 8

# 🌦️ Fetch 7-day forecast from MET Norway
def get_met_weather_forecast(lat, lon):
    url = f"https://api.met.no/weatherapi/locationforecast/2.0/compact?lat={lat}&lon={lon}"
    headers = {
        "User-Agent": "smart-agri-dashboard/1.0 yuvaraj@example.com"
    }

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"MET API Error: {response.status_code} - {response.text}")

    data = response.json()
    timeseries = data['properties']['timeseries']

    forecast_list = []
    seen_dates = set()

    for entry in timeseries:
        timestamp = entry['time']
        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))

        if dt.hour == 12 and dt.date() not in seen_dates:
            details = entry['data']['instant']['details']
            rain = entry['data'].get('next_6_hours', {}).get('details', {}).get('precipitation_amount', 0.0)

            forecast = {
                'Date': dt.strftime('%Y-%m-%d'),
                'Temperature': details.get('air_temperature', 0),
                'Humidity': details.get('relative_humidity', 0),
                'Rainfall': rain,
                'Cloud Cover': details.get('cloud_area_fraction', 0),
                'Wind Speed': details.get('wind_speed', 0)
            }

            forecast['Leaf Wetness'] = round(estimate_leaf_wetness(forecast['Humidity'], forecast['Rainfall']), 2)

            forecast_list.append(forecast)
            seen_dates.add(dt.date())

        if len(forecast_list) == 7:
            break

    return pd.DataFrame(forecast_list)

# 🧠 Predict risk level using trained model
def predict_risk_for_all_diseases(forecast_df):
    model = joblib.load(MODEL_PATH)
    le_disease = joblib.load(DISEASE_ENCODER_PATH)
    le_risk = joblib.load(RISK_ENCODER_PATH)

    all_results = []

    for disease_name in le_disease.classes_:
        disease_encoded = le_disease.transform([disease_name])[0]
        temp_df = forecast_df.copy()
        temp_df["Disease_enc"] = disease_encoded

        features = ["Disease_enc", "Temperature", "Humidity", "Rainfall",
                    "Cloud Cover", "Wind Speed", "Leaf Wetness"]
        X = temp_df[features]

        predictions = model.predict(X)
        temp_df["Predicted Risk"] = le_risk.inverse_transform(predictions)
        temp_df["Disease"] = disease_name

        all_results.append(temp_df)

    return pd.concat(all_results, ignore_index=True)

# 🚀 Main Program
if __name__ == "__main__":
    print("🌾 Smart Agri: Potato Disease Risk Predictor (7-Day Forecast)\n")

    try:
        print(f"📍 Location: Lat={LAT}, Lon={LON}")
        print("📡 Fetching 7-day weather forecast from MET Norway API...")
        weather_df = get_met_weather_forecast(LAT, LON)

        print("🧠 Predicting disease risks for all known diseases...")
        risk_df = predict_risk_for_all_diseases(weather_df)

        # Display output
        for disease in risk_df['Disease'].unique():
            print(f"\n🔬 Disease: {disease}")
            display_df = risk_df[risk_df['Disease'] == disease][
                ["Date", "Temperature", "Humidity", "Rainfall", "Cloud Cover", "Leaf Wetness", "Predicted Risk"]
            ]
            print(display_df.to_string(index=False))

    except Exception as e:
        print("❌ Error:", e)
