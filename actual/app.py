import streamlit as st
import pandas as pd
import requests
import joblib
from datetime import datetime
from streamlit_folium import st_folium
import folium
import json
import os

# ---------------------- CONFIG ----------------------
FIELDS_FILE = "fields.json"
MODEL_PATH = "risk_predictor_model.pkl"
DISEASE_ENCODER_PATH = "disease_label_encoder.pkl"
RISK_ENCODER_PATH = "risk_label_encoder.pkl"
# -----------------------------------------------------

# 🌿 Leaf Wetness Estimate
def estimate_leaf_wetness(humidity, rainfall):
    if humidity > 90 and rainfall > 0:
        return 13 + (humidity - 90) * 0.1 + rainfall * 0.5
    elif humidity > 90:
        return 11 + (humidity - 90) * 0.2
    elif rainfall > 0:
        return 10 + rainfall * 0.5
    else:
        return 8

# 🌦️ MET API Forecast (No key needed)
def get_met_weather_forecast(lat, lon):
    url = f"https://api.met.no/weatherapi/locationforecast/2.0/compact?lat={lat}&lon={lon}"
    headers = {
        "User-Agent": "smart-agri-dashboard/1.0 contact@example.com"
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
                'Wind Speed': details.get('wind_speed', 0),
            }

            forecast['Leaf Wetness'] = round(estimate_leaf_wetness(forecast['Humidity'], forecast['Rainfall']), 2)
            forecast_list.append(forecast)
            seen_dates.add(dt.date())

        if len(forecast_list) == 7:
            break

    return pd.DataFrame(forecast_list)

# 🧠 Predict Disease Risks
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

# 📁 Load/Save Field Locations
def load_fields():
    if os.path.exists(FIELDS_FILE):
        try:
            with open(FIELDS_FILE, "r") as f:
                content = f.read().strip()
                if content == "":
                    return {}
                return json.loads(content)
        except json.JSONDecodeError:
            st.warning("⚠️ 'fields.json' is corrupted. Resetting field data.")
            return {}
    return {}

def save_fields(fields):
    with open(FIELDS_FILE, "w") as f:
        json.dump(fields, f, indent=2)

# 🌾 Streamlit App
st.set_page_config(page_title="Smart Agri | Disease Risk", layout="wide")
st.title("🌾 Smart Agriculture - Potato Disease Risk Forecast")

# 🔁 Load existing fields
fields = load_fields()

# ➕ Add New Farming Field
st.subheader("➕ Add New Farming Field")

col1, col2 = st.columns([2, 3])
with col1:
    field_name = st.text_input("Enter Field Name")

with col2:
    m = folium.Map(location=[15.3, 75.7], zoom_start=5)
    map_data = st_folium(m, height=300, returned_objects=["last_clicked"])

if map_data and map_data["last_clicked"]:
    lat = map_data["last_clicked"]["lat"]
    lon = map_data["last_clicked"]["lng"]
    st.success(f"📍 Selected Location: {lat:.4f}, {lon:.4f}")

    if st.button("💾 Save Field Location"):
        if field_name.strip() == "":
            st.warning("Field name cannot be empty.")
        else:
            fields[field_name] = {"lat": lat, "lon": lon}
            save_fields(fields)
            st.success(f"✅ Field '{field_name}' saved successfully!")

# 📂 View Saved Fields
st.subheader("📂 View Saved Fields")

if fields:
    selected_field = st.selectbox("Choose a field:", list(fields.keys()))
    lat, lon = fields[selected_field]["lat"], fields[selected_field]["lon"]
    st.map(pd.DataFrame([[lat, lon]], columns=["lat", "lon"]))

    if st.button("📊 Show Risk Forecast"):
        try:
            with st.spinner("Fetching weather and predicting..."):
                weather_df = get_met_weather_forecast(lat, lon)
                risk_df = predict_risk_for_all_diseases(weather_df)

            for disease in risk_df["Disease"].unique():
                st.subheader(f"🔬 Disease: {disease}")
                display_df = risk_df[risk_df["Disease"] == disease][
                    ["Date", "Temperature", "Humidity", "Rainfall", "Cloud Cover", "Leaf Wetness", "Predicted Risk"]
                ]
                st.dataframe(display_df, use_container_width=True)
        except Exception as e:
            st.error(f"❌ Error: {e}")
else:
    st.info("No fields saved yet. Add a field above.")
