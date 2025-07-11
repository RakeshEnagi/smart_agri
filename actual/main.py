from fastapi import FastAPI, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, HTMLResponse
import pandas as pd
import joblib
import requests
from datetime import datetime
from pydantic import BaseModel
from typing import List, Dict
import json
import os

app = FastAPI(title="Smart Agriculture API")

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Config
MODEL_PATH = "model/risk_predictor_model.pkl"
DISEASE_ENCODER_PATH = "model/disease_label_encoder.pkl"
RISK_ENCODER_PATH = "model/risk_label_encoder.pkl"
FIELDS_FILE = "data/fields.json"

# Load models
model = joblib.load(MODEL_PATH)
le_disease = joblib.load(DISEASE_ENCODER_PATH)
le_risk = joblib.load(RISK_ENCODER_PATH)

class Field(BaseModel):
    name: str
    lat: float
    lon: float

def estimate_leaf_wetness(humidity: float, rainfall: float) -> float:
    if humidity > 90 and rainfall > 0:
        return 13 + (humidity - 90) * 0.1 + rainfall * 0.5
    elif humidity > 90:
        return 11 + (humidity - 90) * 0.2
    elif rainfall > 0:
        return 10 + rainfall * 0.5
    else:
        return 8

def get_met_weather_forecast(lat: float, lon: float):
    url = f"https://api.met.no/weatherapi/locationforecast/2.0/compact?lat={lat}&lon={lon}"
    headers = {
        "User-Agent": "smart-agri-dashboard/1.0 contact@example.com"
    }

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=f"MET API Error: {response.text}")

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

    return forecast_list

def predict_risk_for_all_diseases(forecast_data: List[Dict]):
    forecast_df = pd.DataFrame(forecast_data)
    all_results = []
    
    for disease_name in le_disease.classes_:
        disease_encoded = le_disease.transform([disease_name])[0]
        temp_df = forecast_df.copy()
        temp_df["Disease_enc"] = disease_encoded

        features = ["Disease_enc", "Temperature", "Humidity", "Rainfall",
                   "Cloud Cover", "Wind Speed", "Leaf Wetness"]
        X = temp_df[features]
        predictions = model.predict(X)
        
        for idx, row in temp_df.iterrows():
            result = {
                "date": row["Date"],
                "disease": disease_name,
                "risk": le_risk.inverse_transform([predictions[idx]])[0],
                "temperature": row["Temperature"],
                "humidity": row["Humidity"],
                "rainfall": row["Rainfall"],
                "cloud_cover": row["Cloud Cover"],
                "leaf_wetness": row["Leaf Wetness"]
            }
            all_results.append(result)
    
    return all_results

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/fields")
async def add_field(field: Field):
    try:
        fields = {}
        if os.path.exists(FIELDS_FILE):
            with open(FIELDS_FILE, "r") as f:
                fields = json.load(f)
        
        fields[field.name] = {"lat": field.lat, "lon": field.lon}
        
        os.makedirs(os.path.dirname(FIELDS_FILE), exist_ok=True)
        with open(FIELDS_FILE, "w") as f:
            json.dump(fields, f, indent=2)
            
        return {"message": "Field added successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/fields")
async def get_fields():
    try:
        if os.path.exists(FIELDS_FILE):
            with open(FIELDS_FILE, "r") as f:
                return json.load(f)
        return {}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/forecast/{field_name}")
async def get_forecast(field_name: str):
    try:
        with open(FIELDS_FILE, "r") as f:
            fields = json.load(f)
        
        if field_name not in fields:
            raise HTTPException(status_code=404, detail="Field not found")
            
        field = fields[field_name]
        forecast_data = get_met_weather_forecast(field["lat"], field["lon"])
        risk_predictions = predict_risk_for_all_diseases(forecast_data)
        
        return risk_predictions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
