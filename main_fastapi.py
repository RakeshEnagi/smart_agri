from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import joblib
import pandas as pd
from utils import fetch_weather_data, get_hourly_forecast, recommend_fertilizer, predict_stress_level

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

yield_model = joblib.load("model/yield_model.pkl")
time_model = joblib.load("model/best_window_model.pkl")
fert_model = joblib.load("model/fert_model.pkl")
stress_model = joblib.load("model/stress_model.pkl")
crop_model = joblib.load("model/crop_model.pkl")

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/get_agri_data")
def get_agri_data(lat: float, lon: float):
    weather = fetch_weather_data(lat, lon)
    if not weather:
        return JSONResponse({'error': 'Weather data unavailable'}, status_code=500)
    recommendations = "Use the dashboard features for yield, fertilizer, and stress prediction."
    return {"weather": weather, "recommendations": recommendations}

@app.get("/predict_yield")
def predict_yield(lat: float, lon: float, ozone: float, soil: float):
    weather = fetch_weather_data(lat, lon)
    if not weather:
        return JSONResponse({'result': None}, status_code=400)
    temp = weather['temp']
    rain = weather['rain']
    features = pd.DataFrame([[ozone, temp, rain, soil]], columns=["ozone", "temp", "rain", "soil"])
    prediction = yield_model.predict(features)[0]
    return {"result": f"Predicted Potato Yield: {prediction:.2f} tonnes/hectare"}

@app.get("/recommend_fertilizer")
def recommend_fertilizer_api(lat: float, lon: float, ozone: float, soil: float, ph: float, stage: str):
    weather = fetch_weather_data(lat, lon)
    if not weather:
        return JSONResponse({'result': None}, status_code=400)
    temp = weather['temp']
    rain = weather['rain']
    input_df = pd.DataFrame([{
        "ozone": ozone,
        "temp": temp,
        "rain": rain,
        "soil": soil,
        "ph": ph,
        "stage": stage
    }])
    result = recommend_fertilizer(input_df, fert_model)
    return {"result": f"Recommended Fertilizer: {result}"}

@app.get("/predict_stress")
def predict_stress(lat: float, lon: float, ozone: float, temp: float, humidity: float, color: str, symptom: str):
    input_df = pd.DataFrame([[ozone, temp, humidity, color, symptom]],
                            columns=["ozone", "temp", "humidity", "color", "symptom"])
    level, explanation = predict_stress_level(stress_model, input_df)
    return {"result": f"Stress Level: {level}", "explanation": explanation}

@app.get("/recommend_crop")
def recommend_crop(N: float, P: float, K: float, temperature: float, humidity: float, ph: float, rainfall: float, ozone: float):
    features = [[N, P, K, temperature, humidity, ph, rainfall, ozone]]
    try:
        pred = crop_model.predict(features)[0]
        known_crops = set(str(c) for c in crop_model.classes_)
        if str(pred).strip().lower() in (c.strip().lower() for c in known_crops):
            return {"recommended_crop": pred}
        else:
            return {"recommended_crop": None, "message": "No preferred crop available for the given conditions."}
    except Exception as e:
        return {"recommended_crop": None, "message": f"Prediction error: {e}"}
