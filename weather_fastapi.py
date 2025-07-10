from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from utils import fetch_weather_data, get_hourly_forecast, generate_weather_alerts, get_7_day_forecast
import joblib
import pandas as pd

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

time_model = joblib.load("model/best_window_model.pkl")

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("weather.html", {"request": request})

@app.get("/weather")
def get_weather(lat: float, lon: float):
    weather = fetch_weather_data(lat, lon)
    if not weather:
        return JSONResponse({'error': 'Weather data unavailable'}, status_code=500)
    return {"weather": weather}

@app.get("/spray_window")
def spray_window(lat: float, lon: float):
    hourly_data = get_hourly_forecast(lat, lon)
    if hourly_data.empty:
        return JSONResponse({'result': 'No hourly forecast data available.', 'window': None})
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
        msg = f"Best 3-hour window to spray: {best_window} (Confidence: {best_score:.2f})"
    else:
        msg = f"No ideal 3-hour window, but highest confidence: {best_window} (Confidence: {best_score:.2f})"
    return {"result": msg, "window": best_window, "confidence": best_score}

@app.get("/weather_alerts")
def weather_alerts(lat: float, lon: float):
    forecast_df = get_7_day_forecast(lat, lon)
    if forecast_df.empty:
        return {"alerts": ["No forecast data available for alerts."]}
    alerts = generate_weather_alerts(forecast_df)
    return {"alerts": alerts}
