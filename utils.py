import requests
import numpy as np

def fetch_weather_data(lat, lon):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m,precipitation&forecast_days=1&timezone=auto"
    response = requests.get(url)
    if response.status_code != 200:
        return None
    data = response.json()["hourly"]
    return {
        "temp": np.mean(data["temperature_2m"]),
        "rain": np.mean(data["precipitation"])
    } 