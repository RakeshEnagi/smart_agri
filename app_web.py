from flask import Flask, render_template, request, jsonify
from utils import fetch_weather_data, get_hourly_forecast, recommend_fertilizer, predict_stress_level

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_agri_data')
def get_agri_data():
    lat = request.args.get('lat', type=float)
    lon = request.args.get('lon', type=float)
    if lat is None or lon is None:
        return jsonify({'error': 'Missing coordinates'}), 400
    weather = fetch_weather_data(lat, lon)
    if not weather:
        return jsonify({'error': 'Weather data unavailable'}), 500
    # Placeholder for recommendations (can be extended)

    recommendations = "Use the dashboard features for yield, fertilizer, and stress prediction."
    return jsonify({'weather': weather, 'recommendations': recommendations})

# --- Dashboard API endpoints ---
import joblib
import pandas as pd

yield_model = joblib.load("model/yield_model.pkl")
fert_model = joblib.load("model/fert_model.pkl")
stress_model = joblib.load("model/stress_model.pkl")
time_model = joblib.load("model/best_window_model.pkl")
crop_model = joblib.load("model/crop_model.pkl")
@app.route('/recommend_crop')
def recommend_crop():
    # Get input features from request.args
    try:
        features = [float(request.args.get(f)) for f in ['N','P','K','temperature','humidity','ph','rainfall','ozone']]
    except Exception:
        return jsonify({'error': 'Invalid or missing input'}), 400
    pred = crop_model.predict([features])[0]
    return jsonify({'recommended_crop': pred})
@app.route('/best_time_to_spray')
def best_time_to_spray():
    lat = request.args.get('lat', type=float)
    lon = request.args.get('lon', type=float)
    if lat is None or lon is None:
        return jsonify({'result': None, 'window': None}), 400
    hourly_data = get_hourly_forecast(lat, lon)
    if hourly_data.empty:
        return jsonify({'result': 'No hourly forecast data available.', 'window': None})
    # Predict probability for each hour
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
    return jsonify({'result': msg, 'window': best_window, 'confidence': best_score})

@app.route('/predict_yield')
def predict_yield():
    lat = request.args.get('lat', type=float)
    lon = request.args.get('lon', type=float)
    ozone = request.args.get('ozone', type=float)
    soil = request.args.get('soil', type=float)
    weather = fetch_weather_data(lat, lon)
    if not weather:
        return jsonify({'result': None}), 400
    temp = weather['temp']
    rain = weather['rain']
    features = pd.DataFrame([[ozone, temp, rain, soil]], columns=["ozone", "temp", "rain", "soil"])
    prediction = yield_model.predict(features)[0]
    return jsonify({'result': f"Predicted Potato Yield: {prediction:.2f} tonnes/hectare"})

@app.route('/recommend_fertilizer')
def recommend_fertilizer_api():
    lat = request.args.get('lat', type=float)
    lon = request.args.get('lon', type=float)
    ozone = request.args.get('ozone', type=float)
    soil = request.args.get('soil', type=float)
    ph = request.args.get('ph', type=float)
    stage = request.args.get('stage', type=str)
    weather = fetch_weather_data(lat, lon)
    if not weather:
        return jsonify({'result': None}), 400
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
    return jsonify({'result': f"Recommended Fertilizer: {result}"})

@app.route('/predict_stress')
def predict_stress():
    lat = request.args.get('lat', type=float)
    lon = request.args.get('lon', type=float)
    ozone = request.args.get('ozone', type=float)
    temp = request.args.get('temp', type=float)
    humidity = request.args.get('humidity', type=float)
    color = request.args.get('color', type=str)
    symptom = request.args.get('symptom', type=str)
    input_df = pd.DataFrame([[ozone, temp, humidity, color, symptom]],
                            columns=["ozone", "temp", "humidity", "color", "symptom"])
    level, explanation = predict_stress_level(stress_model, input_df)
    return jsonify({'result': f"Stress Level: {level}", 'explanation': explanation})

if __name__ == '__main__':
    app.run(debug=True)
