// This script fetches weather and model outputs every minute and displays popups for all models except crop recommendation.

function showPopup(message) {
    const popup = document.createElement('div');
    popup.className = 'popup-message';
    popup.innerHTML = message;
    document.body.appendChild(popup);
    setTimeout(() => { popup.remove(); }, 8000);
}

async function fetchAndDisplayAllModels(lat, lon) {
    try {
        // Weather and recommendations
        const res = await fetch(`/get_agri_data?lat=${lat}&lon=${lon}`);
        const data = await res.json();
        if (data.weather) {
            showPopup(`<b>Weather Update:</b><br>Temp: ${data.weather.temp}°C, Humidity: ${data.weather.humidity}%, Rain: ${data.weather.rain}mm, Wind: ${data.weather.wind}km/h`);
        }
        // Yield Prediction
        const ozone = 40, soil = 0.25; // Example values, can randomize or fetch
        const yieldRes = await fetch(`/predict_yield?lat=${lat}&lon=${lon}&ozone=${ozone}&soil=${soil}`);
        const yieldData = await yieldRes.json();
        if (yieldData.result) {
            showPopup(`<b>Yield Prediction:</b><br>${yieldData.result}`);
        }
        // Fertilizer Recommendation
        const ph = 6.5, stage = 'Bulking';
        const fertRes = await fetch(`/recommend_fertilizer?lat=${lat}&lon=${lon}&ozone=${ozone}&soil=${soil}&ph=${ph}&stage=${stage}`);
        const fertData = await fertRes.json();
        if (fertData.result) {
            showPopup(`<b>Fertilizer Recommendation:</b><br>${fertData.result}`);
        }
        // Stress Prediction
        const temp = 25, humidityVal = 60, color = 'Dark Green', symptom = 'None';
        const stressRes = await fetch(`/predict_stress?lat=${lat}&lon=${lon}&ozone=${ozone}&temp=${temp}&humidity=${humidityVal}&color=${color}&symptom=${symptom}`);
        const stressData = await stressRes.json();
        if (stressData.result) {
            showPopup(`<b>Stress Level:</b><br>${stressData.result}<br>${stressData.explanation || ''}`);
        }
    } catch (err) {
        showPopup('<span style="color:red">Error fetching auto-update data.</span>');
    }
}

// Get default or last used lat/lon (customize as needed)
let lat = 20.5937, lon = 78.9629;
if (document.getElementById('lat') && document.getElementById('lon')) {
    lat = document.getElementById('lat').value || lat;
    lon = document.getElementById('lon').value || lon;
}

setInterval(() => {
    fetchAndDisplayAllModels(lat, lon);
}, 60000); // Every 1 minute

// Add popup CSS
const style = document.createElement('style');
style.innerHTML = `.popup-message { position: fixed; top: 30px; right: 30px; background: #222; color: #fff; padding: 1.2em 2em; border-radius: 8px; box-shadow: 0 4px 16px rgba(0,0,0,0.2); z-index: 9999; font-size: 1.1em; margin-bottom: 10px; opacity: 0.95; }`;
document.head.appendChild(style);
