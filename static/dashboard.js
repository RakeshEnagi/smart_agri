document.addEventListener('DOMContentLoaded', function() {
    // Tab switching logic
    const tabs = document.querySelectorAll('.tab');
    const tabContents = document.querySelectorAll('.tab-content');
    tabs.forEach(tab => {
        tab.addEventListener('click', function() {
            tabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            tabContents.forEach(tc => tc.style.display = 'none');
            document.getElementById(tab.dataset.target).style.display = 'block';
        });
    });
    // Show first tab by default
    if (tabs.length > 0) {
        tabs[0].click();
    }
});

// AJAX for dashboard features (example for yield prediction)
async function getYieldPrediction() {
    const lat = document.getElementById('lat').value;
    const lon = document.getElementById('lon').value;
    const ozone = document.getElementById('ozone').value;
    const soil = document.getElementById('soil').value;
    const res = await fetch(`/predict_yield?lat=${lat}&lon=${lon}&ozone=${ozone}&soil=${soil}`);
    const data = await res.json();
    document.getElementById('yield-result').innerHTML = data.result ? `<b>${data.result}</b>` : 'Error';
}

async function getFertilizerRecommendation() {
    const lat = document.getElementById('lat').value;
    const lon = document.getElementById('lon').value;
    const ozone = document.getElementById('ozone').value;
    const soil = document.getElementById('soil').value;
    const ph = document.getElementById('ph').value;
    const stage = document.getElementById('stage').value;
    const res = await fetch(`/recommend_fertilizer?lat=${lat}&lon=${lon}&ozone=${ozone}&soil=${soil}&ph=${ph}&stage=${stage}`);
    const data = await res.json();
    document.getElementById('fertilizer-result').innerHTML = data.result ? `<b>${data.result}</b>` : 'Error';
}

async function getStressPrediction() {
    const lat = document.getElementById('lat').value;
    const lon = document.getElementById('lon').value;
    const ozone = document.getElementById('ozone').value;
    const temp = document.getElementById('temp').value;
    const humidity = document.getElementById('humidity').value;
    const color = document.getElementById('color').value;
    const symptom = document.getElementById('symptom').value;
    const res = await fetch(`/predict_stress?lat=${lat}&lon=${lon}&ozone=${ozone}&temp=${temp}&humidity=${humidity}&color=${color}&symptom=${symptom}`);
    const data = await res.json();
    document.getElementById('stress-result').innerHTML = data.result ? `<b>${data.result}</b><br>${data.explanation}` : 'Error';
}

async function getBestTimeToSpray() {
    const lat = document.getElementById('lat').value;
    const lon = document.getElementById('lon').value;
    const res = await fetch(`/best_time_to_spray?lat=${lat}&lon=${lon}`);
    const data = await res.json();
    document.getElementById('besttime-result').innerHTML = data.result ? `<b>${data.result}</b>` : 'Error';
}

async function getCropRecommendation() {
    const fields = ['N','P','K','temperature','humidity','ph','rainfall','ozone'];
    const params = fields.map(f => `${f}=` + encodeURIComponent(document.getElementById(f).value)).join('&');
    const res = await fetch(`/recommend_crop?${params}`);
    const data = await res.json();
    document.getElementById('crop-result').innerHTML = data.recommended_crop ? `<b>Recommended Crop: ${data.recommended_crop}</b>` : (data.error || 'Error');
}
