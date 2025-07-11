let map;
let marker;
let selectedLocation = null;

// Initialize map
window.addEventListener('DOMContentLoaded', (event) => {
    initializeMap();
    loadFields();
});

function initializeMap() {
    map = L.map('map').setView([15.3, 75.7], 5);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors'
    }).addTo(map);

    map.on('click', function(e) {
        if (marker) {
            map.removeLayer(marker);
        }
        marker = L.marker(e.latlng).addTo(map);
        selectedLocation = e.latlng;
    });
}

async function loadFields() {
    try {
        const response = await fetch('/api/fields');
        const fields = await response.json();
        
        const select = document.getElementById('fieldSelect');
        select.innerHTML = '<option value="">Select a field...</option>';
        
        for (const [name, coords] of Object.entries(fields)) {
            const option = document.createElement('option');
            option.value = name;
            option.textContent = name;
            select.appendChild(option);
        }
    } catch (error) {
        console.error('Error loading fields:', error);
        alert('Failed to load fields');
    }
}

async function saveField() {
    const fieldName = document.getElementById('fieldName').value.trim();
    
    if (!fieldName) {
        alert('Please enter a field name');
        return;
    }
    
    if (!selectedLocation) {
        alert('Please select a location on the map');
        return;
    }
    
    try {
        const response = await fetch('/api/fields', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                name: fieldName,
                lat: selectedLocation.lat,
                lon: selectedLocation.lng
            }),
        });
        
        if (!response.ok) throw new Error('Failed to save field');
        
        alert('Field saved successfully!');
        document.getElementById('fieldName').value = '';
        loadFields();
    } catch (error) {
        console.error('Error saving field:', error);
        alert('Failed to save field');
    }
}

async function showFieldOnMap() {
    const fieldName = document.getElementById('fieldSelect').value;
    if (!fieldName) return;
    
    try {
        const response = await fetch('/api/fields');
        const fields = await response.json();
        
        const field = fields[fieldName];
        if (!field) return;
        
        if (marker) {
            map.removeLayer(marker);
        }
        
        const latlng = L.latLng(field.lat, field.lon);
        marker = L.marker(latlng).addTo(map);
        map.setView(latlng, 13);
    } catch (error) {
        console.error('Error showing field:', error);
        alert('Failed to show field on map');
    }
}

async function getForecast() {
    const fieldName = document.getElementById('fieldSelect').value;
    if (!fieldName) {
        alert('Please select a field');
        return;
    }
    
    try {
        const response = await fetch(`/api/forecast/${encodeURIComponent(fieldName)}`);
        const forecast = await response.json();
        
        displayForecast(forecast);
    } catch (error) {
        console.error('Error getting forecast:', error);
        alert('Failed to get forecast');
    }
}

function displayForecast(forecast) {
    const container = document.getElementById('forecast-results');
    container.innerHTML = '';
    
    // Group by disease
    const byDisease = {};
    forecast.forEach(entry => {
        if (!byDisease[entry.disease]) {
            byDisease[entry.disease] = [];
        }
        byDisease[entry.disease].push(entry);
    });
    
    // Create a card for each disease
    for (const [disease, entries] of Object.entries(byDisease)) {
        const card = document.createElement('div');
        card.className = 'disease-card';
        
        card.innerHTML = `
            <h4>🔬 ${disease}</h4>
            <table class="forecast-table">
                <thead>
                    <tr>
                        <th>Date</th>
                        <th>Risk Level</th>
                        <th>Temperature</th>
                        <th>Humidity</th>
                        <th>Rainfall</th>
                        <th>Cloud Cover</th>
                        <th>Leaf Wetness</th>
                    </tr>
                </thead>
                <tbody>
                    ${entries.map(entry => `
                        <tr>
                            <td>${entry.date}</td>
                            <td class="risk-${entry.risk.toLowerCase()}">${entry.risk}</td>
                            <td>${entry.temperature.toFixed(1)}°C</td>
                            <td>${entry.humidity.toFixed(1)}%</td>
                            <td>${entry.rainfall.toFixed(1)}mm</td>
                            <td>${entry.cloud_cover.toFixed(1)}%</td>
                            <td>${entry.leaf_wetness.toFixed(1)}hrs</td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        `;
        
        container.appendChild(card);
    }
}
