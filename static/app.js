document.getElementById('agriForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    const lat = document.getElementById('lat').value;
    const lon = document.getElementById('lon').value;
    
    document.getElementById('results').style.display = 'none';
    document.getElementById('weather').innerHTML = '';
    document.getElementById('recommendations').innerHTML = '';

    try {
        // Call backend API (Flask endpoint assumed)
        const res = await fetch(`/get_agri_data?lat=${lat}&lon=${lon}`);
        const data = await res.json();

        if (data.weather) {
            document.getElementById('weather').innerHTML = `
                <h3>Weather</h3>
                <ul>
                    <li>Temperature: <b>${data.weather.temp} °C</b></li>
                    <li>Humidity: <b>${data.weather.humidity} %</b></li>
                    <li>Rain: <b>${data.weather.rain} mm</b></li>
                    <li>Wind: <b>${data.weather.wind} km/h</b></li>
                </ul>
            `;
        }
        if (data.recommendations) {
            document.getElementById('recommendations').innerHTML = `
                <h3>Recommendations</h3>
                <p>${data.recommendations}</p>
            `;
        }
        document.getElementById('results').style.display = 'block';
    } catch (err) {
        document.getElementById('weather').innerHTML = '<span style="color:red">Error fetching data.</span>';
        document.getElementById('results').style.display = 'block';
    }
});
