import pandas as pd
import numpy as np

# List of diseases
diseases = [
    'Late Blight', 'Early Blight', 'Common Scab', 'Bacterial Wilt',
    'Black Scurf', 'Powdery Scab', 'Fusarium Dry Rot', 'Potato Leaf Roll Virus'
]

# Number of samples per disease
samples_per_disease = 100

# Synthetic dataset list
data = []

# Generate Low Risk data
for disease in diseases:
    for _ in range(samples_per_disease):
        temperature = np.random.uniform(18, 26)  # °C — moderate
        humidity = np.random.uniform(65, 85)     # % — not too high
        rainfall = np.random.uniform(0, 2)       # mm — dry conditions
        cloud_cover = np.random.uniform(10, 50)  # % — partial sun
        wind_speed = np.random.uniform(1, 4)     # km/h — gentle
        leaf_wetness = np.random.uniform(5, 10)  # hrs — dry

        data.append([
            disease, round(temperature, 2), round(humidity, 2), round(rainfall, 2),
            round(cloud_cover, 2), round(wind_speed, 2), round(leaf_wetness, 2), 'Low Risk'
        ])

# Convert to DataFrame
columns = [
    "Disease", "Temperature (°C)", "Humidity (%)", "Rainfall (mm)",
    "Cloud Cover (%)", "Wind Speed (km/h)", "Leaf Wetness (hrs)", "Risk La"
]
df = pd.DataFrame(data, columns=columns)

# Save to CSV
csv_file = "low_risk_potato_disease_data.csv"
df.to_csv(csv_file, index=False)

print(f"✅ Synthetic Low Risk data saved to: {csv_file}")
