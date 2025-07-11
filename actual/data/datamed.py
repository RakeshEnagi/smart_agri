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

# Generate Medium Risk data
for disease in diseases:
    for _ in range(samples_per_disease):
        temperature = np.random.uniform(22, 30)
        humidity = np.random.uniform(80, 92)
        rainfall = np.random.uniform(1, 5)
        cloud_cover = np.random.uniform(40, 75)
        wind_speed = np.random.uniform(2, 6)
        leaf_wetness = np.random.uniform(9, 13)

        data.append([
            disease, round(temperature, 2), round(humidity, 2), round(rainfall, 2),
            round(cloud_cover, 2), round(wind_speed, 2), round(leaf_wetness, 2), 'Medium Risk'
        ])

# Convert to DataFrame
columns = [
    "Disease", "Temperature (°C)", "Humidity (%)", "Rainfall (mm)",
    "Cloud Cover (%)", "Wind Speed (km/h)", "Leaf Wetness (hrs)", "Risk La"
]
df = pd.DataFrame(data, columns=columns)

# Save to CSV
csv_file = "medium_risk_potato_disease_data.csv"
df.to_csv(csv_file, index=False)

print(f"✅ Synthetic Medium Risk data saved to: {csv_file}")
