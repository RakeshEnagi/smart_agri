# Full Python code to generate 1600-row synthetic potato disease dataset based on environmental factors

import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Function to simulate environmental conditions for each disease
def simulate_disease_data(disease_name, temp_range, rh_range, rain_range, cloud_range, wind_range, leaf_wetness_range, label, rows=200):
    temperature = np.random.uniform(*temp_range, rows).round(1)
    humidity = np.random.uniform(*rh_range, rows).round(1)
    rainfall = np.random.uniform(*rain_range, rows).round(1)
    cloud_cover = np.random.uniform(*cloud_range, rows).round(1)
    wind_speed = np.random.uniform(*wind_range, rows).round(1)
    leaf_wetness = np.random.uniform(*leaf_wetness_range, rows).round(1)

    return pd.DataFrame({
        "Disease": [disease_name] * rows,
        "Temperature (°C)": temperature,
        "Humidity (%)": humidity,
        "Rainfall (mm)": rainfall,
        "Cloud Cover (%)": cloud_cover,
        "Wind Speed (km/h)": wind_speed,
        "Leaf Wetness (hrs)": leaf_wetness,
        "Risk Label": [label] * rows
    })

# Generate datasets for each disease
late_blight = simulate_disease_data(
    "Late Blight", (10, 25), (90, 100), (1, 15), (70, 100), (0, 10), (6, 14), "High Risk"
)

early_blight = simulate_disease_data(
    "Early Blight", (24, 32), (65, 85), (1, 10), (50, 90), (5, 15), (5, 10), "Moderate Risk"
)

common_scab = simulate_disease_data(
    "Common Scab", (20, 26), (50, 70), (0, 2), (10, 40), (5, 20), (0, 2), "High Risk"
)

bacterial_wilt = simulate_disease_data(
    "Bacterial Wilt", (25, 35), (85, 100), (5, 20), (60, 100), (0, 10), (7, 14), "High Risk"
)

black_scurf = simulate_disease_data(
    "Black Scurf", (10, 20), (80, 95), (2, 8), (60, 90), (0, 8), (6, 10), "Moderate Risk"
)

powdery_scab = simulate_disease_data(
    "Powdery Scab", (10, 15), (85, 100), (3, 10), (60, 100), (0, 10), (6, 10), "Moderate Risk"
)

fusarium_dry_rot = simulate_disease_data(
    "Fusarium Dry Rot", (15, 22), (80, 95), (2, 6), (50, 90), (0, 10), (4, 8), "Moderate Risk"
)

plrv = simulate_disease_data(
    "Potato Leaf Roll Virus", (24, 32), (50, 70), (0, 2), (40, 90), (5, 15), (0, 3), "High Risk"
)

# Combine all disease datasets
combined_df = pd.concat([
    late_blight,
    early_blight,
    common_scab,
    bacterial_wilt,
    black_scurf,
    powdery_scab,
    fusarium_dry_rot,
    plrv
], ignore_index=True)

# Preview the full dataset shape and sample
combined_df.shape, combined_df.sample(5)
combined_df.to_csv("potato_disease_environment_dataset.csv", index=False)
