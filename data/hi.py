import pandas as pd
import numpy as np

# Load the crop data
df = pd.read_csv("crop.csv")  # crop.csv already has headers: N,P,K,temperature,humidity,ph,rainfall,label

# Ozone ranges by crop label (min, max)
ozone_ranges = {
    'rice': (20, 39),
    'maize': (30, 49),
    'chickpea': (30, 49),
    'kidneybeans': (25, 44),
    'pigeonpeas': (30, 49),
    'mothbeans': (30, 49),
    'mungbean': (20, 39),
    'blackgram': (20, 39),
    'lentil': (30, 49),
    'pomegranate': (40, 59),
    'banana': (40, 59),
    'mango': (40, 59),
    'grapes': (30, 49),
    'watermelon': (40, 59),
    'muskmelon': (40, 59),
    'apple': (30, 49),
    'orange': (30, 49),
    'papaya': (40, 59),
    'coconut': (40, 59),
    'cotton': (25, 44),
    'jute': (30, 49),
    'coffee': (30, 49)
}

def assign_ozone(label):
    rng = ozone_ranges.get(label.lower())
    if rng:
        return np.random.randint(rng[0], rng[1]+1)
    return np.nan

# Add ozone column based on label (case-insensitive match)
df['ozone'] = df['label'].apply(assign_ozone)

# Save the updated DataFrame back to crop.csv
df.to_csv("crop.csv", index=False)

# Print unique ozone values for verification
print(df[['label', 'ozone']].drop_duplicates().sort_values('label'))
print(df.head())
