# predict_stress_level.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Sample synthetic data
data = {
    "ozone": [30, 45, 80, 60, 90, 70, 40, 85, 50, 95],
    "temp": [20, 25, 30, 28, 35, 27, 22, 33, 24, 36],
    "humidity": [60, 55, 40, 45, 30, 35, 65, 33, 50, 25],
    "soil": [0.3, 0.25, 0.15, 0.2, 0.1, 0.18, 0.28, 0.12, 0.22, 0.1],
    "color": ["Green", "Green", "Yellow", "Yellow", "Brown", "Yellow", "Green", "Brown", "Yellow", "Brown"],
    "symptom": ["None", "None", "Spots", "Spots", "Wilting", "Spots", "None", "Wilting", "Spots", "Wilting"],
    "stress": ["Low", "Low", "Moderate", "Moderate", "High", "Moderate", "Low", "High", "Moderate", "High"]
}

df = pd.DataFrame(data)

# One-hot encode categorical columns
df = pd.get_dummies(df, columns=["color", "symptom"])

X = df.drop("stress", axis=1)
y = df["stress"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, "model/stress_model.pkl")
print("✅ Crop stress model trained and saved as model/stress_model.pkl")
