## File: model/model_training.py

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import sys
import os

# Add the parent directory to sys.path to resolve 'data' import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.sample_data import generate_data

# Generate synthetic data for potato crops
df = generate_data(300)
X = df[["ozone", "temp", "rain", "soil"]]
y = df["yield"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest for more accuracy
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model/yield_model.pkl")
print("✅ Potato yield model trained and saved to model/yield_model.pkl")
