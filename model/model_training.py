import os
import sys

# ✅ Manually add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from data.sample_data import generate_data

# Generate synthetic dataset
df = generate_data(200)
X = df[["ozone", "temp", "rain", "soil"]]
y = df["yield"]

# Train a linear regression model
model = LinearRegression()
model.fit(X, y)

# Save the model
joblib.dump(model, "model/yield_model.pkl")
print("✅ Model trained and saved to model/yield_model.pkl")
