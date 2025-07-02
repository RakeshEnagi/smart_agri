import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

# Simulate training data for spray timing
np.random.seed(42)
n = 500
df = pd.DataFrame({
    "hour": np.random.randint(0, 24, n),
    "temp": np.random.uniform(15, 35, n),
    "humidity": np.random.uniform(30, 90, n),
    "wind": np.random.uniform(0, 10, n),
    "ozone": np.random.uniform(30, 100, n),
    "rain": np.random.uniform(0, 10, n)
})

# Label: 1 = good time to spray, 0 = not good
df["spray"] = ((df["rain"] < 2) & (df["wind"] < 4) & (df["humidity"] > 40) & (df["ozone"] < 70)).astype(int)

X = df[["hour", "temp", "humidity", "wind", "ozone", "rain"]]
y = df["spray"]

model = RandomForestClassifier()
model.fit(X, y)

# Save model
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/time_model.pkl")
print("✅ Fertilizer timing model trained and saved.")
