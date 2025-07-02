import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

# Generate dummy data
np.random.seed(42)
n = 300

data = pd.DataFrame({
    "ozone": np.random.uniform(30, 100, n),
    "temp": np.random.uniform(15, 35, n),
    "soil": np.random.uniform(0.2, 0.5, n),
    "rain": np.random.uniform(0, 100, n),
    "humidity": np.random.uniform(20, 90, n),
})

# Simulated fertilizer types
fertilizers = ['15-15-15', '20-20-20', 'Calcium Nitrate', 'Compost', 'Manure']
data["fertilizer"] = np.random.choice(fertilizers, size=n)

# Train model
X = data.drop("fertilizer", axis=1)
y = data["fertilizer"]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model
joblib.dump(model, "model/fert_model.pkl")
print("✅ Fertilizer model trained and saved as model/fert_model.pkl")
