import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

# ----------- Step 1: Load and Clean Data -----------
# Replace this with the correct path to your file
df = pd.read_csv("combined_potato_disease_data.csv")

# Rename columns for easier access
df = df.rename(columns={
    "Temperature (°C)": "Temperature",
    "Humidity (%)": "Humidity",
    "Rainfall (mm)": "Rainfall",
    "Cloud Cover (%)": "Cloud Cover",
    "Wind Speed (km/h)": "Wind Speed",
    "Leaf Wetness (hrs)": "Leaf Wetness",
    "Risk La": "Risk"
})

# Drop rows with missing Disease or Risk labels
df = df.dropna(subset=["Disease", "Risk"])

# Ensure labels are strings
df["Disease"] = df["Disease"].astype(str)
df["Risk"] = df["Risk"].astype(str)

# ----------- Step 2: Encode Categorical Columns -----------
le_disease = LabelEncoder()
le_risk = LabelEncoder()

df["Disease_enc"] = le_disease.fit_transform(df["Disease"])
df["Risk_enc"] = le_risk.fit_transform(df["Risk"])

# ----------- Step 3: Feature Selection -----------
features = ["Disease_enc", "Temperature", "Humidity", "Rainfall",
            "Cloud Cover", "Wind Speed", "Leaf Wetness"]
X = df[features]
y = df["Risk_enc"]

# ----------- Step 4: Train-Test Split -----------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----------- Step 5: Train Model -----------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ----------- Step 6: Evaluate Model -----------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n✅ Classification Report:")
print(classification_report(y_test, y_pred, target_names=le_risk.classes_))
print(f"\n🎯 Accuracy: {accuracy * 100:.2f}%")

# ----------- Step 7: Save Model & Encoders -----------
joblib.dump(model, "risk_predictor_model.pkl")
joblib.dump(le_disease, "disease_label_encoder.pkl")
joblib.dump(le_risk, "risk_label_encoder.pkl")
print("\n📦 Model and encoders saved successfully.")
