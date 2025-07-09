import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the crop dataset
crop_df = pd.read_csv('../data/crop.csv')

# Debug: Check for missing values
print("Missing values per column:\n", crop_df.isnull().sum())

# Debug: Check unique labels
print("Unique crop labels:", crop_df['label'].unique())

# Features and target
X = crop_df[['N','P','K','temperature','humidity','ph','rainfall','ozone']]
y = crop_df['label']

# Debug: Show feature sample
print("Feature sample:")
print(X.head())

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save the model
joblib.dump(model, '../model/crop_model.pkl')

print('Crop recommendation model trained and saved as crop_model.pkl')
