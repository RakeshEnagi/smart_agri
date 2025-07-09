import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv('data/potato_disease_environment_dataset.csv')

# Encode target labels (e.g., "Late Blight", etc.)
le = LabelEncoder()
df['Disease_Encoded'] = le.fit_transform(df['Disease'])

# Drop non-numeric columns (exclude 'Disease' and 'Risk Label' from features)
feature_cols = df.select_dtypes(include=['number']).drop(columns=['Disease_Encoded']).columns.tolist()

# Scale features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(df[feature_cols])

# Prepare sequences for LSTM (we'll use 5 timesteps per sample)
TIMESTEPS = 5
X = []
y = []

for i in range(len(features_scaled) - TIMESTEPS):
    X.append(features_scaled[i:i+TIMESTEPS])
    y.append(df['Disease_Encoded'].iloc[i + TIMESTEPS])

X = np.array(X)
y = np.array(y)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build LSTM model
model = Sequential()
model.add(LSTM(64, input_shape=(X.shape[1], X.shape[2]), return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(np.unique(y)), activation='softmax'))

# Compile model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.2)

# Save model
model.save('potato_disease_lstm_model.h5')
print("✅ Model trained and saved as 'potato_disease_lstm_model.h5'")
 