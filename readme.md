    
# Smart Agriculture: Ground-Level Ozone Impact on Crop Yield 🌾

This project uses ground-level ozone, temperature, rainfall, and soil moisture to predict crop yield using a machine learning model.

## 🔧 Features
- 📍 Select location on map
- 🌦️ Weather data (temperature & rainfall) auto-fetched
- 🌫️ Manual ozone and soil moisture input
- 📊 Visualize ozone vs yield sensitivity

## 🛠️ Tech Stack
- Python
- Streamlit
- Scikit-learn
- Open-Meteo Weather API
- Streamlit-Folium (interactive maps)

## 📁 Project Structure
```
smart_agriculture_ozone/
├── app.py
├── utils.py
├── data/
│   ├── sample_data.py
│   └── real_data_explained.md
├── model/
│   └── model_training.py
├── requirements.txt
└── README.md
```

## 🚀 Run the App
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Train the model:
```bash
python model/model_training.py
```

3. Launch the Streamlit app:
```bash
streamlit run app.py
```

## 🧠 Model
- `LinearRegression` is trained on synthetic data generated using domain-based relationships.
- Easily replaceable with real-world datasets.

## 📌 Data Sources
See `data/real_data_explained.md` for guidance on integrating real datasets.

## 📬 Contact
For help or improvements, reach out or open a GitHub issue.

---
Developed with ❤️ to support smart agriculture & sustainable farming.
"# smart_agri" 
