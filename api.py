from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import numpy as np
import os

# Initialiser FastAPI
app = FastAPI(title="Hypertension Risk API")

# Middleware CORS pour autoriser le frontend JS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Monter le dossier 'frontend' pour servir HTML/CSS/JS
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# Charger modèle et scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

feature_order = [
    'age', 'gender', 'height', 'weight',
    'ap_hi', 'ap_lo', 'cholesterol',
    'gluc', 'smoke', 'alco', 'active', 'BMI'
]

scaler_features = ['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'BMI']

# Schéma de données pour l'API
class Patient(BaseModel):
    age: float
    gender: int
    height: float
    weight: float
    ap_hi: float
    ap_lo: float
    cholesterol: int
    gluc: int
    smoke: int
    alco: int
    active: int

# Endpoint de prédiction
@app.post("/predict")
def predict(patient: Patient):
    height_m = patient.height / 100
    BMI = patient.weight / (height_m ** 2)

    data = [
        patient.age, patient.gender, patient.height, patient.weight,
        patient.ap_hi, patient.ap_lo, patient.cholesterol,
        patient.gluc, patient.smoke, patient.alco, patient.active, BMI
    ]

    X = np.array(data).reshape(1, -1)
    X_scaled = X.copy()
    idx = [feature_order.index(f) for f in scaler_features]
    X_scaled[:, idx] = scaler.transform(X[:, idx])

    proba = float(model.predict_proba(X_scaled)[0][1])
    pred = int(proba >= 0.65)

    return {
        "probability": round(proba, 4),
        "probability_percent": round(proba * 100, 2),
        "diagnostic": "High risk ⚠️" if pred == 1 else "Low risk ✔️",
        "BMI": round(float(BMI), 2)
    }

# Servir la page d'accueil
@app.get("/")
def index():
    return FileResponse("frontend/index.html")

# Lancer le serveur avec le port Render
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
