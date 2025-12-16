from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Hypertension Risk API")

# CORS (obligatoire pour le frontend JS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Charger modèle et scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

feature_order = [
    'age', 'gender', 'height', 'weight',
    'ap_hi', 'ap_lo', 'cholesterol',
    'gluc', 'smoke', 'alco', 'active', 'BMI'
]

scaler_features = ['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'BMI']

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
