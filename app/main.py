from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load("model.pkl")
VERSION = "v1"


@app.get("/health")
def health():
    return {"status": "ok", "version": VERSION}


@app.post("/predict")
def predict(features: dict):
    df = pd.DataFrame([features])
    return {"prediction": int(model.predict(df)[0]), "version": VERSION}
