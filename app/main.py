import os

from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load("model.pkl")
VERSION = "v1"
COMMIT = os.getenv("GIT_SHA", "dev")


@app.get("/health")
def health():
    return {"status": "ok", "version": VERSION, "commit": COMMIT}


@app.post("/predict")
def predict(features: dict):
    df = pd.DataFrame([features])
    return {"prediction": int(model.predict(df)[0]), "version": VERSION}
